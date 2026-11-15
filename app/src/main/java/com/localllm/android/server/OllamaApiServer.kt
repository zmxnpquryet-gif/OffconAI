package com.localllm.android.server

import android.content.Context
import android.util.Log
import com.localllm.android.R
import com.localllm.android.engine.LlmEngine
import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.i18n.AppStrings
import com.localllm.android.model.GenerationSettings
import com.localllm.android.model.LlmModel
import com.localllm.android.model.ToolCallTemplate
import com.localllm.android.model.ToolCallTemplates
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.security.SecureRandom
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Hardened lightweight HTTP server running on port 11434 (Ollama & OpenAI protocols).
 * Delegates wire formatting to [HttpWireWriter] and payload mapping to [ApiPayloadMapper].
 */
class OllamaApiServer(
    private val context: Context,
    private val llmEngine: LlmEngine,
    private val getActiveModel: () -> LlmModel?,
    private val getAllModels: () -> List<LlmModel>,
    private val getSettings: () -> GenerationSettings,
    initialApiKey: String = generateSecureApiKey()
) {
    companion object {
        const val DEFAULT_PORT = 11434
        private const val TAG = "OllamaApiServer"
        private const val MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
        private const val MAX_HEADER_COUNT = 100
        private const val MAX_HEADER_PHASE_MS = 20_000L
        private const val MAX_BODY_PHASE_MS = 60_000L
        private const val MAX_CONCURRENT_GENERATIONS = 1
        private const val MAX_CONNECTIONS = 32
        private const val MAX_PREAUTH_CONNECTIONS = 8

        internal fun isAllowedCorsOrigin(origin: String?): Boolean {
            if (origin.isNullOrBlank()) return false
            return try {
                val uri = java.net.URI(origin.trim())
                val scheme = uri.scheme?.lowercase() ?: return false
                if (scheme != "http" && scheme != "https") return false
                if (uri.rawUserInfo != null) return false
                val host = uri.host?.lowercase() ?: return false
                host == "localhost" || host == "127.0.0.1" || host == "::1"
            } catch (_: Exception) {
                false
            }
        }

        internal fun readHttpLine(
            input: InputStream,
            deadlineNanos: Long = Long.MAX_VALUE,
            beforeRead: () -> Unit = {}
        ): String? {
            val bytes = ByteArrayOutputStream()
            while (true) {
                if (System.nanoTime() > deadlineNanos) {
                    throw java.net.SocketTimeoutException("HTTP header phase timed out")
                }
                beforeRead()
                val next = input.read()
                if (next == -1) {
                    if (bytes.size() == 0) return null
                    throw EOFException("Incomplete HTTP header")
                }
                if (next == '\n'.code) {
                    return bytes.toString(StandardCharsets.US_ASCII.name()).removeSuffix("\r")
                }
                require(bytes.size() < 8192) { "HTTP header line too long" }
                bytes.write(next)
            }
        }

        internal fun readHttpBody(
            input: InputStream,
            contentLength: Int,
            deadlineNanos: Long = Long.MAX_VALUE,
            beforeRead: () -> Unit = {}
        ): String {
            require(contentLength in 0..MAX_REQUEST_BODY_BYTES) { "Invalid body length" }
            val bytes = ByteArray(contentLength)
            var offset = 0
            while (offset < contentLength) {
                if (System.nanoTime() > deadlineNanos) {
                    throw java.net.SocketTimeoutException("HTTP body phase timed out")
                }
                beforeRead()
                val read = input.read(bytes, offset, minOf(contentLength - offset, 64 * 1024))
                if (read < 0) throw EOFException("Incomplete HTTP body")
                offset += read
            }
            return String(bytes, StandardCharsets.UTF_8)
        }

        fun generateSecureApiKey(): String {
            val random = SecureRandom()
            val bytes = ByteArray(18)
            random.nextBytes(bytes)
            return "sk-local-" + android.util.Base64.encodeToString(
                bytes,
                android.util.Base64.URL_SAFE or android.util.Base64.NO_WRAP or android.util.Base64.NO_PADDING
            )
        }

        internal fun checkAuthorization(headers: Map<String, String>, expectedKey: String): Boolean {
            val authHeader = headers["authorization"]
            if (!authHeader.isNullOrBlank()) {
                val token = authHeader.replaceFirst(Regex("^bearer\\s+", RegexOption.IGNORE_CASE), "").trim()
                if (constantTimeEquals(token, expectedKey)) return true
            }
            val xApiKey = headers["x-api-key"]
            if (!xApiKey.isNullOrBlank() && constantTimeEquals(xApiKey.trim(), expectedKey)) {
                return true
            }
            return false
        }

        internal fun constantTimeEquals(a: String, b: String): Boolean {
            if (a.isEmpty() || b.isEmpty()) return false
            return java.security.MessageDigest.isEqual(
                a.toByteArray(StandardCharsets.UTF_8),
                b.toByteArray(StandardCharsets.UTF_8)
            )
        }
    }

    private var serverSocket: ServerSocket? = null
    @Volatile
    private var isRunning = false
    private var serverJob: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val generationSlots = Semaphore(MAX_CONCURRENT_GENERATIONS)
    private val connectionSlots = Semaphore(MAX_CONNECTIONS)
    private val preAuthSlots = Semaphore(MAX_PREAUTH_CONNECTIONS)

    @Volatile
    var currentApiKey: String = initialApiKey
        private set

    fun setApiKey(key: String) {
        if (key.isNotBlank()) currentApiKey = key
    }

    var boundHost: String = "127.0.0.1"
        private set

    private val requestCounter = java.util.concurrent.atomic.AtomicInteger(0)
    val requestCount: Int get() = requestCounter.get()
    @Volatile
    var lastClientIp: String? = null
        private set
    @Volatile
    var lastEndpoint: String? = null
        private set
    var onRequestProcessed: ((Int) -> Unit)? = null

    fun isServerRunning(): Boolean = isRunning
    fun getPort(): Int = DEFAULT_PORT

    fun regenerateApiKey(): String {
        currentApiKey = generateSecureApiKey()
        return currentApiKey
    }

    fun start(onStatusChange: (Boolean, String?) -> Unit) {
        if (isRunning) {
            onStatusChange(true, AppStrings.get(R.string.aps_already, DEFAULT_PORT))
            return
        }

        try {
            val settings = getSettings()
            boundHost = settings.apiServerBindAddress.ifBlank { "127.0.0.1" }
            val bindAddress = InetAddress.getByName(boundHost)

            serverSocket = ServerSocket().apply {
                reuseAddress = true
                bind(InetSocketAddress(bindAddress, DEFAULT_PORT), 50)
            }
            isRunning = true

            val hostLabel = if (boundHost == "127.0.0.1") {
                AppStrings.get(R.string.aps_loopback)
            } else {
                AppStrings.get(R.string.aps_alliface, boundHost)
            }
            onStatusChange(true, AppStrings.get(R.string.aps_started, hostLabel, DEFAULT_PORT))

            serverJob = scope.launch {
                while (isRunning) {
                    try {
                        val clientSocket = serverSocket?.accept() ?: break
                        if (!connectionSlots.tryAcquire()) {
                            runCatching { clientSocket.close() }
                            Log.w(TAG, "Connection cap ($MAX_CONNECTIONS) hit, socket closed")
                            continue
                        }
                        val released = AtomicBoolean(false)
                        fun releaseConnection() {
                            runCatching { clientSocket.close() }
                            if (released.compareAndSet(false, true)) connectionSlots.release()
                        }
                        val clientJob = launch {
                            try {
                                handleClientSocket(clientSocket)
                            } finally {
                                releaseConnection()
                            }
                        }
                        // A parent can be cancelled before the child starts running;
                        // completion cleanup still closes the accepted socket and frees its slot.
                        clientJob.invokeOnCompletion { releaseConnection() }
                    } catch (e: Exception) {
                        if (e is CancellationException) throw e
                        if (!isRunning) break
                        Log.e(TAG, "Socket accept error", e)
                    }
                }
            }
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            isRunning = false
            Log.e(TAG, "Failed to start API server on $boundHost:$DEFAULT_PORT", e)
            onStatusChange(false, AppStrings.get(R.string.aps_start_fail, boundHost, DEFAULT_PORT, e.localizedMessage))
        }
    }

    fun stop(onStatusChange: (Boolean, String?) -> Unit) {
        isRunning = false
        try {
            serverSocket?.close()
            serverSocket = null
            serverJob?.cancel()
            serverJob = null
            onStatusChange(false, AppStrings.get(R.string.aps_stopped))
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            Log.e(TAG, "Failed to stop server", e)
            onStatusChange(false, AppStrings.get(R.string.aps_stop_err, e.localizedMessage))
        }
    }

    private suspend fun handleClientSocket(socket: Socket) {
        if (!preAuthSlots.tryAcquire()) {
            runCatching { socket.close() }
            return
        }
        var preAuthReleased = false
        try {
            socket.soTimeout = 30000
            val input = socket.getInputStream().buffered()
            val output = socket.getOutputStream()
            val headerDeadline = System.nanoTime() + MAX_HEADER_PHASE_MS * 1_000_000L
            val setHeaderReadTimeout = {
                val remainingNanos = headerDeadline - System.nanoTime()
                if (remainingNanos <= 0L) throw java.net.SocketTimeoutException("HTTP header phase timed out")
                socket.soTimeout = ((remainingNanos + 999_999L) / 1_000_000L).coerceAtLeast(1L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
            }

            val requestLine = readHttpLine(input, headerDeadline, setHeaderReadTimeout) ?: return
            val parts = requestLine.split(' ', limit = 3)
            if (parts.size != 3 || parts[0].isBlank() || !parts[0].all { it in 'A'..'Z' } ||
                !parts[1].startsWith('/') || parts[2] !in setOf("HTTP/1.0", "HTTP/1.1")
            ) {
                HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", "{\"error\":\"Malformed HTTP request line\"}", null)
                return
            }

            val method = parts[0].uppercase()
            val uri = parts[1]

            var contentLength = 0
            var contentLengthSeen = false
            var badContentLength = false
            val headers = mutableMapOf<String, String>()
            var headerCount = 0
            while (true) {
                val line = readHttpLine(input, headerDeadline, setHeaderReadTimeout)
                    ?: run {
                        HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", "{\"error\":\"Incomplete HTTP headers\"}", headers["origin"])
                        return
                    }
                if (line.isEmpty()) break
                headerCount++
                if (headerCount > MAX_HEADER_COUNT) {
                    HttpWireWriter.sendResponse(output, 431, "Request Header Fields Too Large", "application/json", "{\"error\":\"Too many headers\"}", headers["origin"])
                    return
                }
                val colonIdx = line.indexOf(':')
                if (colonIdx <= 0) {
                    HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", "{\"error\":\"Malformed HTTP header\"}", headers["origin"])
                    return
                }
                val key = line.substring(0, colonIdx).trim().lowercase()
                val value = line.substring(colonIdx + 1).trim()
                if (key.isBlank()) {
                    HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", "{\"error\":\"Malformed HTTP header\"}", headers["origin"])
                    return
                }
                headers[key] = value
                if (key == "content-length") {
                    val parsed = value.toLongOrNull()
                    when {
                        parsed == null || parsed < 0 -> badContentLength = true
                        contentLengthSeen && parsed != contentLength.toLong() -> badContentLength = true
                        parsed > Int.MAX_VALUE -> {
                            contentLength = MAX_REQUEST_BODY_BYTES + 1
                            contentLengthSeen = true
                        }
                        else -> {
                            contentLength = parsed.toInt()
                            contentLengthSeen = true
                        }
                    }
                }
            }

            val clientOrigin = headers["origin"]
            if (badContentLength) {
                HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", "{\"error\":\"Invalid Content-Length\"}", clientOrigin)
                return
            }
            if (headers.containsKey("transfer-encoding")) {
                HttpWireWriter.sendResponse(output, 411, "Length Required", "application/json", "{\"error\":\"Chunked request bodies are not supported\"}", clientOrigin)
                return
            }
            if (headers["content-encoding"]?.equals("identity", ignoreCase = true) == false) {
                HttpWireWriter.sendResponse(output, 415, "Unsupported Media Type", "application/json", "{\"error\":\"Compressed request bodies are not supported\"}", clientOrigin)
                return
            }
            if (method == "OPTIONS") {
                HttpWireWriter.sendCorsPreflight(output, clientOrigin)
                return
            }

            val settings = getSettings()
            if (settings.apiServerRequireAuth && uri != "/") {
                if (!checkAuthorization(headers, currentApiKey)) {
                    val unauthorizedJson = JSONObject()
                        .put("error", "Authentication failed: a valid API key is required. Send 'Authorization: Bearer <api_key>' or the 'X-API-Key' header.")
                        .toString()
                    HttpWireWriter.sendResponse(output, 401, "Unauthorized", "application/json", unauthorizedJson, clientOrigin)
                    return
                }
            }

            lastClientIp = socket.inetAddress?.hostAddress
            lastEndpoint = "$method $uri"
            onRequestProcessed?.invoke(requestCounter.incrementAndGet())

            if (contentLength !in 0..MAX_REQUEST_BODY_BYTES) {
                HttpWireWriter.sendResponse(output, 413, "Content Too Large", "application/json", "{\"error\":\"Request body too large\"}", clientOrigin)
                return
            }
            val expect = headers["expect"]
            if (expect != null && !expect.equals("100-continue", ignoreCase = true)) {
                HttpWireWriter.sendResponse(output, 417, "Expectation Failed", "application/json", "{\"error\":\"Unsupported expectation\"}", clientOrigin)
                return
            }
            if (expect != null && contentLength > 0) HttpWireWriter.sendContinue(output)
            val bodyDeadline = System.nanoTime() + MAX_BODY_PHASE_MS * 1_000_000L
            val setBodyReadTimeout = {
                val remainingNanos = bodyDeadline - System.nanoTime()
                if (remainingNanos <= 0L) throw java.net.SocketTimeoutException("HTTP body phase timed out")
                socket.soTimeout = ((remainingNanos + 999_999L) / 1_000_000L).coerceAtLeast(1L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
            }
            val body = readHttpBody(input, contentLength, bodyDeadline, setBodyReadTimeout)
            preAuthSlots.release()
            preAuthReleased = true

            routeRequest(method, uri, body, output, clientOrigin)
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            Log.e(TAG, "Error handling client request", e)
        } finally {
            if (!preAuthReleased) preAuthSlots.release()
            runCatching { socket.close() }
        }
    }

    private suspend fun routeRequest(
        method: String,
        uri: String,
        body: String,
        output: OutputStream,
        clientOrigin: String?
    ) {
        val path = uri.substringBefore('?')
        val knownPaths = setOf(
            "/", "/api/tags", "/v1/models", "/api/show", "/api/version",
            "/api/generate", "/api/chat", "/v1/chat/completions"
        )
        when {
            path == "/" && method == "GET" ->
                HttpWireWriter.sendResponse(output, 200, "OK", "text/plain", "Ollama is running on OffconAI\n", clientOrigin)
            (path == "/api/tags" || path == "/v1/models" || path == "/api/show" || path == "/api/version") && method == "GET" -> when (path) {
                "/api/tags" ->
                HttpWireWriter.sendResponse(output, 200, "OK", "application/json", ApiPayloadMapper.buildOllamaTagsJson(getAllModels(), getActiveModel()), clientOrigin)
                "/v1/models" ->
                    HttpWireWriter.sendResponse(output, 200, "OK", "application/json", ApiPayloadMapper.buildOpenAiModelsJson(getAllModels(), getActiveModel()), clientOrigin)
                "/api/show" ->
                    HttpWireWriter.sendResponse(output, 200, "OK", "application/json", ApiPayloadMapper.buildOllamaShowJson(getActiveModel(), getSettings()), clientOrigin)
                else ->
                    HttpWireWriter.sendResponse(output, 200, "OK", "application/json", JSONObject().put("version", "0.5.1").toString(), clientOrigin)
            }
            path == "/api/generate" && method == "POST" ->
                handleOllamaCompletion(output, body, clientOrigin, chatFormat = false)
            path == "/api/chat" && method == "POST" ->
                handleOllamaCompletion(output, body, clientOrigin, chatFormat = true)
            path == "/v1/chat/completions" && method == "POST" ->
                handleOpenAiChat(output, body, clientOrigin)
            path in knownPaths ->
                HttpWireWriter.sendResponse(output, 405, "Method Not Allowed", "application/json", "{\"error\":\"Method not allowed\"}", clientOrigin)
            else -> {
                val errorJson = JSONObject().put("error", "Endpoint not found: $uri").toString()
                HttpWireWriter.sendResponse(output, 404, "Not Found", "application/json", errorJson, clientOrigin)
            }
        }
    }

    private fun isBusyError(e: Throwable): Boolean =
        e.message?.contains("BUSY_INFERENCE") == true

    private suspend fun handleOllamaCompletion(
        output: OutputStream,
        body: String,
        origin: String?,
        chatFormat: Boolean
    ) {
        val json = try {
            JSONObject(body)
        } catch (_: Exception) {
            HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", JSONObject().put("error", "Request body must be a JSON object").toString(), origin)
            return
        }
        val isStream = json.optBoolean("stream", true)
        val modelName = getActiveModel()?.name ?: "local-model"

        val (history, prompt) = if (chatFormat) {
            ApiPayloadMapper.parseMessages(json.optJSONArray("messages") ?: JSONArray())
        } else {
            emptyList<Pair<String, String>>() to json.optString("prompt", "")
        }

        if (prompt.isBlank()) {
            val errMsg = if (chatFormat) "No user message to respond to" else "Prompt cannot be empty"
            HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", JSONObject().put("error", errMsg).toString(), origin)
            return
        }
        if (!llmEngine.isModelReady()) {
            val err = JSONObject().put("error", "No local model is selected or loaded. Load a model in the app first.").toString()
            HttpWireWriter.sendResponse(output, 503, "Service Unavailable", "application/json", err, origin)
            return
        }

        val request = ApiPayloadMapper.resolveRequestOptions(json, getSettings())
        if (!generationSlots.tryAcquire()) {
            val err = JSONObject().put("error", "Server is busy: another generation is already running.").toString()
            HttpWireWriter.sendResponse(output, 429, "Too Many Requests", "application/json", err, origin)
            return
        }

        fun buildOllamaLine(content: String, done: Boolean, error: String? = null, evalCount: Int? = null): String =
            JSONObject().apply {
                put("model", modelName)
                if (error == null) put("created_at", HttpWireWriter.isoTimestamp())
                if (chatFormat) {
                    put("message", JSONObject().put("role", "assistant").put("content", content))
                } else {
                    put("response", content)
                }
                put("done", done)
                if (evalCount != null) put("eval_count", evalCount)
                if (error != null) put("error", error)
            }.toString()

        try {
            if (isStream) {
                HttpWireWriter.sendStreamHeaders(output, "application/x-ndjson", origin)
                var failed = false
                val finalContent = StringBuilder()
                val emittedContent = StringBuilder()
                try {
                    llmEngine.streamGenerate(
                        prompt = prompt,
                        history = history,
                        settings = request.settings,
                        attachment = null,
                        numPredictOverride = request.numPredict
                    ).collect { chunk ->
                        if (chunk.isComplete) {
                            finalContent.setLength(0)
                            finalContent.append(chunk.currentContentText)
                        } else {
                            finalContent.append(chunk.contentDelta)
                        }
                        val delta = if (chunk.isComplete) {
                            ApiPayloadMapper.visibleStreamDelta(chunk.currentContentText, emittedContent.toString())
                        } else {
                            chunk.contentDelta.takeIf { it.isNotEmpty() }
                        }
                        if (delta != null) {
                            HttpWireWriter.sendChunk(output, buildOllamaLine(delta, done = false) + "\n")
                            emittedContent.append(delta)
                        }
                    }
                } catch (e: Exception) {
                    if (e is CancellationException) throw e
                    failed = true
                    val errText = if (isBusyError(e)) {
                        "Server is busy: another generation is already running."
                    } else {
                        "Inference failed: ${e.localizedMessage}"
                    }
                    HttpWireWriter.sendChunk(output, buildOllamaLine("", done = true, error = errText) + "\n")
                }
                if (!failed) {
                    val rest = ApiPayloadMapper.visibleStreamDelta(finalContent.toString(), emittedContent.toString())
                    if (rest != null) {
                        HttpWireWriter.sendChunk(output, buildOllamaLine(rest, done = false) + "\n")
                    }
                    HttpWireWriter.sendChunk(output, buildOllamaLine("", done = true) + "\n")
                }
                HttpWireWriter.endChunk(output)
            } else {
                val answer = StringBuilder()
                var totalTokens = 0
                try {
                    llmEngine.streamGenerate(
                        prompt = prompt,
                        history = history,
                        settings = request.settings,
                        attachment = null,
                        numPredictOverride = request.numPredict
                    ).collect { chunk ->
                        if (chunk.isComplete) {
                            answer.setLength(0)
                            answer.append(chunk.currentContentText)
                        } else {
                            answer.append(chunk.contentDelta)
                        }
                        totalTokens = chunk.totalTokens
                    }
                } catch (e: Exception) {
                    if (e is CancellationException) throw e
                    val status = if (isBusyError(e)) 429 else 500
                    val statusLabel = if (status == 429) "Too Many Requests" else "Internal Server Error"
                    val errText = if (status == 429) {
                        "Server is busy: another generation is already running."
                    } else {
                        "Inference failed: ${e.localizedMessage}"
                    }
                    HttpWireWriter.sendResponse(output, status, statusLabel, "application/json", JSONObject().put("error", errText).toString(), origin)
                    return
                }
                HttpWireWriter.sendResponse(output, 200, "OK", "application/json", buildOllamaLine(answer.toString(), done = true, evalCount = totalTokens), origin)
            }
        } finally {
            generationSlots.release()
        }
    }

    private suspend fun handleOpenAiChat(output: OutputStream, body: String, origin: String?) {
        val json = try {
            JSONObject(body)
        } catch (_: Exception) {
            val error = JSONObject().put("error", JSONObject().put("message", "Request body must be a JSON object")).toString()
            HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", error, origin)
            return
        }
        val isStream = json.optBoolean("stream", false)
        val activeModel = getActiveModel()
        val modelName = activeModel?.name ?: "local-model"
        val chatId = "chatcmpl-${System.currentTimeMillis()}"

        if (!llmEngine.isModelReady()) {
            val err = JSONObject().put("error", JSONObject().put("message", "No local model is selected or loaded.")).toString()
            HttpWireWriter.sendResponse(output, 503, "Service Unavailable", "application/json", err, origin)
            return
        }

        val (historyList, lastUserPrompt) = ApiPayloadMapper.parseMessages(json.optJSONArray("messages") ?: JSONArray())
        if (lastUserPrompt.isBlank()) {
            val err = JSONObject().put("error", JSONObject().put("message", "No user message to respond to")).toString()
            HttpWireWriter.sendResponse(output, 400, "Bad Request", "application/json", err, origin)
            return
        }

        val request = ApiPayloadMapper.resolveRequestOptions(json, getSettings())
        val offeredTools = if (request.settings.toolCallsEnabled) {
            ApiPayloadMapper.parseOpenAiTools(json)
        } else {
            emptyList()
        }
        val toolDialect = activeModel?.let { ToolCallTemplates.resolve(it, null) } ?: ToolCallTemplate.HERMES
        val toolsPrompt = if (offeredTools.isNotEmpty() && toolDialect != ToolCallTemplate.NONE) {
            McpToolProtocol.buildToolsPrompt("OpenAI Client", offeredTools, toolDialect)
        } else {
            null
        }

        if (!generationSlots.tryAcquire()) {
            val err = JSONObject().put("error", JSONObject().put("message", "Server is busy: another generation is already running.")).toString()
            HttpWireWriter.sendResponse(output, 429, "Too Many Requests", "application/json", err, origin)
            return
        }

        try {
            if (isStream) {
                HttpWireWriter.sendStreamHeaders(output, "text/event-stream", origin)
                val finalContentSb = StringBuilder()
                val emittedContent = StringBuilder()
                var emittedVisible = ""
                var lastToolScanRawLength = 0
                try {
                    llmEngine.streamGenerate(
                        prompt = lastUserPrompt,
                        history = historyList,
                        settings = request.settings,
                        attachment = null,
                        mcpToolsContext = toolsPrompt,
                        numPredictOverride = request.numPredict
                    ).collect { chunk ->
                        if (chunk.isComplete) {
                            finalContentSb.setLength(0)
                            finalContentSb.append(chunk.currentContentText)
                        } else {
                            finalContentSb.append(chunk.contentDelta)
                        }
                        if (toolsPrompt == null) {
                            val delta = if (chunk.isComplete) {
                                ApiPayloadMapper.visibleStreamDelta(chunk.currentContentText, emittedContent.toString())
                            } else {
                                chunk.contentDelta.takeIf { it.isNotEmpty() }
                            }
                            if (delta != null) {
                                HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, delta, null))
                                emittedContent.append(delta)
                            }
                        } else if (chunk.isComplete || finalContentSb.length - lastToolScanRawLength >= 128) {
                            val visible = McpToolProtocol.stableVisiblePrefix(
                                McpToolProtocol.visibleText(finalContentSb.toString(), toolDialect)
                            )
                            val delta = ApiPayloadMapper.visibleStreamDelta(visible, emittedVisible)
                            if (delta != null) {
                                HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, delta, null))
                                emittedVisible = visible
                            }
                            lastToolScanRawLength = finalContentSb.length
                        }
                    }
                } catch (e: Exception) {
                    if (e is CancellationException) throw e
                    val errText = if (isBusyError(e)) {
                        "Server is busy: another generation is already running."
                    } else {
                        "Inference failed: ${e.localizedMessage}"
                    }
                    HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, null, "stop", errText))
                    HttpWireWriter.sendChunk(output, "data: [DONE]\n\n")
                    HttpWireWriter.endChunk(output)
                    return
                }
                val finalContentText = finalContentSb.toString()
                val parsedBatch = if (toolsPrompt != null) {
                    McpToolProtocol.parseBatch(finalContentText, offeredTools, template = toolDialect)
                } else null
                if (parsedBatch != null && parsedBatch.calls.isNotEmpty()) {
                    HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiToolCallSseChunk(chatId, modelName, parsedBatch.calls))
                    HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, null, "tool_calls"))
                } else {
                    // Flush whatever visible text never made it into a delta (e.g. a
                    // chunk where the visible prefix check failed). Diff against the
                    // emitted visible text — never slice the raw text, which still
                    // contains reasoning/tool markup at different offsets.
                    if (toolsPrompt != null) {
                        val tail = McpToolProtocol.visibleText(finalContentText, toolDialect)
                        val rest = ApiPayloadMapper.visibleStreamDelta(tail, emittedVisible)
                        if (rest != null) {
                            HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, rest, null))
                        }
                    } else {
                        val rest = ApiPayloadMapper.visibleStreamDelta(finalContentText, emittedContent.toString())
                        if (rest != null) {
                            HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, rest, null))
                        }
                    }
                    HttpWireWriter.sendChunk(output, ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, null, "stop"))
                }
                HttpWireWriter.sendChunk(output, "data: [DONE]\n\n")
                HttpWireWriter.endChunk(output)
            } else {
                // Deltas accumulate O(delta) per token; the isComplete chunk
                // replaces them with the authoritative full text (finish() flush).
                val finalContentSb = StringBuilder()
                var totalTokens = 0
                try {
                    llmEngine.streamGenerate(
                        prompt = lastUserPrompt,
                        history = historyList,
                        settings = request.settings,
                        attachment = null,
                        mcpToolsContext = toolsPrompt,
                        numPredictOverride = request.numPredict
                    ).collect { chunk ->
                        if (chunk.isComplete) {
                            finalContentSb.setLength(0)
                            finalContentSb.append(chunk.currentContentText)
                        } else {
                            finalContentSb.append(chunk.contentDelta)
                        }
                        totalTokens = chunk.totalTokens
                    }
                } catch (e: Exception) {
                    if (e is CancellationException) throw e
                    val status = if (isBusyError(e)) 429 else 500
                    val statusLabel = if (status == 429) "Too Many Requests" else "Internal Server Error"
                    val errText = if (status == 429) {
                        "Server is busy: another generation is already running."
                    } else {
                        "Inference failed: ${e.localizedMessage}"
                    }
                    val err = JSONObject().put("error", JSONObject().put("message", errText)).toString()
                    HttpWireWriter.sendResponse(output, status, statusLabel, "application/json", err, origin)
                    return
                }
                val finalContentText = finalContentSb.toString()
                val parsedBatch = if (toolsPrompt != null) {
                    McpToolProtocol.parseBatch(finalContentText, offeredTools, template = toolDialect)
                } else null
                val responseBody = if (parsedBatch != null && parsedBatch.calls.isNotEmpty()) {
                    ApiPayloadMapper.buildOpenAiCompletionJson(chatId, modelName, parsedBatch.prose, totalTokens, parsedBatch.calls)
                } else {
                    val visible = if (toolsPrompt != null) {
                        McpToolProtocol.visibleText(finalContentText, toolDialect)
                    } else {
                        finalContentText
                    }
                    ApiPayloadMapper.buildOpenAiCompletionJson(chatId, modelName, visible, totalTokens)
                }
                HttpWireWriter.sendResponse(
                    output,
                    200,
                    "OK",
                    "application/json",
                    responseBody,
                    origin
                )
            }
        } finally {
            generationSlots.release()
        }
    }
}
