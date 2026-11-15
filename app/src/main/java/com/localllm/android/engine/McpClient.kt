package com.localllm.android.engine

import com.localllm.android.BuildConfig
import com.localllm.android.R
import com.localllm.android.i18n.AppStrings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.Call
import okhttp3.Callback
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.BufferedSource
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class McpTool(
    val name: String,
    val description: String,
    val parameters: String
)

data class McpConnectionResult(
    val isSuccess: Boolean,
    val serverName: String,
    val tools: List<McpTool>,
    val latencyMs: Long,
    val message: String
)

data class McpToolCallResult(
    val isSuccess: Boolean,
    val text: String,
    val message: String
)

/**
 * Real MCP client speaking JSON-RPC 2.0.
 *
 * Transports tried in order:
 *  1. Streamable HTTP — POST initialize/tools/list/tools/call directly to the URL,
 *     accepting both plain-JSON and SSE (`data:`) response bodies.
 *  2. Legacy SSE transport — GET event stream, read the `endpoint` event, then
 *     POST JSON-RPC messages there.
 *
 * Deliberately reports failure instead of inventing tools: a previous version
 * returned hard-coded tools with `isSuccess = true` even when the server was
 * unreachable, which misled both the UI and the model prompt.
 */
class McpClient(client: OkHttpClient? = null) {

    private val http: OkHttpClient = client ?: OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .followRedirects(true)
        // Never downgrade https to http on redirect: the bearer token must not
        // travel in cleartext because a server answered with a Location header.
        .followSslRedirects(false)
        .build()

    // The legacy transport keeps this GET stream open for the whole session,
    // so it needs a generous idle timeout (servers usually send SSE pings) and
    // must not share the short timeout used for one-shot HTTP calls.
    private val sseHttp: OkHttpClient = http.newBuilder()
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    private val idSeq = AtomicLong(0)

    @Volatile private var sessionId: String? = null
    @Volatile private var messageEndpoint: String? = null
    @Volatile private var serverName: String = ""
    @Volatile private var cachedTools: List<McpTool> = emptyList()
    @Volatile private var bearerToken: String? = null
    private val activeHttpCalls = ConcurrentHashMap<Call, Long>()

    // ---- legacy SSE transport state ----
    private val sseScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private data class PendingSse(val epoch: Long, val response: CompletableDeferred<JSONObject>)
    private data class StoppedSse(
        val epoch: Long?,
        val generation: Long,
        val job: Job?,
        val call: Call?,
        val response: Response?
    )
    private val sseStateLock = Any()
    private val sseEpochSequence = AtomicLong(0)
    private var connectionGeneration = 0L
    @Volatile private var activeSseEpoch: Long? = null
    private val ssePending = ConcurrentHashMap<String, PendingSse>()
    @Volatile private var sseJob: Job? = null
    @Volatile private var sseCall: Call? = null
    @Volatile private var sseResponse: Response? = null

    fun currentTools(): List<McpTool> = cachedTools

    fun isConnected(): Boolean = messageEndpoint != null

    fun disconnect() {
        stopSseTransport()
    }

    private fun stopSseTransport(): Long {
        val stopped = synchronized(sseStateLock) {
            val oldGeneration = connectionGeneration
            connectionGeneration++
            val oldEpoch = activeSseEpoch
            val oldJob = sseJob
            val oldCall = sseCall
            val oldResponse = sseResponse
            sessionId = null
            messageEndpoint = null
            serverName = ""
            cachedTools = emptyList()
            bearerToken = null
            activeSseEpoch = null
            sseJob = null
            sseCall = null
            sseResponse = null
            sseEpochSequence.incrementAndGet()
            StoppedSse(oldEpoch, oldGeneration, oldJob, oldCall, oldResponse)
        }
        stopped.job?.cancel()
        try {
            stopped.call?.cancel()
        } catch (_: Throwable) {
        }
        try {
            stopped.response?.close()
        } catch (_: Throwable) {
        }
        stopped.epoch?.let { failPendingSse("MCP SSE transport closed", it) }
        activeHttpCalls.entries
            .filter { it.value == stopped.generation }
            .forEach { it.key.cancel() }
        return stopped.generation + 1
    }

    private fun currentConnectionGeneration(): Long = synchronized(sseStateLock) { connectionGeneration }

    private fun isCurrentConnection(generation: Long): Boolean =
        synchronized(sseStateLock) { connectionGeneration == generation }

    private fun installConnection(generation: Long, endpoint: String, session: String?): Boolean =
        synchronized(sseStateLock) {
            if (connectionGeneration != generation) return@synchronized false
            sessionId = session
            messageEndpoint = endpoint
            true
        }

    private fun installTools(generation: Long, tools: List<McpTool>, name: String): Boolean =
        synchronized(sseStateLock) {
            if (connectionGeneration != generation) return@synchronized false
            cachedTools = tools.toList()
            serverName = name
            true
        }

    private fun disconnectIfCurrent(generation: Long) {
        stopSseTransportIfCurrent(generation)
    }

    private fun stopSseTransportIfCurrent(generation: Long) {
        val stopped = synchronized(sseStateLock) {
            if (connectionGeneration != generation) return
            val oldEpoch = activeSseEpoch
            val oldJob = sseJob
            val oldCall = sseCall
            val oldResponse = sseResponse
            connectionGeneration++
            sessionId = null
            messageEndpoint = null
            serverName = ""
            cachedTools = emptyList()
            bearerToken = null
            activeSseEpoch = null
            sseJob = null
            sseCall = null
            sseResponse = null
            sseEpochSequence.incrementAndGet()
            StoppedSse(oldEpoch, generation, oldJob, oldCall, oldResponse)
        }
        stopped.job?.cancel()
        runCatching { stopped.call?.cancel() }
        runCatching { stopped.response?.close() }
        stopped.epoch?.let { failPendingSse("MCP SSE transport closed", it) }
        activeHttpCalls.entries.filter { it.value == generation }.forEach { it.key.cancel() }
    }

    private fun failPendingSse(reason: String, epoch: Long) {
        val pending = synchronized(sseStateLock) {
            ssePending.entries.mapNotNull { entry ->
                if (entry.value.epoch == epoch && ssePending.remove(entry.key, entry.value)) entry.value else null
            }
        }
        // An IOException, not a CancellationException: a closed stream is a
        // transport failure, and must not silently cancel the caller's coroutine.
        pending.forEach { it.response.completeExceptionally(IOException(reason)) }
    }

    /**
     * @param bearerToken optional `Authorization: Bearer` credential. It is only ever sent
     *   to the configured origin (SSE endpoints on another origin are rejected, and OkHttp
     *   drops Authorization on cross-host redirects), and never over plain HTTP to a
     *   public host.
     */
    suspend fun connectServer(url: String, bearerToken: String? = null): McpConnectionResult = withContext(Dispatchers.IO) {
        val generation = stopSseTransport()
        val clean = url.trim()
        if (clean.isBlank()) {
            return@withContext McpConnectionResult(false, "Unknown", emptyList(), 0, AppStrings.get(R.string.mcp_enter_url))
        }
        val parsed = try {
            clean.toHttpUrlOrNull()
        } catch (_: Throwable) {
            null
        }
        if (parsed == null || (parsed.scheme != "http" && parsed.scheme != "https")) {
            return@withContext McpConnectionResult(false, "Unknown", emptyList(), 0, AppStrings.get(R.string.mcp_bad_url))
        }
        val token = bearerToken?.trim()?.takeIf { it.isNotEmpty() }
        if (token != null && !isTokenSafeTransport(parsed)) {
            return@withContext McpConnectionResult(
                false, parsed.host, emptyList(), 0,
                "Refusing to send a bearer token over plain HTTP to a public host. Use an https:// URL."
            )
        }
        synchronized(sseStateLock) {
            if (connectionGeneration != generation) {
                return@withContext McpConnectionResult(false, parsed.host, emptyList(), 0, "MCP connection was cancelled")
            }
            this@McpClient.bearerToken = token
        }

        val start = System.currentTimeMillis()
        val elapsed = { System.currentTimeMillis() - start }
        val initReq = rpcRequest(
            "initialize",
            JSONObject()
                .put("protocolVersion", "2024-11-05")
                .put("capabilities", JSONObject())
                .put("clientInfo", JSONObject().put("name", "OffconAI-Android").put("version", BuildConfig.VERSION_NAME))
        )

        // 1) Streamable HTTP: initialize directly on the URL.
        val direct = postJson(clean, initReq.json, null, initReq.id, generation)
        if (!isCurrentConnection(generation)) {
            return@withContext McpConnectionResult(false, parsed.host, emptyList(), elapsed(), "MCP connection was cancelled")
        }
        if (direct != null && direct.code in 200..299 && direct.bodyJson?.optJSONObject("result") != null) {
            val name = parseServerName(direct.bodyJson) ?: parsed.host
            if (!installConnection(generation, clean, direct.sessionId)) {
                return@withContext McpConnectionResult(false, parsed.host, emptyList(), elapsed(), "MCP connection was cancelled")
            }
            return@withContext try {
                val initialized = postJson(clean, rpcNotification("notifications/initialized", JSONObject()), sessionId, expectedGeneration = generation)
                if (initialized == null || initialized.code !in 200..299) {
                    throw IllegalStateException("MCP notifications/initialized was rejected")
                }
                val tools = fetchTools(clean, generation)
                if (!installTools(generation, tools, name) || !isCurrentConnection(generation)) {
                    McpConnectionResult(false, name, emptyList(), elapsed(), "MCP connection was cancelled")
                } else {
                    McpConnectionResult(true, name, tools, elapsed(), AppStrings.get(R.string.mcp_connected, tools.size))
                }
            } catch (e: CancellationException) {
                disconnectIfCurrent(generation)
                throw e
            } catch (e: Exception) {
                disconnectIfCurrent(generation)
                McpConnectionResult(false, name, emptyList(), elapsed(), AppStrings.get(R.string.mcp_tools_fail, e.message))
            }
        }

        if (direct?.bodyJson?.has("error") == true) {
            val message = direct.bodyJson.optJSONObject("error")?.optString("message") ?: AppStrings.get(R.string.eng_unknown_err)
            return@withContext McpConnectionResult(false, parsed.host, emptyList(), elapsed(), message)
        }

        // 2) Legacy SSE transport handshake: same initialize/initialized/tools
        // sequence as the Streamable path, including the session id the server
        // returns. Skipping either step breaks spec-strict MCP servers.
        val sseEndpoint = openSseEndpoint(clean, generation)
        if (!isCurrentConnection(generation)) {
            return@withContext McpConnectionResult(false, parsed.host, emptyList(), elapsed(), "MCP connection was cancelled")
        }
        if (sseEndpoint != null) {
            val init = postJson(sseEndpoint, initReq.json, null, initReq.id, generation)
            if (!isCurrentConnection(generation)) {
                return@withContext McpConnectionResult(false, parsed.host, emptyList(), elapsed(), "MCP connection was cancelled")
            }
            if (init != null && init.code in 200..299 && init.bodyJson?.optJSONObject("result") != null) {
                val name = parseServerName(init.bodyJson) ?: parsed.host
                if (!installConnection(generation, sseEndpoint, init.sessionId)) {
                    return@withContext McpConnectionResult(false, parsed.host, emptyList(), elapsed(), "MCP connection was cancelled")
                }
                return@withContext try {
                    val initialized = postJson(sseEndpoint, rpcNotification("notifications/initialized", JSONObject()), sessionId, expectedGeneration = generation)
                    if (initialized == null || initialized.code !in 200..299) {
                        throw IllegalStateException("MCP notifications/initialized was rejected")
                    }
                    val tools = fetchTools(sseEndpoint, generation)
                    if (!installTools(generation, tools, name) || !isCurrentConnection(generation)) {
                        McpConnectionResult(false, name, emptyList(), elapsed(), "MCP connection was cancelled")
                    } else {
                        McpConnectionResult(true, name, tools, elapsed(), AppStrings.get(R.string.mcp_connected_sse, tools.size))
                    }
                } catch (e: CancellationException) {
                    disconnectIfCurrent(generation)
                    throw e
                } catch (e: Exception) {
                    disconnectIfCurrent(generation)
                    McpConnectionResult(false, name, emptyList(), elapsed(), AppStrings.get(R.string.mcp_tools_fail, e.message))
                }
            }
        }

        val hint = if (direct != null) "HTTP ${direct.code}" else AppStrings.get(R.string.mcp_conn_fail)
        McpConnectionResult(
            false, parsed.host, emptyList(), elapsed(),
            AppStrings.get(R.string.mcp_handshake, hint)
        )
    }

    suspend fun callTool(name: String, argumentsJson: String = "{}"): McpToolCallResult = withContext(Dispatchers.IO) {
        val generation = currentConnectionGeneration()
        val endpoint = messageEndpoint
            ?: return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_not_connected))
        val cleanName = name.trim()
        if (cleanName.isEmpty() || cleanName.length > MAX_TOOL_NAME_CHARS || cachedTools.none { it.name == cleanName }) {
            return@withContext McpToolCallResult(false, "", "The requested MCP tool is not in the server's advertised tool list.")
        }
        if (argumentsJson.length > MAX_TOOL_ARGUMENT_CHARS ||
            argumentsJson.toByteArray(StandardCharsets.UTF_8).size > MAX_TOOL_ARGUMENT_BYTES
        ) {
            return@withContext McpToolCallResult(false, "", "Tool arguments exceed the local size limit.")
        }
        // Malformed arguments are reported, not silently replaced by {}: calling a tool
        // with arguments the model never wrote would be a wrong action, not a soft error.
        val args = try {
            JSONObject(argumentsJson.ifBlank { "{}" })
        } catch (_: Exception) {
            return@withContext McpToolCallResult(false, "", "Tool arguments are not a valid JSON object.")
        }
        val call = rpcRequest("tools/call", JSONObject().put("name", cleanName).put("arguments", args))
        val resp = postJson(endpoint, call.json, sessionId, call.id, generation)
            ?: return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_net))
        if (resp.code !in 200..299) {
            return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_http, resp.code))
        }
        val body = resp.bodyJson
            ?: return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_parse))
        if (body.has("error") && !body.isNull("error")) {
            val msg = body.optJSONObject("error")?.optString("message") ?: AppStrings.get(R.string.eng_unknown_err)
            return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_tool_error, msg))
        }
        val result = body.optJSONObject("result")
            ?: return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_parse))
        val content = result.optJSONArray("content")
            ?: return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_parse))
        if (result.has("isError") && result.opt("isError") !is Boolean) {
            return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_parse))
        }
        val text = StringBuilder()
        var truncated = false
        var malformedContent = false
        val truncationMarker = "\n[tool output truncated]"
        fun appendBounded(value: String) {
            val room = MAX_TOOL_RESULT_CHARS - truncationMarker.length - text.length
            if (room <= 0) {
                truncated = truncated || value.isNotEmpty()
                return
            }
            if (value.length > room) truncated = true
            text.append(value.take(room))
        }
        for (i in 0 until content.length()) {
            val item = content.optJSONObject(i)
            if (item == null) {
                malformedContent = true
                appendBounded(if (text.isEmpty()) "[invalid MCP content block omitted]" else "\n[invalid MCP content block omitted]")
                if (truncated) break
                continue
            }
            when (val type = item.optString("type")) {
                "text" -> {
                    val itemText = item.opt("text") as? String
                        ?: return@withContext McpToolCallResult(false, "", AppStrings.get(R.string.mcp_call_parse))
                    appendBounded(if (text.isEmpty()) itemText else "\n$itemText")
                }
                else -> {
                    if (type.isBlank()) malformedContent = true
                    appendBounded(
                        if (type.isBlank()) "[invalid MCP content type omitted]"
                        else "[$type content omitted: only text is supported]"
                    )
                }
            }
            if (truncated) break
        }
        if (truncated) {
            text.append(truncationMarker)
        }
        val failed = result.optBoolean("isError", false)
        val success = !failed && !malformedContent
        McpToolCallResult(
            success,
            text.toString(),
            when {
                failed -> AppStrings.get(R.string.mcp_tool_error, "Tool returned an error result")
                malformedContent -> AppStrings.get(R.string.mcp_tool_error, "Tool returned malformed content")
                text.isEmpty() -> AppStrings.get(R.string.mcp_empty)
                else -> AppStrings.get(R.string.mcp_exec_ok)
            }
        )
    }

    companion object {
        internal const val MAX_TOOL_SCHEMA_CHARS = 2000
        internal const val MAX_TOOLS_CONTEXT_CHARS = 4000
        internal const val MAX_TOOL_COUNT = 64
        internal const val MAX_TOOL_NAME_CHARS = 128
        internal const val MAX_TOOL_DESCRIPTION_CHARS = 1000
        internal const val MAX_TOOL_RESULT_CHARS = 4000
        internal const val MAX_TOOL_ARGUMENT_CHARS = 256 * 1024
        internal const val MAX_TOOL_ARGUMENT_BYTES = 256 * 1024
        internal const val MAX_TOOL_PAGES = 8
        internal const val MAX_RESPONSE_BYTES = 8L * 1024 * 1024
        internal const val MAX_SSE_LINE_BYTES = 256L * 1024
        internal const val MAX_SSE_EVENT_BYTES = 1024L * 1024
        internal const val MAX_SSE_ENDPOINT_CHARS = 4096
        /** POST may be only an acknowledgement; the real reply arrives over SSE. */
        internal const val SSE_RESPONSE_TIMEOUT_MS = 60_000L
    }

    // ---- JSON-RPC plumbing ----

    private data class JsonResponse(val code: Int, val sessionId: String?, val bodyJson: JSONObject?)

    private data class SseEndpointStream(
        val endpoint: String,
        val call: Call,
        val response: Response,
        val source: BufferedSource
    )

    private data class RpcPayload(val json: String, val id: Long)

    private fun rpcRequest(method: String, params: JSONObject): RpcPayload {
        val id = idSeq.incrementAndGet()
        val json = JSONObject()
            .put("jsonrpc", "2.0")
            .put("id", id)
            .put("method", method)
            .put("params", params)
            .toString()
        return RpcPayload(json, id)
    }

    private fun rpcNotification(method: String, params: JSONObject): String =
        JSONObject()
            .put("jsonrpc", "2.0")
            .put("method", method)
            .put("params", params)
            .toString()

    /**
     * POSTs JSON-RPC without ever blocking a thread: cancellation (the user hitting
     * Stop mid tool call) aborts the in-flight HTTP call instead of leaving an orphan
     * request running against the tool server for up to 30 seconds.
     */
    private suspend fun postJson(
        url: String,
        payload: String,
        session: String?,
        expectedId: Long? = null,
        expectedGeneration: Long? = null
    ): JsonResponse? {
        if (expectedGeneration != null && !isCurrentConnection(expectedGeneration)) return null
        val builder = Request.Builder()
            .url(url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
        if (!session.isNullOrBlank()) builder.header("Mcp-Session-Id", session)
        authHeaderFor(url)?.let { builder.header("Authorization", it) }
        builder.post(payload.toRequestBody("application/json; charset=utf-8".toMediaType()))

        // Legacy SSE servers acknowledge the POST with 202 and deliver the
        // JSON-RPC response later on the still-open event stream. Register the
        // waiter before sending so an early response cannot be lost.
        val pendingKey = expectedId?.toString()
        val pending = if (pendingKey != null) synchronized(sseStateLock) {
            val epoch = activeSseEpoch
            if (epoch == null || sseResponse == null) {
                null
            } else {
                PendingSse(epoch, CompletableDeferred()).also { ssePending[pendingKey] = it }
            }
        } else null

        val call = http.newCall(builder.build())
        val requestGeneration = expectedGeneration ?: currentConnectionGeneration()
        activeHttpCalls[call] = requestGeneration
        return try {
            if (!isCurrentConnection(requestGeneration)) {
                call.cancel()
                return null
            }
            val response = call.awaitJsonResponse(expectedId)
            if (!isCurrentConnection(requestGeneration)) return null
            var responseBody = response.bodyJson
            if (responseBody == null && pending != null && response.code in 200..299) {
                responseBody = withTimeoutOrNull(SSE_RESPONSE_TIMEOUT_MS) { pending.response.await() }
            }
            response.copy(bodyJson = responseBody)
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            null
        } finally {
            activeHttpCalls.remove(call)
            if (pendingKey != null && pending != null) {
                synchronized(sseStateLock) { ssePending.remove(pendingKey, pending) }
            }
        }
    }

    /** Reads the bounded response on OkHttp's callback thread while retaining a
     * cancellation hook to `Call.cancel()` for the entire body read. */
    private suspend fun Call.awaitJsonResponse(expectedId: Long?): JsonResponse = suspendCancellableCoroutine { cont ->
        cont.invokeOnCancellation {
            try {
                cancel()
            } catch (_: Throwable) {
            }
        }
        enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!cont.isCompleted) cont.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                val parsed = try {
                    response.use { resp ->
                        JsonResponse(
                            code = resp.code,
                            sessionId = resp.header("Mcp-Session-Id"),
                            bodyJson = readResponseMessage(resp, expectedId)
                        )
                    }
                } catch (e: Exception) {
                    if (!cont.isCompleted) cont.resumeWithException(e)
                    return
                }
                if (!cont.isCompleted) cont.resume(parsed)
            }
        })
    }

    /** Reads JSON bodies to EOF, but returns as soon as a matching message arrives
     * on an SSE response so a valid long-lived stream cannot stall the POST. */
    private fun readResponseMessage(response: Response, expectedId: Long?): JSONObject? {
        val source = response.body?.source() ?: return null
        val contentType = response.header("Content-Type") ?: ""
        if (contentType.contains("text/event-stream", ignoreCase = true)) {
            if (expectedId == null) return null
            val eventData = StringBuilder()
            var eventBytes = 0L
            var totalBytes = 0L
            fun parseCurrentEvent(): JSONObject? = extractJsonMessage(
                eventData.toString().lineSequence().joinToString("\n") { "data: $it" },
                expectedId
            )
            while (true) {
                val line = readBoundedLine(source) ?: break
                totalBytes += line.toByteArray(StandardCharsets.UTF_8).size + 1L
                if (totalBytes > MAX_RESPONSE_BYTES) throw IOException("MCP response exceeds $MAX_RESPONSE_BYTES bytes")
                if (line.isEmpty()) {
                    val parsed = parseCurrentEvent()
                    if (parsed != null) return parsed
                    eventData.setLength(0)
                    eventBytes = 0L
                    continue
                }
                if (line.startsWith(':')) continue
                val colon = line.indexOf(':')
                val field = if (colon < 0) line else line.substring(0, colon)
                if (field == "data") {
                    var value = if (colon < 0) "" else line.substring(colon + 1)
                    if (value.startsWith(' ')) value = value.substring(1)
                    val payloadBytes = value.toByteArray(StandardCharsets.UTF_8).size.toLong()
                    val separatorBytes = if (eventData.isNotEmpty()) 1L else 0L
                    eventBytes += separatorBytes + payloadBytes
                    if (eventBytes > MAX_SSE_EVENT_BYTES) throw IOException("MCP SSE event exceeds $MAX_SSE_EVENT_BYTES bytes")
                    if (eventData.isNotEmpty()) eventData.append('\n')
                    eventData.append(value)
                }
            }
            return if (eventData.isEmpty()) null else parseCurrentEvent()
        }
        if (source.request(MAX_RESPONSE_BYTES + 1)) {
            throw IOException("MCP response exceeds $MAX_RESPONSE_BYTES bytes")
        }
        return extractJsonMessage(source.readUtf8(), expectedId)
    }

    /** Handles both plain-JSON and SSE (`data: {...}`) response bodies. */
    internal fun extractJsonMessage(raw: String?, expectedId: Long? = null): JSONObject? {
        if (raw.isNullOrBlank()) return null
        val trimmed = raw.trim()
        if (trimmed.startsWith("{")) {
            return try {
                JSONObject(trimmed).takeIf { isValidJsonRpcResponse(it, expectedId) }
            } catch (_: Exception) {
                null
            }
        }
        val data = StringBuilder()
        fun consumeEvent(): JSONObject? {
            if (data.isEmpty()) return null
            val payload = data.toString().trim()
            data.setLength(0)
            if (payload.isEmpty() || payload == "[DONE]") return null
            return try {
                JSONObject(payload).takeIf { isValidJsonRpcResponse(it, expectedId) }
            } catch (_: Exception) {
                null
            }
        }
        for (line in trimmed.lineSequence()) {
            if (line.isEmpty()) {
                consumeEvent()?.let { return it }
                continue
            }
            if (line.startsWith(':')) continue
            val colon = line.indexOf(':')
            val field = if (colon < 0) line else line.substring(0, colon)
            if (field == "data") {
                var value = if (colon < 0) "" else line.substring(colon + 1)
                if (value.startsWith(' ')) value = value.substring(1)
                if (data.isNotEmpty()) data.append('\n')
                data.append(value)
            }
        }
        // Some servers omit the terminating blank line on a finite HTTP response.
        return consumeEvent()
    }

    private fun isValidJsonRpcResponse(obj: JSONObject, expectedId: Long?): Boolean {
        if (obj.opt("jsonrpc") != "2.0") return false
        val id = obj.opt("id") ?: return false
        if (id == JSONObject.NULL) return false
        if (expectedId != null) {
            val number = id as? Number ?: return false
            val asLong = number.toLong()
            if (!number.toDouble().isFinite() || number.toDouble() != asLong.toDouble() || asLong != expectedId) return false
        } else if (id !is Number && id !is String) {
            return false
        }

        val hasResult = obj.has("result")
        val hasError = obj.has("error")
        if (hasResult == hasError) return false
        if (hasError) {
            val error = obj.optJSONObject("error") ?: return false
            val code = error.opt("code") as? Number ?: return false
            val codeLong = code.toLong()
            if (!code.toDouble().isFinite() || code.toDouble() != codeLong.toDouble()) return false
            if (error.opt("message") !is String) return false
        }
        return true
    }

    private fun numericResponseId(obj: JSONObject): Long? {
        val number = obj.opt("id") as? Number ?: return null
        val value = number.toDouble()
        if (!value.isFinite()) return null
        val longValue = number.toLong()
        return longValue.takeIf { value == it.toDouble() }
    }

    /** readUtf8Line() with a hard bound: a newline-free stream must not OOM us. */
    private fun readBoundedLine(source: BufferedSource): String? {
        var want = 64L
        while (true) {
            if (!source.request(want)) {
                if (source.buffer.size == 0L) return null
                return source.readUtf8().trimEnd('\n', '\r')
            }
            val nl = source.buffer.indexOf('\n'.code.toByte())
            if (nl >= 0) {
                // The whole line may already be buffered in one socket read:
                // enforce the cap on the line length, not just on `want`.
                if (nl > MAX_SSE_LINE_BYTES) {
                    throw IOException("SSE line exceeds $MAX_SSE_LINE_BYTES bytes")
                }
                return source.readUtf8(nl + 1).trimEnd('\n', '\r')
            }
            if (source.buffer.size > MAX_SSE_LINE_BYTES) {
                throw IOException("SSE line exceeds $MAX_SSE_LINE_BYTES bytes")
            }
            want = minOf(want * 2, MAX_SSE_LINE_BYTES + 1)
        }
    }

    private fun parseServerName(envelope: JSONObject?): String? {
        val info = envelope?.optJSONObject("result")?.optJSONObject("serverInfo") ?: return null
        val name = info.optString("name").ifBlank { return null }
        val version = info.optString("version")
        return if (version.isBlank()) name else "$name $version"
    }

    private suspend fun fetchTools(endpoint: String, generation: Long): List<McpTool> {
        val out = mutableListOf<McpTool>()
        val names = HashSet<String>()
        val cursors = HashSet<String>()
        var cursor: String? = null

        repeat(MAX_TOOL_PAGES) {
            if (!isCurrentConnection(generation)) throw IllegalStateException("MCP connection was cancelled")
            val params = JSONObject().apply { if (cursor != null) put("cursor", cursor) }
            val req = rpcRequest("tools/list", params)
            val response = postJson(endpoint, req.json, sessionId, req.id, generation)
                ?: throw IllegalStateException(AppStrings.get(R.string.mcp_list_net))
            if (!isCurrentConnection(generation)) throw IllegalStateException("MCP connection was cancelled")
            if (response.code !in 200..299) throw IllegalStateException(AppStrings.get(R.string.mcp_list_http, response.code))
            val envelope = response.bodyJson ?: throw IllegalStateException(AppStrings.get(R.string.mcp_list_parse))
            if (envelope.has("error")) {
                val msg = envelope.optJSONObject("error")?.optString("message") ?: AppStrings.get(R.string.eng_unknown_err)
                throw IllegalStateException(AppStrings.get(R.string.mcp_server_error, msg))
            }
            val result = envelope.optJSONObject("result")
                ?: throw IllegalStateException(AppStrings.get(R.string.mcp_list_parse))
            val tools = result.optJSONArray("tools")
                ?: throw IllegalStateException(AppStrings.get(R.string.mcp_list_parse))
            if (tools.length() > MAX_TOOL_COUNT - out.size) {
                throw IllegalStateException("MCP server advertised more than $MAX_TOOL_COUNT tools")
            }
            for (i in 0 until tools.length()) {
                val item = tools.optJSONObject(i)
                    ?: throw IllegalStateException("MCP tools/list contained a non-object tool")
                val toolName = (item.opt("name") as? String)?.trim()
                    ?: throw IllegalStateException("MCP tool is missing a string name")
                if (toolName.isEmpty() || toolName.length > MAX_TOOL_NAME_CHARS) {
                    throw IllegalStateException("MCP tool name is empty or too long")
                }
                if (!names.add(toolName)) throw IllegalStateException("MCP server advertised duplicate tool name: $toolName")
                val rawDescription = item.opt("description")
                val description = when {
                    rawDescription == null || rawDescription === JSONObject.NULL -> ""
                    rawDescription is String -> rawDescription.take(MAX_TOOL_DESCRIPTION_CHARS)
                    else -> throw IllegalStateException("MCP tool description must be a string")
                }
                val schema = item.optJSONObject("inputSchema")
                    ?: throw IllegalStateException("MCP tool inputSchema must be an object")
                val schemaText = schema.toString().let { serialized ->
                    if (serialized.length <= MAX_TOOL_SCHEMA_CHARS) serialized else {
                        JSONObject()
                            .put("type", "object")
                            .put("description", "Input schema omitted because it exceeds the local size limit.")
                            .toString()
                    }
                }
                out.add(McpTool(toolName, description, schemaText))
            }

            if (!result.has("nextCursor") || result.isNull("nextCursor")) return out.toList()
            val next = result.opt("nextCursor") as? String
                ?: throw IllegalStateException("MCP tools/list nextCursor must be a string")
            if (next.isBlank() || next.length > MAX_SSE_ENDPOINT_CHARS || !cursors.add(next) || next == cursor) {
                throw IllegalStateException("MCP tools/list returned an invalid or repeated cursor")
            }
            cursor = next
        }
        throw IllegalStateException("MCP tools/list exceeded the $MAX_TOOL_PAGES page limit")
    }

    /**
     * Legacy SSE transport handshake. The GET stream is deliberately kept open:
     * MCP must continue reading it to receive responses to every subsequent POST.
     * Returns the POST message endpoint, or null.
     */
    private suspend fun openSseEndpoint(baseUrl: String, generation: Long): String? {
        var keepOpen = false
        var openedStream: SseEndpointStream? = null
        val reqBuilder = Request.Builder().url(baseUrl).header("Accept", "text/event-stream").get()
        authHeaderFor(baseUrl)?.let { reqBuilder.header("Authorization", it) }
        val call = sseHttp.newCall(reqBuilder.build())
        activeHttpCalls[call] = generation
        return try {
            val stream = call.awaitSseEndpoint(baseUrl) ?: return null
            openedStream = stream
            if (!startSsePump(stream.source, stream.response, stream.call, generation)) return null
            keepOpen = true
            stream.endpoint
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            null
        } finally {
            activeHttpCalls.remove(call)
            if (!keepOpen) {
                try {
                    call.cancel()
                } catch (_: Throwable) {
                }
                try {
                    // The callback may already have obtained a response when a
                    // cancellation races with endpoint parsing.
                    openedStream?.response?.close()
                } catch (_: Throwable) {
                }
            }
        }
    }

    /** Reads the initial endpoint event on the cancellable OkHttp callback. */
    private suspend fun Call.awaitSseEndpoint(baseUrl: String): SseEndpointStream? = suspendCancellableCoroutine { cont ->
        cont.invokeOnCancellation {
            try {
                cancel()
            } catch (_: Throwable) {
            }
        }
        enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!cont.isCompleted) cont.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                fun finish(value: SseEndpointStream?) {
                    if (value == null) {
                        response.close()
                        call.cancel()
                    }
                    if (!cont.isCompleted) {
                        cont.resume(value) { _, stream, _ ->
                            stream?.response?.close()
                            stream?.call?.cancel()
                        }
                    } else {
                        response.close()
                    }
                }

                try {
                    if (response.code !in 200..299 ||
                        !(response.header("Content-Type") ?: "").contains("text/event-stream", ignoreCase = true)
                    ) {
                        finish(null)
                        return
                    }
                    val source = response.body?.source() ?: run {
                        finish(null)
                        return
                    }
                    var eventName: String? = null
                    for (index in 0 until 200) {
                        val line = readBoundedLine(source) ?: break
                        when {
                            line.startsWith("event:") -> eventName = line.removePrefix("event:").trim()
                            line.startsWith("data:") -> {
                                val data = line.removePrefix("data:").trim()
                                if (eventName == "endpoint" && data.isNotBlank()) {
                                    val endpoint = if (data.length <= MAX_SSE_ENDPOINT_CHARS) resolveEndpoint(baseUrl, data) else null
                                    if (endpoint == null) finish(null) else finish(SseEndpointStream(endpoint, call, response, source))
                                    return
                                }
                            }
                            line.isEmpty() -> eventName = null
                        }
                    }
                    finish(null)
                } catch (e: Exception) {
                    response.close()
                    if (!cont.isCompleted) cont.resumeWithException(e)
                }
            }
        })
    }

    /**
     * Reads the event stream until it closes. Every JSON-RPC object carrying an
     * id completes the matching pending POST; notifications are ignored.
     */
    private fun startSsePump(source: BufferedSource, response: Response, call: Call, connection: Long): Boolean {
        val epoch = sseEpochSequence.incrementAndGet()
        val oldJob = synchronized(sseStateLock) {
            if (connectionGeneration != connection) return false
            val old = sseJob
            activeSseEpoch = epoch
            sseJob = null
            sseCall = call
            sseResponse = response
            old
        }
        oldJob?.cancel()
        val job = sseScope.launch(start = CoroutineStart.LAZY) {
            try {
                val data = StringBuilder()
                var dataBytes = 0L
                while (isActive) {
                    val line = readBoundedLine(source) ?: break
                    when {
                        line.isEmpty() -> {
                            if (data.isNotEmpty()) dispatchSseData(data.toString(), epoch)
                            data.setLength(0)
                            dataBytes = 0L
                        }
                        line.startsWith("data:") -> {
                            val payload = line.removePrefix("data:").trim()
                            if (payload.isEmpty() || payload == "[DONE]") continue
                            val payloadBytes = payload.toByteArray(StandardCharsets.UTF_8).size.toLong()
                            val separatorBytes = if (data.isNotEmpty()) 1L else 0L
                            if (dataBytes + separatorBytes + payloadBytes > MAX_SSE_EVENT_BYTES) {
                                throw IOException("MCP SSE event exceeds $MAX_SSE_EVENT_BYTES bytes")
                            }
                            if (data.isNotEmpty()) data.append('\n')
                            data.append(payload)
                            dataBytes += separatorBytes + payloadBytes
                        }
                        // event:/id:/retry:/comment lines carry no JSON payload.
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // A read timeout or a server restart ends this stream; pending
                // requests fail fast instead of hanging until their timeout.
                android.util.Log.w("McpClient", "SSE stream ended: ${e.message}")
            } finally {
                val ownsTransport = synchronized(sseStateLock) {
                    if (activeSseEpoch == epoch && connectionGeneration == connection) {
                        activeSseEpoch = null
                        sseJob = null
                        sseCall = null
                        sseResponse = null
                        messageEndpoint = null
                        sessionId = null
                        serverName = ""
                        cachedTools = emptyList()
                        bearerToken = null
                        connectionGeneration++
                        true
                    } else {
                        false
                    }
                }
                try {
                    response.close()
                } catch (_: Throwable) {
                }
                if (ownsTransport) {
                    failPendingSse("MCP SSE stream ended", epoch)
                    activeHttpCalls.entries.filter { it.value == connection }.forEach { it.key.cancel() }
                }
            }
        }
        synchronized(sseStateLock) {
            if (activeSseEpoch == epoch) sseJob = job else job.cancel()
        }
        job.start()
        return true
    }

    private fun dispatchSseData(payload: String, epoch: Long) {
        val obj = try {
            JSONObject(payload)
        } catch (_: Exception) {
            return
        }
        if (!isValidJsonRpcResponse(obj, expectedId = null)) return
        val id = numericResponseId(obj)?.toString() ?: return
        val pending = synchronized(sseStateLock) {
            if (activeSseEpoch != epoch) return
            val candidate = ssePending[id] ?: return
            if (candidate.epoch != epoch || !ssePending.remove(id, candidate)) return
            candidate
        }
        pending.response.complete(obj)
    }

    /**
     * Resolves the `endpoint` event of the legacy SSE transport. The result must be on
     * the same origin (scheme, host, port) as the URL the user configured: the server
     * controls this value, so following an absolute URL anywhere would let a hostile or
     * compromised MCP server point this client (and its bearer token) at an arbitrary
     * host, including loopback/LAN services. Cross-origin endpoints are rejected.
     */
    internal fun resolveEndpoint(baseUrl: String, data: String): String? {
        return try {
            if (data.length > MAX_SSE_ENDPOINT_CHARS) return null
            val base = baseUrl.toHttpUrlOrNull() ?: return null
            val resolved = base.resolve(data.trim()) ?: return null
            if (!sameOrigin(base, resolved) || resolved.username.isNotEmpty() || resolved.password.isNotEmpty()) {
                null
            } else {
                resolved.toString()
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun sameOrigin(a: HttpUrl, b: HttpUrl): Boolean {
        if (a.scheme != b.scheme || a.port != b.port) return false
        // 127.0.0.1 and localhost are the same machine talking to itself; a local
        // server may advertise either spelling in its endpoint event.
        if (isLoopbackHost(a.host) && isLoopbackHost(b.host)) return true
        return a.host.equals(b.host, ignoreCase = true)
    }

    internal fun isLoopbackHost(host: String): Boolean {
        val h = host.lowercase().removePrefix("[").removeSuffix("]")
        if (h == "localhost" || h == "::1") return true
        val parts = h.split('.').mapNotNull { it.toIntOrNull() }
        return parts.size == 4 && parts.all { it in 0..255 } && parts[0] == 127
    }

    /**
     * The `Authorization` value for one request URL, or null when no token may go
     * there. Checked per request — not only when connecting — so a redirect or a
     * changed endpoint can never coax the stored token onto an unsafe transport.
     *
     * A bearer token may go over https anywhere, or over plain http only to loopback
     * or a private/LAN address (an on-device or home-network server). Plain http to a
     * public host would expose the credential on the wire. mDNS `.local` names are
     * deliberately NOT exempted: they resolve over broadcast and are spoofable on
     * untrusted networks, so they need https like everything else.
     */
    internal fun authHeaderFor(requestUrl: String): String? {
        val token = bearerToken ?: return null
        val httpUrl = try {
            requestUrl.toHttpUrlOrNull()
        } catch (_: Throwable) {
            null
        } ?: return null
        return if (isTokenSafeTransport(httpUrl)) "Bearer $token" else null
    }

    internal fun isTokenSafeTransport(url: HttpUrl): Boolean {
        if (url.isHttps) return true
        val host = url.host.lowercase()
        if (host == "localhost") return true
        val parts = host.split('.').mapNotNull { it.toIntOrNull() }
        if (parts.size == 4 && parts.all { it in 0..255 }) {
            val (a, b) = parts[0] to parts[1]
            return a == 127 || a == 10 ||
                (a == 192 && b == 168) ||
                (a == 172 && b in 16..31) ||
                (a == 169 && b == 254)
        }
        return host == "::1" || host == "[::1]"
    }
}
