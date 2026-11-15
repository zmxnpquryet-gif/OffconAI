package com.localllm.android.server

import com.localllm.android.engine.McpTool
import com.localllm.android.engine.McpToolCall
import com.localllm.android.model.GenerationSettings
import com.localllm.android.model.LlmModel
import org.json.JSONArray
import org.json.JSONObject

/**
 * Pure JSON request/response mappers for Ollama (`/api/`) and OpenAI (`/v1/`) protocols.
 */
internal object ApiPayloadMapper {

    internal const val MAX_HISTORY_MESSAGES = 64
    internal const val MAX_HISTORY_CHARS = 256 * 1024
    internal const val MAX_HISTORY_MESSAGE_CHARS = 64 * 1024
    internal const val MAX_PROMPT_CHARS = 128 * 1024
    internal const val MAX_OPENAI_TOOLS = 64
    internal const val MAX_TOOL_NAME_CHARS = 128
    internal const val MAX_TOOL_DESCRIPTION_CHARS = 1_000
    internal const val MAX_TOOL_SCHEMA_CHARS = 8 * 1024

    private const val OMITTED_CONTEXT = "\n[middle content omitted]\n"

    data class EffectiveRequestOptions(
        val settings: GenerationSettings,
        val numPredict: Int?
    )

    fun resolveRequestOptions(json: JSONObject, base: GenerationSettings): EffectiveRequestOptions {
        val options = json.optJSONObject("options")
        fun optDouble(vararg keys: String, fallback: Double): Double {
            for (key in keys) {
                if (json.has(key) && !json.isNull(key)) {
                    val v = json.optDouble(key, Double.NaN)
                    if (!v.isNaN()) return v
                }
                if (options != null && options.has(key) && !options.isNull(key)) {
                    val v = options.optDouble(key, Double.NaN)
                    if (!v.isNaN()) return v
                }
            }
            return fallback
        }
        fun optInt(vararg keys: String, fallback: Int): Int {
            for (key in keys) {
                if (json.has(key) && !json.isNull(key)) {
                    val v = json.optInt(key, Int.MIN_VALUE)
                    if (v != Int.MIN_VALUE) return v
                }
                if (options != null && options.has(key) && !options.isNull(key)) {
                    val v = options.optInt(key, Int.MIN_VALUE)
                    if (v != Int.MIN_VALUE) return v
                }
            }
            return fallback
        }

        val temperature = optDouble("temperature", fallback = base.temperature.toDouble()).coerceIn(0.0, 2.0).toFloat()
        val topP = optDouble("top_p", fallback = base.topP.toDouble()).coerceIn(0.01, 1.0).toFloat()
        val topK = optInt("top_k", fallback = base.topK).coerceIn(1, 200)
        val numPredictRaw = optInt("num_predict", "max_tokens", fallback = Int.MIN_VALUE)
        val numPredict = if (numPredictRaw == Int.MIN_VALUE) null else numPredictRaw.coerceIn(1, 16384)

        return EffectiveRequestOptions(
            settings = base.copy(temperature = temperature, topP = topP, topK = topK),
            numPredict = numPredict
        )
    }

    fun parseMessages(messagesArray: JSONArray): Pair<List<Pair<String, String>>, String> {
        val historyList = mutableListOf<Pair<String, String>>()
        var lastUserPrompt = ""

        for (i in 0 until messagesArray.length()) {
            val msgObj = messagesArray.optJSONObject(i) ?: continue
            val rawRole = msgObj.optString("role", "user")
            val content = extractMessageContent(msgObj, rawRole)
            // History pairs only carry user/assistant: tool outputs and out-of-band
            // system/developer instructions are folded into user turns so a later
            // takeLast cut can never surface them as a leading assistant message
            // (some templates refuse prompts that open with the assistant).
            val normalizedRole = if (rawRole == "assistant") "assistant" else "user"
            if (i == messagesArray.length() - 1 && normalizedRole == "user") {
                lastUserPrompt = limitText(content, MAX_PROMPT_CHARS)
            } else {
                historyList.add(normalizedRole to limitText(content, MAX_HISTORY_MESSAGE_CHARS))
                while (historyList.size > MAX_HISTORY_MESSAGES || historyChars(historyList) > MAX_HISTORY_CHARS) {
                    historyList.removeAt(0)
                }
            }
        }

        if (lastUserPrompt.isBlank() && historyList.isNotEmpty()) {
            val idx = historyList.indexOfLast { it.first == "user" && it.second.isNotBlank() }
            if (idx >= 0) {
                lastUserPrompt = limitText(historyList[idx].second, MAX_PROMPT_CHARS)
                historyList.removeAt(idx)
            }
        }

        return historyList to lastUserPrompt
    }

    private fun extractMessageContent(msgObj: JSONObject, role: String): String {
        val text = extractTextContent(msgObj.opt("content"))
        if (role == "tool") {
            val name = safeToolName(msgObj.optString("name", "tool"))
            val safeText = text.replace(Regex("</tool_result\\s*>", RegexOption.IGNORE_CASE), "&lt;/tool_result&gt;")
            return "<tool_result name=\"$name\" status=\"ok\">\n$safeText\n</tool_result>"
        }
        if (role == "assistant") {
            val toolCalls = msgObj.optJSONArray("tool_calls")
            if (toolCalls != null && toolCalls.length() > 0) {
                val callsRendered = (0 until toolCalls.length()).mapNotNull { idx ->
                    val tc = toolCalls.optJSONObject(idx) ?: return@mapNotNull null
                    val fn = tc.optJSONObject("function") ?: tc
                    val fnName = safeToolName(fn.optString("name", ""))
                    if (fnName.isBlank()) return@mapNotNull null
                    val argsStr = normalizedToolArguments(fn.opt("arguments"))
                    "<tool_call>{\"name\":${JSONObject.quote(fnName)},\"arguments\":$argsStr}</tool_call>"
                }.joinToString("\n")
                return if (text.isBlank()) callsRendered else "$text\n$callsRendered"
            }
        }
        return text
    }

    private fun historyChars(history: List<Pair<String, String>>): Int =
        history.sumOf { it.second.length }

    /** Keep both ends of an oversized turn and never split a UTF-16 surrogate pair. */
    private fun limitText(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val marker = OMITTED_CONTEXT
        if (maxChars <= marker.length) return text.take(maxChars)
        val remaining = maxChars - marker.length
        var headEnd = remaining / 2
        var tailStart = text.length - (remaining - headEnd)
        if (headEnd > 0 && text[headEnd - 1].isHighSurrogate()) headEnd--
        if (tailStart < text.length && text[tailStart].isLowSurrogate()) tailStart++
        return text.substring(0, headEnd) + marker + text.substring(tailStart)
    }

    private fun safeToolName(name: String): String = buildString(minOf(name.length, MAX_TOOL_NAME_CHARS)) {
        for (char in name.take(MAX_TOOL_NAME_CHARS)) {
            append(if (char.isLetterOrDigit() || char == '_' || char == '-' || char == '.') char else '_')
        }
    }

    private fun normalizedToolArguments(raw: Any?): String = when (raw) {
        is JSONObject -> raw.toString()
        is String -> {
            val trimmed = raw.trim()
            if (trimmed.isEmpty()) "{}" else runCatching { JSONObject(trimmed).toString() }
                .getOrElse { JSONObject().put("input", trimmed).toString() }
        }
        null, JSONObject.NULL -> "{}"
        else -> JSONObject().put("input", raw).toString()
    }

    /**
     * Incremental SSE delta for tool-aware streaming: the visible text is the
     * full text minus reasoning/tool markup, so byte offsets into the raw text
     * do not line up with visible offsets. Deltas are diffed against the
     * previously emitted *visible* text; anything else either duplicates or
     * swallows tokens. Returns null when there is nothing new to send.
     */
    fun visibleStreamDelta(fullVisible: String, emittedVisible: String): String? {
        if (fullVisible.length <= emittedVisible.length) return null
        if (!fullVisible.startsWith(emittedVisible)) return null
        return fullVisible.substring(emittedVisible.length).ifEmpty { null }
    }

    fun parseOpenAiTools(json: JSONObject): List<McpTool> {
        val arr = json.optJSONArray("tools") ?: return emptyList()
        val out = mutableListOf<McpTool>()
        val seenNames = HashSet<String>()
        for (i in 0 until arr.length()) {
            if (out.size >= MAX_OPENAI_TOOLS) break
            val item = arr.optJSONObject(i) ?: continue
            if (item.optString("type", "function") != "function") continue
            val fn = item.optJSONObject("function") ?: item
            val name = safeToolName(fn.optString("name", "").trim())
            if (name.isBlank() || !seenNames.add(name)) continue
            val desc = fn.optString("description", "").take(MAX_TOOL_DESCRIPTION_CHARS)
            val rawParameters = fn.opt("parameters")
            val paramsObject = rawParameters as? JSONObject
                ?: runCatching { JSONObject(rawParameters?.toString() ?: "{}") }.getOrDefault(JSONObject())
            val params = if (paramsObject.toString().length <= MAX_TOOL_SCHEMA_CHARS) {
                paramsObject.toString()
            } else {
                JSONObject()
                    .put("type", "object")
                    .put("description", "Input schema omitted because it exceeds the local size limit.")
                    .toString()
            }
            out += McpTool(name = name, description = desc, parameters = params)
        }
        return out
    }

    fun extractTextContent(raw: Any?): String = when (raw) {
        null, JSONObject.NULL -> ""
        is JSONArray -> (0 until raw.length()).joinToString("") { i ->
            val part = raw.optJSONObject(i)
            if (part != null && part.optString("type", "text") == "text") part.optString("text", "") else ""
        }
        else -> raw.toString()
    }

    fun buildOllamaTagsJson(models: List<LlmModel>, active: LlmModel?): String {
        val modelsArray = JSONArray()
        models.forEach { m ->
            modelsArray.put(
                JSONObject().apply {
                    put("name", m.name)
                    put("model", m.name)
                    put("modified_at", HttpWireWriter.isoTimestamp())
                    put("size", m.sizeBytes)
                    put("digest", m.id)
                    put("details", JSONObject().apply {
                        put("format", m.runtimeType.badge)
                        put("family", m.runtimeType.label)
                        put("parameter_size", "Local")
                        put("quantization_level", m.quantization)
                    })
                }
            )
        }
        if (modelsArray.length() == 0 && active != null) {
            modelsArray.put(
                JSONObject().apply {
                    put("name", active.name)
                    put("model", active.name)
                    put("modified_at", HttpWireWriter.isoTimestamp())
                    put("size", active.sizeBytes)
                    put("digest", active.id)
                }
            )
        }
        return JSONObject().put("models", modelsArray).toString()
    }

    fun buildOpenAiModelsJson(models: List<LlmModel>, active: LlmModel?): String {
        val dataArray = JSONArray()
        val nowEpoch = System.currentTimeMillis() / 1000
        models.forEach { m ->
            dataArray.put(
                JSONObject().apply {
                    put("id", m.name)
                    put("object", "model")
                    put("created", nowEpoch)
                    put("owned_by", "local-device")
                }
            )
        }
        if (dataArray.length() == 0 && active != null) {
            dataArray.put(
                JSONObject().apply {
                    put("id", active.name)
                    put("object", "model")
                    put("created", nowEpoch)
                    put("owned_by", "local-device")
                }
            )
        }
        return JSONObject().apply {
            put("object", "list")
            put("data", dataArray)
        }.toString()
    }

    fun buildOllamaShowJson(currentModel: LlmModel?, serving: GenerationSettings): String =
        JSONObject().apply {
            put("license", "Open")
            put("modelfile", "# Modelfile generated by OffconAI\nFROM ${currentModel?.fileName ?: "model.gguf"}")
            put("parameters", "temperature ${serving.temperature}\ntop_p ${serving.topP}")
            put("template", "{{ .Prompt }}")
            put("details", JSONObject().apply {
                put("parent_model", "")
                put("format", currentModel?.runtimeType?.badge ?: "GGUF")
                put("family", currentModel?.runtimeType?.label ?: "llama.cpp")
                put("parameter_size", "Local")
                put("quantization_level", currentModel?.quantization ?: "Q4_K_M")
            })
        }.toString()

    fun buildOpenAiSseChunk(chatId: String, modelName: String, token: String?, finishReason: String?, errorMessage: String? = null): String {
        val obj = JSONObject().apply {
            put("id", chatId)
            put("object", "chat.completion.chunk")
            put("created", System.currentTimeMillis() / 1000)
            put("model", modelName)
            put("choices", JSONArray().put(JSONObject().apply {
                put("index", 0)
                put("delta", if (token != null) JSONObject().put("content", token) else JSONObject())
                put("finish_reason", finishReason ?: JSONObject.NULL)
            }))
            if (errorMessage != null) {
                put("error", JSONObject().put("message", errorMessage))
            }
        }
        return "data: $obj\n\n"
    }

    fun buildOpenAiToolCallSseChunk(chatId: String, modelName: String, calls: List<McpToolCall>): String {
        val toolCallsArray = JSONArray()
        calls.forEachIndexed { idx, call ->
            toolCallsArray.put(
                JSONObject().apply {
                    put("index", idx)
                    put("id", "call_${chatId.takeLast(8)}_$idx")
                    put("type", "function")
                    put("function", JSONObject().apply {
                        put("name", call.name)
                        put("arguments", call.argumentsJson)
                    })
                }
            )
        }
        val obj = JSONObject().apply {
            put("id", chatId)
            put("object", "chat.completion.chunk")
            put("created", System.currentTimeMillis() / 1000)
            put("model", modelName)
            put("choices", JSONArray().put(JSONObject().apply {
                put("index", 0)
                put("delta", JSONObject().put("tool_calls", toolCallsArray))
                put("finish_reason", JSONObject.NULL)
            }))
        }
        return "data: $obj\n\n"
    }

    fun buildOpenAiCompletionJson(
        chatId: String,
        modelName: String,
        fullAnswer: String,
        totalTokens: Int,
        toolCalls: List<McpToolCall> = emptyList()
    ): String =
        JSONObject().apply {
            put("id", chatId)
            put("object", "chat.completion")
            put("created", System.currentTimeMillis() / 1000)
            put("model", modelName)
            put("choices", JSONArray().put(JSONObject().apply {
                put("index", 0)
                put("message", JSONObject().apply {
                    put("role", "assistant")
                    if (toolCalls.isEmpty()) {
                        put("content", fullAnswer)
                    } else {
                        put("content", if (fullAnswer.isBlank()) JSONObject.NULL else fullAnswer)
                        val arr = JSONArray()
                        toolCalls.forEachIndexed { idx, call ->
                            arr.put(
                                JSONObject().apply {
                                    put("id", "call_${chatId.takeLast(8)}_$idx")
                                    put("type", "function")
                                    put("function", JSONObject().apply {
                                        put("name", call.name)
                                        put("arguments", call.argumentsJson)
                                    })
                                }
                            )
                        }
                        put("tool_calls", arr)
                    }
                })
                put("finish_reason", if (toolCalls.isNotEmpty()) "tool_calls" else "stop")
            }))
            put("usage", JSONObject().apply {
                put("completion_tokens", totalTokens)
                put("total_tokens", totalTokens)
            })
        }.toString()
}
