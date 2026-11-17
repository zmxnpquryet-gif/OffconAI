package com.localllm.android.server

import com.localllm.android.engine.GenerationChunk
import com.localllm.android.engine.McpTool
import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.model.ToolCallTemplate
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import java.io.OutputStream

/**
 * Writes an OpenAI-compatible `chat.completions` SSE response for one generation.
 *
 * Extracted from [OllamaApiServer] so the wire behaviour — delta ordering, reasoning
 * removal, tool-call finish reason, and `[DONE]` framing — can be verified against a
 * plain byte stream and a scripted chunk flow instead of a loaded model.
 *
 * Ordering contract (a client concatenating `choices[0].delta.content` must end up with
 * exactly the visible answer):
 *  1. content deltas, each one the unseen tail of the visible text;
 *  2. the visible tail that was held back while the turn was still running;
 *  3. when the model asked for tools: one `delta.tool_calls` frame, then
 *     `finish_reason = "tool_calls"`; otherwise `finish_reason = "stop"`;
 *  4. `data: [DONE]`.
 */
internal object OpenAiSseStream {

    /**
     * @param dialect the tool dialect in use, or null when tools were not offered to
     *   the model (then the reply body is already the visible text).
     * @param offeredTools the tools the client declared, used to parse the final reply.
     * @return true when the generation completed, false when it failed mid-stream.
     */
    suspend fun writeChatCompletion(
        output: OutputStream,
        origin: String?,
        chunks: Flow<GenerationChunk>,
        chatId: String,
        modelName: String,
        dialect: ToolCallTemplate?,
        offeredTools: List<McpTool>
    ): Boolean {
        HttpWireWriter.sendStreamHeaders(output, "text/event-stream", origin)
        var finalContentText = ""
        var emittedContentLength = 0
        // With tools in play the reply mixes prose with markup, so only a visible-text
        // diff is a correct delta. Without tools the content buffer is already visible
        // and append-only, so its growth is the delta.
        val visibleStream = dialect?.let { VisibleTextStream(it) }
        try {
            chunks.collect { chunk ->
                finalContentText = chunk.currentContentText
                if (visibleStream == null) {
                    // `chunk.token` is not usable here: for a reasoning model it still
                    // carries the <think> channel, which must not become the assistant
                    // content of the response.
                    if (finalContentText.length > emittedContentLength) {
                        val delta = finalContentText.substring(emittedContentLength)
                        emittedContentLength = finalContentText.length
                        sendChunk(output, chatId, modelName, delta, null)
                    }
                } else {
                    val delta = visibleStream.accept(finalContentText)
                    if (delta != null) sendChunk(output, chatId, modelName, delta, null)
                }
            }
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            val errText = if (isBusyError(e)) {
                "Server is busy: another generation is already running."
            } else {
                "Inference failed: ${e.localizedMessage}"
            }
            sendChunk(output, chatId, modelName, null, "stop", errText)
            HttpWireWriter.sendChunk(output, SSE_DONE)
            HttpWireWriter.endChunk(output)
            return false
        }

        val parsedBatch = dialect?.let { activeDialect ->
            McpToolProtocol.parseBatch(finalContentText, offeredTools, template = activeDialect)
        }
        // Flush the held-back visible tail before the terminal frame: prose the model
        // wrote after the last unstable fragment is part of the answer.
        visibleStream?.finish(finalContentText)?.let { rest ->
            sendChunk(output, chatId, modelName, rest, null)
        }
        if (parsedBatch != null && parsedBatch.calls.isNotEmpty()) {
            HttpWireWriter.sendChunk(
                output,
                ApiPayloadMapper.buildOpenAiToolCallSseChunk(chatId, modelName, parsedBatch.calls)
            )
            sendChunk(output, chatId, modelName, null, "tool_calls")
        } else {
            sendChunk(output, chatId, modelName, null, "stop")
        }
        HttpWireWriter.sendChunk(output, SSE_DONE)
        HttpWireWriter.endChunk(output)
        return true
    }

    private fun sendChunk(
        output: OutputStream,
        chatId: String,
        modelName: String,
        token: String?,
        finishReason: String?,
        errorMessage: String? = null
    ) {
        HttpWireWriter.sendChunk(
            output,
            ApiPayloadMapper.buildOpenAiSseChunk(chatId, modelName, token, finishReason, errorMessage)
        )
    }

    private fun isBusyError(e: Throwable): Boolean = e.message?.contains("BUSY_INFERENCE") == true

    const val SSE_DONE = "data: [DONE]\n\n"
}
