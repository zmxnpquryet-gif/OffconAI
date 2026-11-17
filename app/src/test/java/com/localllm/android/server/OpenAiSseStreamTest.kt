package com.localllm.android.server

import com.localllm.android.engine.GenerationChunk
import com.localllm.android.engine.McpTool
import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.model.ToolCallTemplate
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets

/**
 * Wire-level tests for the OpenAI-compatible SSE path. A scripted chunk flow stands in
 * for the engine so delta ordering, reasoning removal, tool-call termination and SSE
 * framing are asserted on the real bytes instead of on a loaded model.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class OpenAiSseStreamTest {

    private val tools = listOf(
        McpTool("get_weather", "current weather", """{"type":"object"}""")
    )

    private fun chunk(
        token: String = "",
        content: String = "",
        reasoning: String = "",
        isReasoning: Boolean = false,
        complete: Boolean = false
    ) = GenerationChunk(
        token = token,
        isReasoning = isReasoning,
        currentReasoningText = reasoning,
        currentContentText = content,
        tps = 1f,
        promptSpeed = 1f,
        totalTokens = 1,
        isComplete = complete
    )

    private class Response(val raw: ByteArray) {
        private val headerEnd = indexOf(raw, "\r\n\r\n".toByteArray(StandardCharsets.US_ASCII)) + 4

        /** Chunked body decoded to text; fails loudly when the framing is malformed. */
        val body: String by lazy { decodeChunked(raw.copyOfRange(headerEnd, raw.size)) }

        /** SSE payloads in wire order, `[DONE]` included. */
        val events: List<String> by lazy {
            body.split("\n\n").map { it.trim('\n') }.filter { it.isNotEmpty() }.map { event ->
                require(event.startsWith("data: ")) { "not an SSE data event: <$event>" }
                event.removePrefix("data: ")
            }
        }

        /** Parsed JSON events, `[DONE]` excluded. */
        val json: List<JSONObject> by lazy { events.filter { it != "[DONE]" }.map { JSONObject(it) } }

        val contentDeltas: String by lazy {
            json.mapNotNull { event ->
                event.optJSONArray("choices")?.optJSONObject(0)
                    ?.optJSONObject("delta")?.optString("content", "")?.takeIf { it.isNotEmpty() }
            }.joinToString("")
        }

        val finishReasons: List<String?> by lazy {
            json.mapNotNull { event ->
                event.optJSONArray("choices")?.optJSONObject(0)
                    ?.takeIf { choice -> !choice.isNull("finish_reason") }
                    ?.optString("finish_reason")
            }
        }

        val lastEvent: String get() = events.last()
        val statusLine: String get() =
            String(raw, 0, indexOf(raw, "\r\n".toByteArray(StandardCharsets.US_ASCII)), StandardCharsets.US_ASCII)
    }

    private fun write(
        dialect: ToolCallTemplate?,
        chunks: Flow<GenerationChunk>,
        offeredTools: List<McpTool> = if (dialect == null) emptyList() else tools
    ): Pair<Response, Boolean> {
        val out = ByteArrayOutputStream()
        val completed = runBlocking {
            OpenAiSseStream.writeChatCompletion(
                output = out,
                origin = "http://localhost:3000",
                chunks = chunks,
                chatId = "chatcmpl-test",
                modelName = "test-model",
                dialect = dialect,
                offeredTools = offeredTools
            )
        }
        return Response(out.toByteArray()) to completed
    }

    // ---------------- reasoning ----------------

    @Test
    fun `reasoning tokens never reach the assistant content`() {
        // What LlmEngine emits: `token` still carries the reasoning channel, while
        // `currentContentText` is the reasoning-free reply body.
        val (response, completed) = write(
            dialect = null,
            chunks = flow {
                emit(chunk(token = "<think>Let me consider.</think>", reasoning = "Let me consider.", isReasoning = true))
                emit(chunk(token = "Hello", content = "Hello"))
                emit(chunk(token = " there", content = "Hello there"))
                emit(chunk(content = "Hello there", complete = true))
            }
        )

        assertTrue(completed)
        assertEquals("Hello there", response.contentDeltas)
        assertFalse("reasoning leaked into content", response.body.contains("Let me consider"))
        assertFalse(response.body.contains("<think>"))
        assertEquals(listOf("stop"), response.finishReasons)
        assertEquals("[DONE]", response.lastEvent)
    }

    @Test
    fun `interleaved reasoning and prose keep one ordered visible stream`() {
        val (response, completed) = write(
            dialect = null,
            chunks = flow {
                emit(chunk(token = "<think>plan", reasoning = "plan", isReasoning = true))
                emit(chunk(token = "One", content = "One"))
                emit(chunk(token = "</think>", reasoning = "plan", isReasoning = false))
                emit(chunk(token = " two", content = "One two"))
                emit(chunk(content = "One two", complete = true))
            }
        )

        assertTrue(completed)
        assertEquals("One two", response.contentDeltas)
        assertEquals(1, response.finishReasons.size)
        assertEquals("stop", response.finishReasons.single())
    }

    // ---------------- tool markup ----------------

    @Test
    fun `tool markup is stripped and the visible tail is flushed before tool_calls`() {
        val call = """<tool_call>{"name":"get_weather","arguments":{"city":"Seoul"}}</tool_call>"""
        val (response, completed) = write(
            dialect = ToolCallTemplate.HERMES,
            chunks = flow {
                emit(chunk(token = "Checking", content = "Checking"))
                emit(chunk(token = " the forecast", content = "Checking the forecast"))
                // Partial marker: the old flowing diff published `<tool_ca` here and
                // could never take it back.
                emit(chunk(token = " <tool_ca", content = "Checking the forecast <tool_ca"))
                emit(chunk(token = "ll>{", content = "Checking the forecast <tool_ca" + "ll>{"))
                emit(chunk(content = "Checking the forecast $call"))
                emit(chunk(content = "Checking the forecast $call One moment.", complete = true))
            }
        )

        assertTrue(completed)
        assertEquals("Checking the forecast One moment.", response.contentDeltas)
        assertEquals(
            McpToolProtocol.visibleText(
                "Checking the forecast $call One moment.",
                ToolCallTemplate.HERMES
            ),
            response.contentDeltas
        )
        assertEquals(listOf("tool_calls"), response.finishReasons)
        val toolCallFrame = response.json.firstOrNull {
            it.optJSONArray("choices")?.optJSONObject(0)
                ?.optJSONObject("delta")?.has("tool_calls") == true
        }
        assertNotNull("no tool_calls delta was emitted", toolCallFrame)
        val function = toolCallFrame!!.getJSONArray("choices").getJSONObject(0)
            .getJSONObject("delta").getJSONArray("tool_calls").getJSONObject(0)
            .getJSONObject("function")
        assertEquals("get_weather", function.optString("name"))
        assertEquals("Seoul", JSONObject(function.optString("arguments")).optString("city"))
        assertEquals("[DONE]", response.lastEvent)
    }

    @Test
    fun `a tool-call-only reply streams no content deltas`() {
        val call = """<tool_call>{"name":"get_weather","arguments":{"city":"Busan"}}</tool_call>"""
        val (response, completed) = write(
            dialect = ToolCallTemplate.HERMES,
            chunks = flow {
                emit(chunk(token = call, content = call, complete = true))
            }
        )

        assertTrue(completed)
        assertEquals("", response.contentDeltas)
        assertEquals(listOf("tool_calls"), response.finishReasons)
        assertEquals("[DONE]", response.lastEvent)
        // Exactly one non-content choice frame carries the calls, then the terminal one.
        assertEquals(2, response.json.size)
    }

    @Test
    fun `a failed generation still terminates the response`() {
        val (response, completed) = write(
            dialect = null,
            chunks = flow {
                emit(chunk(token = "partial", content = "partial"))
                throw IllegalStateException("BUSY_INFERENCE: busy")
            }
        )

        assertFalse(completed)
        assertEquals("partial", response.contentDeltas)
        val error = response.json.last().optJSONObject("error")
        assertNotNull(error)
        assertTrue(error!!.optString("message").contains("busy"))
        assertEquals(listOf("stop"), response.finishReasons)
        assertEquals("[DONE]", response.lastEvent)
    }

    // ---------------- framing ----------------

    @Test
    fun `chunked framing counts utf8 bytes and every event is a data frame`() {
        val korean = "안녕하세요 세계"
        val out = ByteArrayOutputStream()
        HttpWireWriter.sendStreamHeaders(out, "text/event-stream", "http://127.0.0.1:11434")
        HttpWireWriter.sendChunk(out, "data: $korean\n\n")
        HttpWireWriter.sendChunk(out, "")
        HttpWireWriter.endChunk(out)
        val response = Response(out.toByteArray())

        assertEquals("HTTP/1.1 200 OK", response.statusLine)
        assertEquals("data: $korean\n\n", response.body)
        assertTrue(response.body.endsWith("\n\n"))
    }

    @Test
    fun `preflight advertises cors only for a loopback origin`() {
        val allowed = ByteArrayOutputStream().also { HttpWireWriter.sendCorsPreflight(it, "http://localhost:3000") }
        val denied = ByteArrayOutputStream().also { HttpWireWriter.sendCorsPreflight(it, "https://evil.example") }
        val allowedText = String(allowed.toByteArray(), StandardCharsets.US_ASCII)
        val deniedText = String(denied.toByteArray(), StandardCharsets.US_ASCII)

        assertTrue(allowedText.startsWith("HTTP/1.1 204"))
        assertTrue(allowedText.contains("Access-Control-Allow-Origin: http://localhost:3000"))
        assertFalse(deniedText.contains("Access-Control-Allow-Origin"))
    }

    @Test
    fun `no content event ever duplicates an earlier one`() {
        val (response, _) = write(
            dialect = ToolCallTemplate.HERMES,
            chunks = flow {
                emit(chunk(token = "A", content = "A"))
                emit(chunk(token = "B", content = "AB"))
                emit(chunk(token = "C", content = "ABC"))
                emit(chunk(content = "ABC", complete = true))
            }
        )
        val deltas = response.json.mapNotNull { event ->
            event.optJSONArray("choices")?.optJSONObject(0)
                ?.optJSONObject("delta")?.optString("content", "")?.takeIf { it.isNotEmpty() }
        }
        assertEquals(listOf("A", "B", "C"), deltas)
        // No tool call in this reply: the terminal reason is a normal stop.
        assertEquals(listOf("stop"), response.finishReasons)
        assertNull(
            response.json.firstOrNull {
                it.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("delta")?.has("tool_calls") == true
            }
        )
    }

    private companion object {
        fun indexOf(haystack: ByteArray, needle: ByteArray, from: Int = 0): Int {
            outer@ for (i in from..haystack.size - needle.size) {
                for (j in needle.indices) {
                    if (haystack[i + j] != needle[j]) continue@outer
                }
                return i
            }
            return -1
        }

        /** Decodes an HTTP/1.1 chunked body, asserting each size line is valid hex. */
        fun decodeChunked(body: ByteArray): String {
            val out = ByteArrayOutputStream()
            var at = 0
            while (at < body.size) {
                val lineEnd = indexOf(body, "\r\n".toByteArray(StandardCharsets.US_ASCII), at)
                require(lineEnd > at) { "missing chunk size line at $at" }
                val size = String(body, at, lineEnd - at, StandardCharsets.US_ASCII).toInt(16)
                if (size == 0) break
                val start = lineEnd + 2
                require(start + size <= body.size) { "chunk runs past the body" }
                out.write(body, start, size)
                at = start + size + 2
            }
            return String(out.toByteArray(), StandardCharsets.UTF_8)
        }
    }
}
