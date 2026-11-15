package com.localllm.android.server

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Pinned after the compatibility review: history pairs only carry
 * user/assistant (system/developer/tool fold into user turns), and SSE
 * streaming deltas diff against emitted *visible* text, never raw offsets.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ApiPayloadMapperTest {

    private fun message(role: String, content: String): JSONObject =
        JSONObject().put("role", role).put("content", content)

    @Test
    fun `system and developer messages fold into user turns`() {
        val messages = JSONArray()
            .put(message("system", "Be concise."))
            .put(message("developer", "Use metric units."))
            .put(message("user", "Hi"))
        val (history, lastUser) = ApiPayloadMapper.parseMessages(messages)
        assertEquals(
            listOf("user" to "Be concise.", "user" to "Use metric units."),
            history
        )
        assertEquals("Hi", lastUser)
    }

    @Test
    fun `tool results stay user-attributed with provenance wrapper`() {
        val messages = JSONArray()
            .put(
                JSONObject().put("role", "tool").put("name", "lookup")
                    .put("content", "42")
            )
            .put(message("user", "And?"))
        val (history, lastUser) = ApiPayloadMapper.parseMessages(messages)
        assertEquals(1, history.size)
        assertEquals("user", history[0].first)
        assertEquals("And?", lastUser)
    }

    @Test
    fun `assistant tool calls are re-rendered as inline tool markup`() {
        val toolCalls = JSONArray().put(
            JSONObject().put("function", JSONObject().put("name", "lookup").put("arguments", "{}"))
        )
        val messages = JSONArray()
            .put(
                JSONObject().put("role", "assistant").put("content", "")
                    .put("tool_calls", toolCalls)
            )
            .put(message("user", "next"))
        val (history, lastUser) = ApiPayloadMapper.parseMessages(messages)
        assertEquals("next", lastUser)
        assertEquals(1, history.size)
        assertEquals("assistant", history[0].first)
    }

    @Test
    fun `visible delta emits only unseen tail`() {
        assertEquals(
            " world",
            ApiPayloadMapper.visibleStreamDelta("hello world", "hello")
        )
    }

    @Test
    fun `visible delta is null without growth or on rewrite`() {
        assertNull(ApiPayloadMapper.visibleStreamDelta("hello", "hello"))
        assertNull(ApiPayloadMapper.visibleStreamDelta("hi", "hello world"))
        assertNull(ApiPayloadMapper.visibleStreamDelta("", ""))
    }
}
