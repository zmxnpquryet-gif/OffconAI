package com.localllm.android.ui.controllers

import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.model.ToolCallTemplate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Pins the streaming cost fix: the visible text of a tool-aware turn is extracted only
 * for frames the 45 ms throttle actually publishes (plus the final frame), instead of
 * once per token, and the final frame is exactly the text the message is saved with.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class TurnStreamPresenterTest {

    private class RecordingPresenter(
        toolAware: Boolean,
        template: ToolCallTemplate = ToolCallTemplate.HERMES,
        intervalMs: Long = 45L
    ) {
        var now = 1_000L
        var extractions = 0
        val frames = mutableListOf<String>()
        val presenter = TurnStreamPresenter(
            toolTemplate = template,
            toolAware = toolAware,
            intervalMs = intervalMs,
            clock = { now },
            extractVisible = { text, dialect ->
                extractions++
                McpToolProtocol.visibleText(text, dialect)
            }
        )

        fun feed(rawText: String, isComplete: Boolean = false, advanceMs: Long = 10L) {
            now += advanceMs
            presenter.onChunk(rawText, isComplete)?.let { frames.add(it) }
        }
    }

    @Test
    fun `extraction follows the publish cadence instead of the chunk count`() {
        val recorder = RecordingPresenter(toolAware = true)
        val raw = StringBuilder()
        // 200 chunks, 5 ms apart: the 45 ms throttle can publish at most ~22 of them.
        repeat(200) { index ->
            raw.append("tok").append(index).append(' ')
            recorder.feed(raw.toString(), advanceMs = 5L)
        }

        // 200 chunks produced at most ~22 publishes.
        assertTrue(
            "published ${recorder.frames.size} frames",
            recorder.frames.size in 1..25
        )
        assertEquals(
            "one extraction per published frame",
            recorder.frames.size,
            recorder.extractions
        )
        assertTrue(
            "extracted ${recorder.extractions} times for 200 chunks",
            recorder.extractions < 40
        )
    }

    @Test
    fun `tool-dialect stream publishes stable prose and an exact final frame`() {
        val recorder = RecordingPresenter(toolAware = true)
        val pieces = listOf(
            "Checking",
            " the",
            " forecast",
            " for",
            " Seoul",
            ".",
            " <tool_ca", // partial marker: must never be shown
            "ll>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Seoul\"}}",
            "</tool_call>",
            " One moment."
        )
        val raw = StringBuilder()
        for ((index, piece) in pieces.withIndex()) {
            raw.append(piece)
            recorder.feed(raw.toString(), isComplete = index == pieces.lastIndex)
        }

        val finalText = recorder.presenter.finalText(raw.toString())
        assertEquals("Checking the forecast for Seoul. One moment.", finalText)
        assertEquals("the last frame is the saved text", finalText, recorder.frames.last())

        // Nothing shown was ever taken back: every frame is a prefix of the answer.
        recorder.frames.forEach { frame ->
            assertTrue("frame <$frame> is not a prefix of <$finalText>", finalText.startsWith(frame))
        }
        assertFalse(recorder.frames.any { it.contains("<tool") })
        assertTrue("only ${recorder.frames.size} frames", recorder.frames.size >= 3)
        // 3 published frames + 1 final-text extraction, for 10 chunks.
        assertEquals(recorder.frames.size + 1, recorder.extractions)
    }

    @Test
    fun `a suppressed chunk extracts nothing at all`() {
        val recorder = RecordingPresenter(toolAware = true, intervalMs = 45L)
        recorder.feed("hello")
        assertEquals(1, recorder.extractions)
        // Inside the same 45 ms window: suppressed, so no extraction is wasted.
        recorder.feed("hello world", advanceMs = 5L)
        recorder.feed("hello world again", advanceMs = 5L)
        assertEquals(1, recorder.extractions)
        assertEquals(listOf("hello"), recorder.frames)
        // The complete chunk is always published, even inside the window.
        recorder.feed("hello world again.", isComplete = true, advanceMs = 1L)
        assertEquals(2, recorder.extractions)
        assertEquals("hello world again.", recorder.frames.last())
    }

    @Test
    fun `turns without tool use publish the raw text`() {
        val recorder = RecordingPresenter(toolAware = false)
        recorder.feed("plain")
        recorder.feed("plain text", advanceMs = 100L)
        assertEquals(listOf("plain", "plain text"), recorder.frames)
        // No extraction is needed when there is no markup to hide.
        assertEquals(0, recorder.extractions)
        assertEquals("raw text", recorder.presenter.finalText("raw text"))
    }

    @Test
    fun `throttled frames never expose a partial marker`() {
        val recorder = RecordingPresenter(toolAware = true, template = ToolCallTemplate.GEMMA4)
        val pieces = listOf(
            "Searching",
            " <|tool_",
            "call>{\"name\":\"get_weather\",\"arguments\":{}}",
            "<tool_call|>",
            " done"
        )
        val raw = StringBuilder()
        for ((index, piece) in pieces.withIndex()) {
            raw.append(piece)
            recorder.feed(raw.toString(), isComplete = index == pieces.lastIndex, advanceMs = 50L)
        }
        val finalText = recorder.presenter.finalText(raw.toString())
        assertEquals("Searching done", finalText)
        recorder.frames.forEach { frame ->
            assertFalse("markup fragment published: <$frame>", frame.contains("|tool_"))
            assertTrue(finalText.startsWith(frame))
        }
    }

    @Test
    fun `a reply that is only a tool block publishes no markup`() {
        val recorder = RecordingPresenter(toolAware = true)
        val raw = """<tool_call>{"name":"get_weather","arguments":{"city":"Seoul"}}</tool_call>"""
        recorder.feed(raw, isComplete = true)
        assertEquals(listOf(""), recorder.frames)
        assertNull(recorder.presenter.onChunk(raw, isComplete = false))
        assertEquals("", recorder.presenter.finalText(raw))
    }

    // ---------------- cancelled turn: one writer, exact text ----------------

    @Test
    fun `a cancelled round writes the exact visible text once`() {
        val recorder = RecordingPresenter(toolAware = true)
        val call = """<tool_call>{"name":"get_weather","arguments":{"city":"Seoul"}}</tool_call>"""
        val raw = "Checking the forecast for Seoul. $call One moment"
        // The last frame the UI got is throttled and stale; the saved partial is not.
        recorder.feed("Checking the forecast for Seoul.")
        recorder.feed(raw, advanceMs = 5L)

        val partial = TurnCommitPolicy.cancelledPartial(raw, recorder.presenter, roundCommitted = false)
        assertEquals("Checking the forecast for Seoul. One moment", partial)
        assertEquals(recorder.presenter.finalText(raw), partial)
    }

    @Test
    fun `a round that already committed is not written a second time`() {
        val recorder = RecordingPresenter(toolAware = true)
        val raw = "Checking <tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>"
        recorder.feed(raw)

        assertNull(
            "a committed round must not be replaced by its own partial text",
            TurnCommitPolicy.cancelledPartial(raw, recorder.presenter, roundCommitted = true)
        )
    }

    @Test
    fun `a cancelled round with no answer writes nothing`() {
        val recorder = RecordingPresenter(toolAware = true)
        assertNull(TurnCommitPolicy.cancelledPartial("", recorder.presenter, roundCommitted = false))
        assertNull(
            TurnCommitPolicy.cancelledPartial(
                """<tool_call>{"name":"get_weather","arguments":{}}</tool_call>""",
                recorder.presenter,
                roundCommitted = false
            )
        )
    }
}
