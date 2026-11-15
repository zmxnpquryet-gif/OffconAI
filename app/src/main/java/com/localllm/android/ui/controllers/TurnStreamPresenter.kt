package com.localllm.android.ui.controllers

import android.os.SystemClock
import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.model.ToolCallTemplate

/**
 * Publication gate for one streaming chat turn.
 *
 * Why this exists: the visible text of a turn is the reply with tool markup removed,
 * and deriving it means scanning the whole reply (`McpToolProtocol.visibleText`).
 * The reply grows with every token, so re-deriving it for *every* chunk costs O(n) per
 * chunk and O(n²) per turn, while the 45 ms publish throttle only puts a frame on
 * screen every few dozen chunks. Everything extracted for a suppressed chunk was
 * thrown away. This gate extracts only when a frame is actually due, and always on the
 * final (complete) chunk so the last frame matches the saved message exactly.
 *
 * The raw text it receives is already reasoning-free: `LlmEngine` splits the reasoning
 * channel out with `ReasoningStreamParser` before it emits a chunk, so no `<think>`
 * markers reach this class. The extraction function is injectable so a unit test can
 * count exactly how often it runs.
 */
internal class TurnStreamPresenter(
    private val toolTemplate: ToolCallTemplate = ToolCallTemplate.NONE,
    /** True when tool markup must be hidden from the live answer (tool use is active). */
    private val toolAware: Boolean = false,
    private val intervalMs: Long = DEFAULT_INTERVAL_MS,
    private val clock: () -> Long = { SystemClock.uptimeMillis() },
    private val extractVisible: (String, ToolCallTemplate) -> String = { text, template ->
        McpToolProtocol.visibleText(text, template)
    }
) {
    /** -1 means "nothing published yet": the first chunk is always shown. */
    private var lastPublishAt = -1L

    /**
     * Feeds one streamed chunk.
     *
     * @return the visible text that should be published for this chunk, or null when
     *   the throttle suppresses it — in which case nothing was extracted at all.
     */
    fun onChunk(rawText: String, isComplete: Boolean): String? {
        val now = clock()
        val due = isComplete || lastPublishAt < 0L || now - lastPublishAt >= intervalMs
        if (!due) return null
        lastPublishAt = now
        if (!toolAware) return rawText
        val visible = extractVisible(rawText, toolTemplate)
        // While the turn is still running, a trailing fragment of a tool marker or of
        // the prose's whitespace can still disappear; only the prefix is safe to show.
        // The complete chunk is final, so it is published exactly.
        return if (isComplete) visible else McpToolProtocol.stableVisiblePrefix(visible)
    }

    /**
     * Visible text of the finished turn, extracted exactly once.
     *
     * This is what the message is stored with, so it never depends on whether the last
     * chunk happened to be published by [onChunk]. Markup is always stripped here, even
     * when tool use is off, matching what the saved answer has always shown.
     */
    fun finalText(rawText: String): String = extractVisible(rawText, toolTemplate)

    companion object {
        const val DEFAULT_INTERVAL_MS = 45L
    }
}

/**
 * What a stopped turn is still allowed to write.
 *
 * A cancelled turn used to be persisted twice: once by the Stop handler, from whatever
 * frame the 45 ms throttle last published, and once by the streaming turn itself when it
 * noticed the cancellation. Because the message row is inserted with REPLACE, the older,
 * truncated frame could overwrite the answer that had already been written. Exactly one
 * writer survives here — the turn, which holds the raw text — and it writes only when the
 * round has not already committed.
 */
internal object TurnCommitPolicy {

    /**
     * @return the exact text to persist for a cancelled round, or null when this round
     *   must not write anything (it already committed, or the model produced no text).
     */
    fun cancelledPartial(
        roundText: String,
        presenter: TurnStreamPresenter,
        roundCommitted: Boolean
    ): String? {
        if (roundCommitted) return null
        return presenter.finalText(roundText).ifBlank { null }
    }
}
