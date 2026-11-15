package com.localllm.android.server

import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.model.ToolCallTemplate

/**
 * Turns a growing tool-aware reply into the visible *deltas* of an OpenAI-compatible
 * SSE stream.
 *
 * The reply contains reasoning-free content plus tool markup, and only the prose is
 * visible. Streaming that cannot simply slice the raw text: markup changes the offsets
 * and the prose is trimmed where a block was cut out, so a raw offset delta either
 * duplicates text or swallows it.
 *
 * Bytes already written to the socket can never be taken back, so this class only
 * emits a prefix that is guaranteed to stay a prefix of the final visible text:
 *  - trailing whitespace is held back (the visible text is trimmed when a block opens,
 *    so an emitted trailing space would stop matching the final text);
 *  - a trailing fragment that is still a prefix of a tool marker is held back until it
 *    is either completed (and hidden) or disproved (and emitted).
 *
 * [finish] flushes the remainder once the turn — and therefore the visible text — is
 * final.
 */
internal class VisibleTextStream(
    private val template: ToolCallTemplate,
    private val extractVisible: (String, ToolCallTemplate) -> String = { text, t ->
        McpToolProtocol.visibleText(text, t)
    }
) {
    private var emitted: String = ""

    /** Visible text already sent to the client; test/inspection hook. */
    val emittedText: String get() = emitted

    /**
     * Feeds the latest full reply text.
     *
     * @return the next visible delta, or null when nothing new can be sent safely.
     */
    fun accept(rawText: String): String? =
        emitUpTo(McpToolProtocol.stableVisiblePrefix(extractVisible(rawText, template)))

    /** Flushes the visible tail of the finished reply. */
    fun finish(rawText: String): String? = emitUpTo(extractVisible(rawText, template).trimEnd())

    private fun emitUpTo(target: String): String? {
        // Shared rule with the baseline mapper: a delta may only be the unseen tail of
        // the visible text. When the visible text was rewritten instead of extended
        // (running spaces collapsed, for example) the client already holds the older
        // bytes, so nothing more can be sent for it.
        val delta = ApiPayloadMapper.visibleStreamDelta(target, emitted) ?: return null
        emitted = target
        return delta
    }
}
