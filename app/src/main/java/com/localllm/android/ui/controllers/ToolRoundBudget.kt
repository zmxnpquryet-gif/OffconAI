package com.localllm.android.ui.controllers

import com.localllm.android.engine.McpToolProtocol

/** Bounded tool retries for one user message. */
internal class ToolRoundBudget(
    private val maxRounds: Int = McpToolProtocol.MAX_TOOL_ROUNDS,
    private val maxInvalidReplies: Int = MAX_INVALID_REPLIES
) {
    var rounds: Int = 0
        private set

    private var invalidReplies = 0

    fun canRequestTools(): Boolean = rounds < maxRounds

    fun isExhausted(): Boolean = rounds >= maxRounds

    fun onToolRound() {
        rounds++
    }

    /** Returns false once consecutive invalid requests have used their retry budget. */
    fun onInvalidReply(): Boolean {
        invalidReplies++
        return invalidReplies < maxInvalidReplies
    }

    fun onValidReply() {
        invalidReplies = 0
    }

    companion object {
        const val MAX_INVALID_REPLIES = 2
    }
}
