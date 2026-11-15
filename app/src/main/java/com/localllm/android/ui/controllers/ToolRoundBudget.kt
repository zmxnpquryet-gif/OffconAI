package com.localllm.android.ui.controllers

import com.localllm.android.engine.McpToolProtocol

/**
 * Bookkeeping for the multi-round tool loop: how many tool rounds are left and how
 * many rejected replies in a row the model may still correct.
 *
 * Extracted from `ChatTurnRunner` so the two caps that protect the device (each round
 * is a full prompt evaluation, and an invalid batch is re-asked rather than executed)
 * are testable without an engine, a database or a coroutine scheduler.
 */
internal class ToolRoundBudget(
    private val maxRounds: Int = McpToolProtocol.MAX_TOOL_ROUNDS,
    private val maxInvalidReplies: Int = MAX_INVALID_REPLIES
) {
    /** Tool rounds started so far. */
    var rounds: Int = 0
        private set

    /** Consecutive replies the parser rejected. */
    var invalidStreak: Int = 0
        private set

    /** True while the model may still be asked to call tools. */
    fun canRequestTools(): Boolean = rounds < maxRounds

    /** True once the budget is spent: the next reply is the final answer, tools or not. */
    fun isExhausted(): Boolean = rounds >= maxRounds

    /** Consumes one round because the reply asked for tools (valid or not). */
    fun onToolRound(): Int = ++rounds

    /**
     * Records a reply the parser rejected.
     *
     * @return true when the model gets another chance to correct itself, false when the
     *   streak limit is reached and the turn has to end.
     */
    fun onInvalidReply(): Boolean {
        invalidStreak++
        return invalidStreak < maxInvalidReplies
    }

    /** Records a reply that parsed cleanly; the streak only counts consecutive failures. */
    fun onValidReply() {
        invalidStreak = 0
    }

    companion object {
        const val MAX_INVALID_REPLIES = 2
    }
}
