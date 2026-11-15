package com.localllm.android.ui.controllers

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the two caps of the multi-round tool loop: a bounded number of tool rounds per
 * user message, and one correction attempt for a batch the parser rejected.
 */
class ToolRoundBudgetTest {

    @Test
    fun `only the configured number of tool rounds may run`() {
        val budget = ToolRoundBudget(maxRounds = 3)

        assertTrue(budget.canRequestTools())
        repeat(2) {
            budget.onToolRound()
            assertTrue(budget.canRequestTools())
            assertFalse(budget.isExhausted())
        }
        budget.onToolRound()
        assertFalse("a fourth round must not be started", budget.canRequestTools())
        assertTrue(budget.isExhausted())
        assertEquals(3, budget.rounds)
    }

    @Test
    fun `an invalid batch gets one correction and then ends the turn`() {
        val budget = ToolRoundBudget()

        assertTrue("first rejected batch is re-asked", budget.onInvalidReply())
        assertFalse("second consecutive rejection ends the turn", budget.onInvalidReply())
    }

    @Test
    fun `a valid batch resets the invalid streak`() {
        val budget = ToolRoundBudget()

        assertTrue(budget.onInvalidReply())
        budget.onValidReply()
        assertEquals(0, budget.invalidStreak)
        assertTrue("the streak counts consecutive failures only", budget.onInvalidReply())
    }

    @Test
    fun `invalid rounds still consume the round budget`() {
        val budget = ToolRoundBudget(maxRounds = 2)
        budget.onToolRound()
        budget.onInvalidReply()
        budget.onToolRound()
        assertTrue(budget.isExhausted())
        assertFalse(budget.canRequestTools())
    }
}
