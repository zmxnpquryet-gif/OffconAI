package com.localllm.android.ui.controllers

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the cancellation race fixed in `ChatTurnRunner`: Stop and a newer send must both
 * invalidate a turn that is still preparing (saving the message, loading the model), so
 * a cancelled turn can never start streaming over the turn that replaced it.
 */
class ActiveTurnGuardTest {

    @Test
    fun `a fresh ticket may start`() {
        val guard = ActiveTurnGuard()
        assertFalse(guard.isStale(guard.beginTurn()))
    }

    @Test
    fun `stop invalidates a turn that has not started yet`() {
        val guard = ActiveTurnGuard()
        val ticket = guard.beginTurn()

        // Stop pressed while the turn is still preparing: there is no stream job to
        // cancel yet, which is exactly the window the pre-fix code lost.
        guard.cancelActiveTurn()

        assertTrue("a cancelled turn must not start streaming", guard.isStale(ticket))
        // A later send takes a fresh ticket: the earlier Stop must not poison it.
        val next = guard.beginTurn()
        assertFalse(guard.isStale(next))
    }

    @Test
    fun `a newer send supersedes the previous ticket`() {
        val guard = ActiveTurnGuard()
        val first = guard.beginTurn()
        val second = guard.beginTurn()

        assertTrue("the older turn must not publish over the newer one", guard.isStale(first))
        assertFalse(guard.isStale(second))
        // Stop still invalidates the newest turn.
        guard.cancelActiveTurn()
        assertTrue(guard.isStale(second))
    }

    @Test
    fun `stop before the first send does not poison it`() {
        val guard = ActiveTurnGuard()
        guard.cancelActiveTurn()
        guard.cancelActiveTurn()

        assertFalse(guard.isStale(guard.beginTurn()))
    }

    @Test
    fun `a completed turn is never stale by itself`() {
        val guard = ActiveTurnGuard()
        val ticket = guard.beginTurn()
        repeat(50) { assertFalse(guard.isStale(ticket)) }
    }
}
