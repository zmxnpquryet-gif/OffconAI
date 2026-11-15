package com.localllm.android.ui.controllers

/**
 * Epoch guard for the single chat turn that may stream at a time.
 *
 * `ChatTurnRunner.sendMessage` cannot start streaming synchronously: it first persists
 * the user message, may load the model, and builds the document context. Stop — or a
 * second send — can arrive anywhere inside that window. The previous implementation
 * only cancelled `currentStreamJob`, which is still null during preparation, so the
 * "cancelled" turn went on to generate and then overwrote the state of the turn that
 * replaced it. A ticket makes the check explicit: a turn may only start while its
 * ticket is still the newest one and no Stop has been recorded since it was taken.
 *
 * Both operations are monotonic counters behind one lock, so the decision is
 * deterministic and can be tested without any coroutine timing.
 */
internal class ActiveTurnGuard {

    class Ticket internal constructor(
        internal val sequence: Long,
        internal val cancellationEpoch: Long
    )

    private val lock = Any()
    private var sequence = 0L
    private var cancellationEpoch = 0L

    /** Registers a new turn; every ticket taken before this call becomes stale. */
    fun beginTurn(): Ticket = synchronized(lock) {
        sequence += 1
        Ticket(sequence, cancellationEpoch)
    }

    /** Records Stop; every ticket taken before this call becomes stale. */
    fun cancelActiveTurn() {
        synchronized(lock) { cancellationEpoch += 1 }
    }

    /** True when the ticket was superseded by a newer send or invalidated by Stop. */
    fun isStale(ticket: Ticket): Boolean = synchronized(lock) {
        ticket.sequence != sequence || ticket.cancellationEpoch != cancellationEpoch
    }
}
