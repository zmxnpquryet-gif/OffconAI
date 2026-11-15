package com.localllm.android.ui.controllers

/** Monotonic ticket guard for preparation and streaming work owned by one chat send. */
internal class ActiveTurnGuard {
    class Ticket internal constructor(
        internal val sequence: Long,
        internal val cancellationEpoch: Long
    )

    private val lock = Any()
    private var sequence = 0L
    private var cancellationEpoch = 0L

    /** A newer send immediately makes every prior ticket stale. */
    fun beginTurn(): Ticket = synchronized(lock) {
        sequence += 1
        Ticket(sequence, cancellationEpoch)
    }

    /** Stop invalidates preparation even when streaming has not started yet. */
    fun cancelActiveTurn() {
        synchronized(lock) { cancellationEpoch += 1 }
    }

    fun isCurrent(ticket: Ticket): Boolean = synchronized(lock) {
        ticket.sequence == sequence && ticket.cancellationEpoch == cancellationEpoch
    }

    fun isStale(ticket: Ticket): Boolean = !isCurrent(ticket)
}
