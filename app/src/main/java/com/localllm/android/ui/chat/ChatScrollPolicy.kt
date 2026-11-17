package com.localllm.android.ui.chat

/** One laid-out LazyColumn item, reduced to the fields the follow decision needs. */
data class ScrollItem(val index: Int, val offset: Int, val size: Int)

/**
 * Transcript follow math, extracted from the screen so it can be unit-tested without an
 * emulator.
 *
 * The old screen used a 100px tolerance on the last visible item, which let a small
 * user drag stay inside the "at the bottom" window and get re-pinned by the next
 * streaming chunk. The tolerance is now a caller-supplied value (12dp on the real
 * screen) and the decision is a pure function of the layout snapshot.
 */
object ChatScrollPolicy {

    /**
     * @return true when the transcript is effectively at its end: the last laid-out item
     * is the final item and its bottom edge is within [tolerancePx] of the viewport end.
     */
    fun isAtBottom(
        lastVisibleIndex: Int,
        lastVisibleEndOffset: Int,
        totalItems: Int,
        viewportEndOffset: Int,
        tolerancePx: Int = 0
    ): Boolean {
        if (totalItems <= 0) return true
        if (lastVisibleIndex < 0) return true
        if (lastVisibleIndex < totalItems - 1) return false
        return lastVisibleEndOffset <= viewportEndOffset + tolerancePx
    }

    fun isAtBottom(
        items: List<ScrollItem>,
        totalItems: Int,
        viewportEndOffset: Int,
        tolerancePx: Int = 0
    ): Boolean {
        val last = items.lastOrNull()
            ?: return totalItems <= 0
        return isAtBottom(last.index, last.offset + last.size, totalItems, viewportEndOffset, tolerancePx)
    }
}

/** Inputs that can change "should the transcript follow new output". */
enum class ChatScrollEvent {
    /** The user touched the transcript: following stops immediately. */
    USER_DRAG_STARTED,

    /** The drag ended with the end of the transcript visible. */
    USER_DRAG_ENDED_AT_BOTTOM,

    /** The drag ended somewhere above the end of the transcript. */
    USER_DRAG_ENDED_ABOVE_BOTTOM,

    /** A scroll (fling or programmatic) settled at the end of the transcript. */
    SETTLED_AT_BOTTOM,

    /** The user submitted a prompt: always follow again. */
    PROMPT_SUBMITTED,

    /** The user pressed "follow answer". */
    JUMP_TO_LATEST,

    /** New streaming content arrived. */
    STREAMING_UPDATE
}

/**
 * Follow state machine for the transcript.
 *
 * Explicit intent beats inferred position: a drag start always unfollows, and only a
 * submit, an explicit jump or a settled-at-bottom scroll re-follows. Each event answers
 * whether the caller should scroll now.
 */
class ChatFollowState(initiallyFollowing: Boolean = true) {

    var isFollowing: Boolean = initiallyFollowing
        private set

    fun onEvent(event: ChatScrollEvent): Boolean {
        when (event) {
            ChatScrollEvent.USER_DRAG_STARTED -> {
                isFollowing = false
                return false
            }
            ChatScrollEvent.USER_DRAG_ENDED_AT_BOTTOM -> {
                isFollowing = true
                return false
            }
            ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM -> {
                isFollowing = false
                return false
            }
            ChatScrollEvent.SETTLED_AT_BOTTOM -> {
                isFollowing = true
                return false
            }
            ChatScrollEvent.PROMPT_SUBMITTED -> {
                isFollowing = true
                return true
            }
            ChatScrollEvent.JUMP_TO_LATEST -> {
                isFollowing = true
                return true
            }
            ChatScrollEvent.STREAMING_UPDATE -> return isFollowing
        }
    }
}
