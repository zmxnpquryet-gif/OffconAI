package com.localllm.android.ui.chat

/** The portion of a LazyColumn layout needed to decide whether the end is visible. */
data class ScrollItem(val index: Int, val offset: Int, val size: Int)

object ChatScrollPolicy {
    fun isAtBottom(
        lastVisibleIndex: Int,
        lastVisibleEndOffset: Int,
        totalItems: Int,
        viewportEndOffset: Int,
        tolerancePx: Int
    ): Boolean {
        if (totalItems <= 0 || lastVisibleIndex < 0) return true
        if (lastVisibleIndex < totalItems - 1) return false
        return lastVisibleEndOffset <= viewportEndOffset + tolerancePx
    }
}

enum class ChatScrollEvent {
    USER_DRAG_STARTED,
    USER_DRAG_ENDED_AT_BOTTOM,
    USER_DRAG_ENDED_ABOVE_BOTTOM,
    SETTLED_AT_BOTTOM,
    PROMPT_SUBMITTED,
    JUMP_TO_LATEST,
    STREAMING_UPDATE
}

/** User intent takes precedence over transcript growth while the assistant streams. */
class ChatFollowState(initiallyFollowing: Boolean = true) {
    var isFollowing: Boolean = initiallyFollowing
        private set

    /** Returns true only when this event should move the list to its latest item. */
    fun onEvent(event: ChatScrollEvent): Boolean = when (event) {
        ChatScrollEvent.USER_DRAG_STARTED,
        ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM -> {
            isFollowing = false
            false
        }
        ChatScrollEvent.USER_DRAG_ENDED_AT_BOTTOM,
        ChatScrollEvent.SETTLED_AT_BOTTOM -> {
            isFollowing = true
            false
        }
        ChatScrollEvent.PROMPT_SUBMITTED,
        ChatScrollEvent.JUMP_TO_LATEST -> {
            isFollowing = true
            true
        }
        ChatScrollEvent.STREAMING_UPDATE -> isFollowing
    }
}
