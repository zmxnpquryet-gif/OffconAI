package com.localllm.android.ui.chat

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Regression tests for the transcript follow behavior:
 *  - streaming must never yank the view while the user is reading further up,
 *  - submitting a prompt (or pressing "follow answer") must re-pin,
 *  - "at the bottom" must be a tight, explicit measurement instead of the old 100px window.
 */
class ChatScrollPolicyTest {

    @Test
    fun emptyTranscriptCountsAsAtBottom() {
        assertTrue(ChatScrollPolicy.isAtBottom(emptyList(), totalItems = 0, viewportEndOffset = 1200))
    }

    @Test
    fun lastItemFullyVisibleIsAtBottom() {
        val items = listOf(ScrollItem(index = 4, offset = 900, size = 300))
        assertTrue(
            ChatScrollPolicy.isAtBottom(items, totalItems = 5, viewportEndOffset = 1200)
        )
    }

    @Test
    fun lastItemCutOffBelowTheViewportIsNotAtBottom() {
        val items = listOf(ScrollItem(index = 4, offset = 1000, size = 600))
        assertFalse(
            ChatScrollPolicy.isAtBottom(items, totalItems = 5, viewportEndOffset = 1200)
        )
    }

    @Test
    fun anEarlierItemVisibleMeansThereIsMoreBelow() {
        val items = listOf(ScrollItem(index = 2, offset = 0, size = 400))
        assertFalse(
            ChatScrollPolicy.isAtBottom(items, totalItems = 5, viewportEndOffset = 1200)
        )
    }

    @Test
    fun toleranceIsExactAndSmall() {
        // 8px past the viewport edge is inside a 24px tolerance, 25px past is not.
        assertTrue(
            ChatScrollPolicy.isAtBottom(
                lastVisibleIndex = 2,
                lastVisibleEndOffset = 1224,
                totalItems = 3,
                viewportEndOffset = 1200,
                tolerancePx = 24
            )
        )
        assertFalse(
            ChatScrollPolicy.isAtBottom(
                lastVisibleIndex = 2,
                lastVisibleEndOffset = 1225,
                totalItems = 3,
                viewportEndOffset = 1200,
                tolerancePx = 24
            )
        )
    }

    @Test
    fun invalidLayoutSnapshotIsTreatedAsAtBottomSoNothingJumps() {
        assertTrue(
            ChatScrollPolicy.isAtBottom(
                lastVisibleIndex = -1,
                lastVisibleEndOffset = 0,
                totalItems = 3,
                viewportEndOffset = 1200
            )
        )
    }

    @Test
    fun scrollingUpStopsFollowingAndStreamingDoesNotScrollAgain() {
        val follow = ChatFollowState()
        assertTrue("a fresh transcript follows", follow.onEvent(ChatScrollEvent.STREAMING_UPDATE))

        assertFalse(follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED))
        assertFalse("the drag itself must not scroll", follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED))
        assertFalse(follow.isFollowing)

        // Many streaming chunks while the user reads further up: never scroll.
        repeat(10) {
            assertFalse(follow.onEvent(ChatScrollEvent.STREAMING_UPDATE))
        }
        assertFalse(follow.onEvent(ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM))
        assertFalse(follow.onEvent(ChatScrollEvent.STREAMING_UPDATE))
    }

    @Test
    fun draggingBackToTheBottomResumesFollowing() {
        val follow = ChatFollowState()
        follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED)
        follow.onEvent(ChatScrollEvent.USER_DRAG_ENDED_AT_BOTTOM)
        assertTrue(follow.isFollowing)
        assertTrue(follow.onEvent(ChatScrollEvent.STREAMING_UPDATE))
    }

    @Test
    fun aFlingThatSettlesAtTheBottomResumesFollowing() {
        val follow = ChatFollowState()
        follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED)
        follow.onEvent(ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM)
        assertFalse(follow.isFollowing)
        follow.onEvent(ChatScrollEvent.SETTLED_AT_BOTTOM)
        assertTrue(follow.isFollowing)
    }

    @Test
    fun submittingAPromptAlwaysRePinsAndScrolls() {
        val follow = ChatFollowState()
        follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED)
        follow.onEvent(ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM)
        assertFalse(follow.isFollowing)

        assertTrue("submitting must scroll", follow.onEvent(ChatScrollEvent.PROMPT_SUBMITTED))
        assertTrue(follow.isFollowing)
        assertTrue(follow.onEvent(ChatScrollEvent.STREAMING_UPDATE))
    }

    @Test
    fun jumpToLatestRePinsAndScrolls() {
        val follow = ChatFollowState()
        follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED)
        assertTrue(follow.onEvent(ChatScrollEvent.JUMP_TO_LATEST))
        assertTrue(follow.isFollowing)
    }

    @Test
    fun readScrollThenNewPromptThenFollowStreaming() {
        // The exact sequence the screen performs.
        val follow = ChatFollowState()
        follow.onEvent(ChatScrollEvent.USER_DRAG_STARTED)
        follow.onEvent(ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM)
        repeat(5) { assertFalse(follow.onEvent(ChatScrollEvent.STREAMING_UPDATE)) }
        assertTrue(follow.onEvent(ChatScrollEvent.PROMPT_SUBMITTED))
        assertTrue(follow.onEvent(ChatScrollEvent.STREAMING_UPDATE))
    }
}
