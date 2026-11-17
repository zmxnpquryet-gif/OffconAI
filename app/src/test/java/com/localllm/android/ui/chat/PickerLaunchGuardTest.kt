package com.localllm.android.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Duplicate taps on an attachment option must not open a second system picker. */
class PickerLaunchGuardTest {

    @Test
    fun guardStartsIdle() {
        val guard = PickerLaunchGuard()
        assertFalse(guard.isLaunching)
        assertNull(guard.pendingId)
    }

    @Test
    fun firstLaunchIsAllowed() {
        val guard = PickerLaunchGuard()
        assertTrue(guard.tryLaunch("image"))
        assertTrue(guard.isLaunching)
        assertEquals("image", guard.pendingId)
    }

    @Test
    fun duplicateTapIsIgnoredWhileThePickerIsOpen() {
        val guard = PickerLaunchGuard()
        assertTrue(guard.tryLaunch("image"))
        assertFalse(guard.tryLaunch("image"))
        assertFalse("a different picker is still a duplicate launch", guard.tryLaunch("document"))
        assertEquals("image", guard.pendingId)
    }

    @Test
    fun aResultReleasesTheGuard() {
        val guard = PickerLaunchGuard()
        guard.tryLaunch("document")
        guard.onResult()
        assertFalse(guard.isLaunching)
        assertTrue("the user may pick again after a cancel", guard.tryLaunch("document"))
    }

    @Test
    fun everyTapAfterTheResultIsAllowedAgain() {
        val guard = PickerLaunchGuard()
        repeat(3) {
            assertTrue(guard.tryLaunch("image"))
            guard.onResult()
        }
    }
}
