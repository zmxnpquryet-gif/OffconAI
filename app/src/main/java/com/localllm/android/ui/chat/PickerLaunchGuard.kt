package com.localllm.android.ui.chat

/** Prevents rapid taps from launching both attachment pickers or the same picker twice. */
class PickerLaunchGuard {
    private var pendingPicker: String? = null

    val isLaunching: Boolean
        get() = pendingPicker != null

    fun tryLaunch(pickerId: String): Boolean {
        if (pendingPicker != null) return false
        pendingPicker = pickerId
        return true
    }

    /** Release on every result, including cancellation, and on launch failure. */
    fun onResult() {
        pendingPicker = null
    }
}
