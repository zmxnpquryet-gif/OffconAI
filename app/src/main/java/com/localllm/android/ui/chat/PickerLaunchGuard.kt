package com.localllm.android.ui.chat

/**
 * Single-flight guard for the two system pickers launched from the attachment sheet.
 *
 * The sheet stays composed while a picker opens, so a second tap on "Attach document"
 * (or a double tap) used to start a second Activity result contract. The extra result
 * then landed after the sheet closed and overwrote the attachment the user had just
 * picked. The guard makes the launch single-flight and is released from every result
 * callback, including cancel/null results.
 */
class PickerLaunchGuard {

    private var pendingPicker: String? = null

    val pendingId: String? get() = pendingPicker

    val isLaunching: Boolean get() = pendingPicker != null

    /** @return true when the caller may launch [pickerId]; false when one is already open. */
    fun tryLaunch(pickerId: String): Boolean {
        if (pendingPicker != null) return false
        pendingPicker = pickerId
        return true
    }

    /**
     * Release the guard. Must be called from the activity-result callback and from any
     * launch failure, so a picker that never returns cannot wedge the sheet.
     */
    fun onResult() {
        pendingPicker = null
    }
}
