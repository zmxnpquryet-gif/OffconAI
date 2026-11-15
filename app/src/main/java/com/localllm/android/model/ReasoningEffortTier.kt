package com.localllm.android.model

import androidx.annotation.StringRes
import com.localllm.android.R

/** Shared localized label selection for the settings summary and chat composer. */
enum class ReasoningEffortTier(@StringRes val labelRes: Int) {
    LOW(R.string.gen_effort_low),
    MEDIUM(R.string.gen_effort_medium),
    HIGH(R.string.gen_effort_high),
    MAX(R.string.gen_effort_max);

    companion object {
        fun of(effort: Float): ReasoningEffortTier = when {
            effort <= 0.25f -> LOW
            effort <= 0.6f -> MEDIUM
            effort <= 0.85f -> HIGH
            else -> MAX
        }
    }
}
