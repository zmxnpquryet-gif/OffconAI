package com.localllm.android.model

import androidx.annotation.StringRes
import com.localllm.android.R

/**
 * Reasoning-effort tiers, shared by the composer pill and [GenerationSettings].
 *
 * The thresholds used to live twice: once as hard-coded English labels
 * ("Low (0.2)", "Medium (0.5)", …) in the composer and once as resource lookups in
 * settings. Keeping one mapping here means a localized label can never drift from the
 * value the engine actually receives.
 */
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
