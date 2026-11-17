package com.localllm.android.model

import com.localllm.android.R
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The composer pill used to hard-code "Low"/"Medium"/"High"/"Max" while settings used the
 * localized resources, so the two could disagree. One mapping is now shared and pinned here.
 */
class ReasoningEffortTierTest {

    @Test
    fun lowTierCoversTheBottomQuarter() {
        assertEquals(ReasoningEffortTier.LOW, ReasoningEffortTier.of(0f))
        assertEquals(ReasoningEffortTier.LOW, ReasoningEffortTier.of(0.2f))
        assertEquals(ReasoningEffortTier.LOW, ReasoningEffortTier.of(0.25f))
    }

    @Test
    fun mediumTierIsTheDefault() {
        assertEquals(ReasoningEffortTier.MEDIUM, ReasoningEffortTier.of(0.26f))
        assertEquals(ReasoningEffortTier.MEDIUM, ReasoningEffortTier.of(0.5f))
        assertEquals(ReasoningEffortTier.MEDIUM, ReasoningEffortTier.of(0.6f))
    }

    @Test
    fun highTierCoversTheUpperMiddleRange() {
        assertEquals(ReasoningEffortTier.HIGH, ReasoningEffortTier.of(0.61f))
        assertEquals(ReasoningEffortTier.HIGH, ReasoningEffortTier.of(0.8f))
        assertEquals(ReasoningEffortTier.HIGH, ReasoningEffortTier.of(0.85f))
    }

    @Test
    fun maxTierIsTheTop() {
        assertEquals(ReasoningEffortTier.MAX, ReasoningEffortTier.of(0.86f))
        assertEquals(ReasoningEffortTier.MAX, ReasoningEffortTier.of(1f))
    }

    @Test
    fun everyTierMapsToItsOwnLocalizedLabel() {
        val resIds = ReasoningEffortTier.entries.map { it.labelRes }
        assertEquals(4, resIds.toSet().size)
        assertTrue(resIds.containsAll(listOf(
            R.string.gen_effort_low,
            R.string.gen_effort_medium,
            R.string.gen_effort_high,
            R.string.gen_effort_max
        )))
    }

    @Test
    fun sliderValuesMapToTheLabelsTheUserSees() {
        // The discrete slider steps are 0.2 / 0.5 / 0.8 / 1.0.
        assertEquals(R.string.gen_effort_low, ReasoningEffortTier.of(0.2f).labelRes)
        assertEquals(R.string.gen_effort_medium, ReasoningEffortTier.of(0.5f).labelRes)
        assertEquals(R.string.gen_effort_high, ReasoningEffortTier.of(0.8f).labelRes)
        assertEquals(R.string.gen_effort_max, ReasoningEffortTier.of(1.0f).labelRes)
    }
}
