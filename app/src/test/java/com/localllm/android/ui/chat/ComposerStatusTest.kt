package com.localllm.android.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The composer status strip is the only feedback the user gets while a model loads or an
 * image turn is prepared. These tests pin the precedence and — importantly — that no
 * phase invents a progress fraction.
 */
class ComposerStatusTest {

    private fun resolve(
        isModelLoading: Boolean = false,
        modelLoadingProgress: Float = 0f,
        isGenerating: Boolean = false,
        tokensPerSecond: Float = 0f,
        promptTokensPerSecond: Float = 0f,
        contextTokens: Int = 0,
        pendingImageInput: Boolean = false
    ) = ComposerStatusResolver.resolve(
        isModelLoading = isModelLoading,
        modelLoadingProgress = modelLoadingProgress,
        isGenerating = isGenerating,
        tokensPerSecond = tokensPerSecond,
        promptTokensPerSecond = promptTokensPerSecond,
        contextTokens = contextTokens,
        pendingImageInput = pendingImageInput
    )

    @Test
    fun idleComposerHasNoStatus() {
        assertNull(resolve())
    }

    @Test
    fun modelLoadReportsTheRealProgressFraction() {
        val status = resolve(isModelLoading = true, modelLoadingProgress = 0.42f)!!
        assertEquals(ComposerPhase.MODEL_LOADING, status.phase)
        assertEquals(0.42f, status.progress!!, 0.0001f)
        assertFalse(status.isIndeterminate)
    }

    @Test
    fun modelLoadWithoutProgressStaysIndeterminate() {
        val zero = resolve(isModelLoading = true, modelLoadingProgress = 0f)!!
        assertEquals(ComposerPhase.MODEL_LOADING, zero.phase)
        assertNull(zero.progress)
        assertTrue(zero.isIndeterminate)

        val outOfRange = resolve(isModelLoading = true, modelLoadingProgress = 1.4f)!!
        assertNull("a fraction above 1 is not progress", outOfRange.progress)
    }

    @Test
    fun modelLoadOutranksStaleGenerationCounters() {
        val status = resolve(
            isModelLoading = true,
            modelLoadingProgress = 0.5f,
            isGenerating = true,
            tokensPerSecond = 12f
        )!!
        assertEquals(ComposerPhase.MODEL_LOADING, status.phase)
    }

    @Test
    fun tokensProduceMetricsPhase() {
        val status = resolve(isGenerating = true, tokensPerSecond = 18.5f, contextTokens = 640)!!
        assertEquals(ComposerPhase.GENERATING, status.phase)
        assertEquals(18.5f, status.tokensPerSecond, 0.0001f)
        assertEquals(640, status.contextTokens)
    }

    @Test
    fun imageTurnBeforeFirstTokenReportsPreparingImage() {
        val status = resolve(isGenerating = true, pendingImageInput = true)!!
        assertEquals(ComposerPhase.PREPARING_IMAGE, status.phase)
    }

    @Test
    fun imageTurnStopsReportingPreparationOnceTokensArrive() {
        val status = resolve(isGenerating = true, pendingImageInput = true, tokensPerSecond = 3.2f)!!
        assertEquals(ComposerPhase.GENERATING, status.phase)
    }

    @Test
    fun imageTurnWithPromptSpeedShowsTheRealPrefillSpeed() {
        val status = resolve(
            isGenerating = true,
            pendingImageInput = true,
            promptTokensPerSecond = 44f
        )!!
        assertEquals(ComposerPhase.PREFILLING, status.phase)
        assertEquals(44f, status.promptTokensPerSecond, 0.0001f)
    }

    @Test
    fun generationWithoutNumbersFallsBackToPlainGenerating() {
        val status = resolve(isGenerating = true)!!
        assertEquals(ComposerPhase.WAITING_FIRST_TOKEN, status.phase)
    }

    @Test
    fun onlyTheModelLoaderEverReportsProgress() {
        val phases = listOf(
            resolve(isGenerating = true, tokensPerSecond = 9f),
            resolve(isGenerating = true, promptTokensPerSecond = 90f),
            resolve(isGenerating = true),
            resolve(isGenerating = true, pendingImageInput = true)
        )
        phases.forEach { status ->
            assertNull("${status!!.phase} must not invent progress", status.progress)
            assertTrue("${status.phase} is indeterminate", status.isIndeterminate)
        }
    }
}
