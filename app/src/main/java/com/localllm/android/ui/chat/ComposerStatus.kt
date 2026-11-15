package com.localllm.android.ui.chat

/**
 * What the composer status strip is reporting right now.
 *
 * The strip used to be three inline `if` branches with Korean literals in them; the
 * phases are named here so the copy can be localized and the precedence unit-tested.
 */
enum class ComposerPhase {
    /** Weights are being loaded; only this phase has a real fraction to show. */
    MODEL_LOADING,

    /** A turn with an image attachment is running and no token has arrived yet. */
    PREPARING_IMAGE,

    /** Tokens are arriving. */
    GENERATING,

    /** Prompt evaluation is running and reports its own speed. */
    PREFILLING,

    /** Generation started but neither speed is known yet. */
    WAITING_FIRST_TOKEN
}

/**
 * One resolved composer status.
 *
 * [progress] is non-null only for [ComposerPhase.MODEL_LOADING] and only when the loader
 * actually reported a fraction. No other phase invents progress.
 */
data class ComposerStatus(
    val phase: ComposerPhase,
    val progress: Float? = null,
    val tokensPerSecond: Float = 0f,
    val promptTokensPerSecond: Float = 0f,
    val contextTokens: Int = 0
) {
    val isIndeterminate: Boolean get() = phase != ComposerPhase.MODEL_LOADING || progress == null
}

/**
 * Resolves the single status line the composer shows.
 *
 * Precedence is deliberate: a model load outranks any stale generation counters, an
 * image-only turn must not claim token generation before the first token exists, and a
 * plain "generating" line is the fallback instead of an empty strip.
 */
object ComposerStatusResolver {

    fun resolve(
        isModelLoading: Boolean,
        modelLoadingProgress: Float,
        isGenerating: Boolean,
        tokensPerSecond: Float,
        promptTokensPerSecond: Float,
        contextTokens: Int,
        pendingImageInput: Boolean
    ): ComposerStatus? {
        if (isModelLoading) {
            return ComposerStatus(
                phase = ComposerPhase.MODEL_LOADING,
                progress = modelLoadingProgress.takeIf { it > 0f && it <= 1f }
            )
        }
        if (!isGenerating) return null
        val noTokensYet = tokensPerSecond <= 0f
        if (!noTokensYet) {
            return ComposerStatus(
                phase = ComposerPhase.GENERATING,
                tokensPerSecond = tokensPerSecond,
                contextTokens = contextTokens
            )
        }
        // A measured prefill rate is real information, so it wins over the generic
        // "still preparing" line even on an image turn.
        if (promptTokensPerSecond > 0f) {
            return ComposerStatus(
                phase = ComposerPhase.PREFILLING,
                promptTokensPerSecond = promptTokensPerSecond,
                contextTokens = contextTokens
            )
        }
        if (pendingImageInput) {
            return ComposerStatus(ComposerPhase.PREPARING_IMAGE, contextTokens = contextTokens)
        }
        return ComposerStatus(ComposerPhase.WAITING_FIRST_TOKEN, contextTokens = contextTokens)
    }
}
