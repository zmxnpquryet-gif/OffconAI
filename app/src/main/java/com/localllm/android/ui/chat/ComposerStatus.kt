package com.localllm.android.ui.chat

enum class ComposerPhase {
    MODEL_LOADING,
    PREPARING_IMAGE,
    GENERATING,
    PREFILLING,
    WAITING_FIRST_TOKEN
}

data class ComposerStatus(
    val phase: ComposerPhase,
    val progress: Float? = null,
    val tokensPerSecond: Float = 0f,
    val promptTokensPerSecond: Float = 0f,
    val contextTokens: Int = 0
)

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
        if (pendingImageInput && tokensPerSecond <= 0f) {
            return ComposerStatus(ComposerPhase.PREPARING_IMAGE, contextTokens = contextTokens)
        }
        if (tokensPerSecond > 0f) {
            return ComposerStatus(
                phase = ComposerPhase.GENERATING,
                tokensPerSecond = tokensPerSecond,
                contextTokens = contextTokens
            )
        }
        if (promptTokensPerSecond > 0f) {
            return ComposerStatus(
                phase = ComposerPhase.PREFILLING,
                promptTokensPerSecond = promptTokensPerSecond,
                contextTokens = contextTokens
            )
        }
        return ComposerStatus(ComposerPhase.WAITING_FIRST_TOKEN, contextTokens = contextTokens)
    }
}
