package com.localllm.android.ui.voice

import androidx.annotation.StringRes
import com.localllm.android.R
import com.localllm.android.voice.InteractiveVoiceState

/**
 * UI-only state model for the voice screen.
 *
 * [InteractiveVoiceState] only describes the hands-free loop (idle/listening/thinking/
 * speaking). It never says "the microphone is missing" or "this turn failed", so the
 * screen had no way to show either: a denied permission produced a toast and a mic
 * failure silently dropped back to idle. This adds the two states the screen needs and
 * keeps the copy mapping in one testable place.
 */
enum class VoiceUiStatus {
    IDLE,
    LISTENING,
    THINKING,
    SPEAKING,
    ERROR;

    companion object {
        fun from(state: InteractiveVoiceState, error: String?): VoiceUiStatus = when {
            !error.isNullOrBlank() -> ERROR
            state == InteractiveVoiceState.LISTENING -> LISTENING
            state == InteractiveVoiceState.PROCESSING -> THINKING
            state == InteractiveVoiceState.SPEAKING -> SPEAKING
            else -> IDLE
        }
    }
}

object VoiceStatusCopy {

    @StringRes
    fun headlineRes(status: VoiceUiStatus): Int = when (status) {
        VoiceUiStatus.IDLE -> R.string.voice_state_idle_hint
        VoiceUiStatus.LISTENING -> R.string.voice_state_listening
        VoiceUiStatus.THINKING -> R.string.voice_state_local_thinking
        VoiceUiStatus.SPEAKING -> R.string.voice_state_speaking_answer
        VoiceUiStatus.ERROR -> R.string.voice_state_error
    }

    /** Short mono label rendered above the headline so each state is visually distinct. */
    @StringRes
    fun labelRes(status: VoiceUiStatus): Int = when (status) {
        VoiceUiStatus.IDLE -> R.string.voice_status_label_idle
        VoiceUiStatus.LISTENING -> R.string.voice_status_label_listening
        VoiceUiStatus.THINKING -> R.string.voice_status_label_thinking
        VoiceUiStatus.SPEAKING -> R.string.voice_status_label_speaking
        VoiceUiStatus.ERROR -> R.string.voice_status_label_error
    }

    /** Action announced for the center dial, which is otherwise an unlabelled image. */
    @StringRes
    fun dialActionRes(status: VoiceUiStatus): Int = when (status) {
        VoiceUiStatus.IDLE, VoiceUiStatus.ERROR -> R.string.voice_dialogue_start
        else -> R.string.voice_stop_dialogue
    }
}
