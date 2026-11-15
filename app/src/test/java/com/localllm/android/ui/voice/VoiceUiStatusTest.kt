package com.localllm.android.ui.voice

import com.localllm.android.R
import com.localllm.android.voice.InteractiveVoiceState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Voice states must stay distinguishable, and a failure must survive on screen instead of
 * silently returning the loop to idle.
 */
class VoiceUiStatusTest {

    @Test
    fun idleMapsToIdle() {
        assertEquals(VoiceUiStatus.IDLE, VoiceUiStatus.from(InteractiveVoiceState.IDLE, null))
    }

    @Test
    fun listeningThinkingAndSpeakingStayDistinct() {
        assertEquals(
            VoiceUiStatus.LISTENING,
            VoiceUiStatus.from(InteractiveVoiceState.LISTENING, null)
        )
        assertEquals(
            VoiceUiStatus.THINKING,
            VoiceUiStatus.from(InteractiveVoiceState.PROCESSING, null)
        )
        assertEquals(
            VoiceUiStatus.SPEAKING,
            VoiceUiStatus.from(InteractiveVoiceState.SPEAKING, null)
        )
    }

    @Test
    fun anErrorOutranksWhateverTheLoopStateIs() {
        InteractiveVoiceState.entries.forEach { state ->
            assertEquals(
                "error must win over $state",
                VoiceUiStatus.ERROR,
                VoiceUiStatus.from(state, "Microphone initialization failed")
            )
        }
    }

    @Test
    fun blankErrorTextIsNotAnError() {
        assertEquals(VoiceUiStatus.IDLE, VoiceUiStatus.from(InteractiveVoiceState.IDLE, "   "))
        assertEquals(VoiceUiStatus.IDLE, VoiceUiStatus.from(InteractiveVoiceState.IDLE, ""))
    }

    @Test
    fun everyStatusHasItsOwnHeadlineAndLabel() {
        val headlines = VoiceUiStatus.entries.map { VoiceStatusCopy.headlineRes(it) }
        val labels = VoiceUiStatus.entries.map { VoiceStatusCopy.labelRes(it) }
        assertEquals("each state needs distinct copy", headlines.size, headlines.toSet().size)
        assertEquals("each state needs a distinct mono label", labels.size, labels.toSet().size)
        assertNotEquals(
            VoiceStatusCopy.headlineRes(VoiceUiStatus.IDLE),
            VoiceStatusCopy.headlineRes(VoiceUiStatus.ERROR)
        )
    }

    @Test
    fun idleAndErrorOfferStartWhileActiveStatesOfferStop() {
        assertEquals(
            R.string.voice_dialogue_start,
            VoiceStatusCopy.dialActionRes(VoiceUiStatus.IDLE)
        )
        assertEquals(
            R.string.voice_dialogue_start,
            VoiceStatusCopy.dialActionRes(VoiceUiStatus.ERROR)
        )
        listOf(VoiceUiStatus.LISTENING, VoiceUiStatus.THINKING, VoiceUiStatus.SPEAKING).forEach {
            assertEquals(R.string.voice_stop_dialogue, VoiceStatusCopy.dialActionRes(it))
        }
    }

    @Test
    fun headlineCopyDoesNotReuseTheIdleHintForActiveStates() {
        val idle = VoiceStatusCopy.headlineRes(VoiceUiStatus.IDLE)
        val active = listOf(
            VoiceUiStatus.LISTENING,
            VoiceUiStatus.THINKING,
            VoiceUiStatus.SPEAKING,
            VoiceUiStatus.ERROR
        ).map { VoiceStatusCopy.headlineRes(it) }
        assertTrue("active states must not show the idle hint", active.none { it == idle })
    }
}
