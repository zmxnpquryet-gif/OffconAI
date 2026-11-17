package com.localllm.android

import android.content.res.Configuration
import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

/**
 * Guards the multilingual UI: every declared string must resolve in the default (English)
 * catalog and in every shipped translation, so the in-app language switcher can never fall
 * back to a missing key. Chinese was shipped without being covered here.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StringResourceParityTest {

    private fun contextFor(locale: Locale) =
        RuntimeEnvironment.getApplication().createConfigurationContext(
            Configuration(RuntimeEnvironment.getApplication().resources.configuration).apply {
                setLocale(locale)
            }
        )

    @Test
    fun everyStringResolvesInEnglishKoreanAndChinese() {
        val app = RuntimeEnvironment.getApplication()
        val packageName = app.packageName
        val locales = listOf(
            "en" to Locale.ENGLISH,
            "ko" to Locale.KOREAN,
            "zh" to Locale.SIMPLIFIED_CHINESE
        )

        val missing = mutableListOf<String>()
        R.string::class.java.fields.forEach { field ->
            val name = field.name
            locales.forEach { (tag, locale) ->
                val resources = if (tag == "en") app.resources else contextFor(locale).resources
                if (resources.getIdentifier(name, "string", packageName) == 0) missing += "$tag:$name"
            }
        }

        assertTrue("string keys missing on one side: $missing", missing.isEmpty())
        assertTrue(
            "expected the trilingual catalog to be substantial",
            R.string::class.java.fields.size > 300
        )
    }

    @Test
    fun composerAndVoiceStringsResolveToNonBlankCopyInEveryLocale() {
        val english = RuntimeEnvironment.getApplication().resources
        val korean = contextFor(Locale.KOREAN).resources
        val chinese = contextFor(Locale.SIMPLIFIED_CHINESE).resources
        val keys = listOf(
            R.string.chat_attachment_image,
            R.string.chat_attachment_file,
            R.string.chat_attachment_remove,
            R.string.chat_status_prefill,
            R.string.chat_status_generating,
            R.string.chat_status_metrics,
            R.string.chat_status_loading_progress,
            R.string.chat_status_preparing_image,
            R.string.chat_state_ready_to_send,
            R.string.chat_state_streaming,
            R.string.chat_send_disabled_hint,
            R.string.chat_reasoning_effort_with_value,
            R.string.chat_reasoning_effort_expand,
            R.string.chat_attach_opening,
            R.string.chat_library_empty,
            R.string.voice_native_session_title,
            R.string.voice_native_session_desc,
            R.string.voice_native_model_fallback,
            R.string.voice_mode_header_native,
            R.string.voice_state_error,
            R.string.voice_status_label_idle,
            R.string.voice_status_label_listening,
            R.string.voice_status_label_thinking,
            R.string.voice_status_label_speaking,
            R.string.voice_status_label_error,
            R.string.voice_error_dismiss,
            R.string.voice_mic_mute,
            R.string.voice_mic_unmute,
            R.string.voice_tts_speaker_item,
            R.string.settings_api_server_off_hint,
            R.string.settings_theme_swatch_desc,
            R.string.drawer_recent_chats,
            R.string.drawer_tools
        )
        keys.forEach { key ->
            listOf("en" to english, "ko" to korean, "zh" to chinese).forEach { (tag, resources) ->
                val value = resources.getString(key)
                assertTrue("$tag resolved an empty string for resource $key", value.isNotBlank())
            }
        }
    }

    @Test
    fun statusCopyKeepsItsPlaceholdersInEveryLocale() {
        val korean = contextFor(Locale.KOREAN).resources
        val chinese = contextFor(Locale.SIMPLIFIED_CHINESE).resources
        val args: Array<Any> = arrayOf<Any>("12.5", 768)
        listOf(korean, chinese).forEach { resources ->
            val metrics = resources.getString(R.string.chat_status_metrics, *args)
            assertTrue("missing rate in \"$metrics\"", metrics.contains("12.5"))
            assertTrue("missing context count in \"$metrics\"", metrics.contains("768"))
        }
    }

    @Test
    fun percentPlaceholderIsEscapedCorrectly() {
        val english = RuntimeEnvironment.getApplication().resources
        val text = english.getString(R.string.chat_status_loading_progress, "Loading weights", 42)
        assertEquals("Loading weights · 42%", text)
    }
}
