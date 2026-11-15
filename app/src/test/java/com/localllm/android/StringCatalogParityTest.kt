package com.localllm.android

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Source-level guard for the three shipped locales (`values`, `values-ko`, `values-zh`).
 *
 * `StringResourceParityTest` proves the app can *resolve* every key at runtime; this test
 * reads the catalogs themselves so it can also catch the two failures a runtime lookup
 * hides: a translated string that drops or reorders a format argument (which throws at
 * format time on that locale only), and a locale that was forgotten when a key was added.
 */
class StringCatalogParityTest {

    private val locales = listOf("values", "values-ko", "values-zh")

    private val entryRegex = Regex(
        """<string\s+name="([^"]+)"[^>]*>(.*?)</string>""",
        RegexOption.DOT_MATCHES_ALL
    )
    private val commentRegex = Regex("""<!--.*?-->""", RegexOption.DOT_MATCHES_ALL)
    private val positionalArgRegex = Regex("""%\d+\$[a-zA-Z]""")

    private fun resDir(): File {
        val candidates = listOf(
            File("src/main/res"),
            File("app/src/main/res"),
            File(System.getProperty("user.dir") ?: ".", "src/main/res"),
            File(System.getProperty("user.dir") ?: ".", "app/src/main/res")
        )
        return candidates.firstOrNull { it.isDirectory }
            ?: error("Could not locate src/main/res from ${File(".").absolutePath}")
    }

    private fun catalog(locale: String): Map<String, String> {
        val file = File(File(resDir(), locale), "strings.xml")
        assertTrue("missing catalog: ${file.absolutePath}", file.isFile)
        val text = file.readText().replace(commentRegex, "")
        return entryRegex.findAll(text).associate { match ->
            match.groupValues[1] to match.groupValues[2]
        }
    }

    private fun args(value: String): List<String> =
        positionalArgRegex.findAll(value).map { it.value }.sorted().toList()

    @Test
    fun everyLocaleDeclaresTheSameKeys() {
        val english = catalog("values").keys
        assertTrue("the English catalog should stay substantial", english.size > 300)
        locales.drop(1).forEach { locale ->
            val keys = catalog(locale).keys
            val missing = (english - keys).sorted()
            val extra = (keys - english).sorted()
            assertTrue("$locale is missing keys: $missing", missing.isEmpty())
            assertTrue("$locale declares keys that English does not: $extra", extra.isEmpty())
        }
    }

    @Test
    fun formatArgumentsMatchAcrossLocales() {
        val english = catalog("values")
        val mismatches = mutableListOf<String>()
        locales.drop(1).forEach { locale ->
            val translated = catalog(locale)
            english.forEach { (key, englishValue) ->
                val other = translated[key] ?: return@forEach
                val expected = args(englishValue)
                val actual = args(other)
                if (expected != actual) {
                    mismatches += "$locale:$key expected=$expected actual=$actual"
                }
            }
        }
        assertTrue("format placeholders drifted: $mismatches", mismatches.isEmpty())
    }

    @Test
    fun noValueIsBlank() {
        val blanks = mutableListOf<String>()
        locales.forEach { locale ->
            catalog(locale).forEach { (key, value) ->
                if (value.isBlank()) blanks += "$locale:$key"
            }
        }
        assertTrue("blank string values: $blanks", blanks.isEmpty())
    }

    @Test
    fun composerAndVoiceCopyIsActuallyTranslated() {
        // Keys added for the usability pass; a copy-paste of the English text into a
        // locale file would otherwise pass every other check.
        val mustBeTranslated = listOf(
            "chat_status_generating",
            "chat_attachment_remove",
            "chat_library_empty",
            "chat_send_disabled_hint",
            "voice_status_label_listening",
            "voice_native_session_title",
            "settings_api_server_off_hint",
            "drawer_tools"
        )
        val english = catalog("values")
        val untranslated = mutableListOf<String>()
        locales.drop(1).forEach { locale ->
            val translated = catalog(locale)
            mustBeTranslated.forEach { key ->
                val en = english[key] ?: error("missing English key $key")
                val other = translated[key]
                assertTrue("$locale is missing $key", other != null && other.isNotBlank())
                if (other == en) untranslated += "$locale:$key"
            }
        }
        assertTrue("still English in another locale: $untranslated", untranslated.isEmpty())
    }

    @Test
    fun theComposerHasNoHardcodedKoreanLeft() {
        // Regression guard for the strings that were literals inside ChatInputBar.
        val hardcoded = listOf("이미지", "프롬프트 처리 중", "생성 중", "삭제")
        val source = File(
            File(resDir().parentFile, "java"),
            "com/localllm/android/ui/chat/ChatInputBar.kt"
        )
        assertTrue("ChatInputBar.kt not found", source.isFile)
        val text = source.readText()
        hardcoded.forEach { literal ->
            assertFalse(
                "ChatInputBar.kt still contains the hardcoded copy \"$literal\"",
                text.contains("\"$literal")
            )
        }
    }

    @Test
    fun everyDeclaredKeyIsReferencedBySomeSource() {
        // Catches the opposite mistake: a key added to the catalog but never wired up, so
        // the visible copy silently stays hardcoded.
        val newKeys = catalog("values").keys.filter {
            it.startsWith("chat_status_") ||
                it.startsWith("chat_attachment") ||
                it.startsWith("chat_state_") ||
                it.startsWith("chat_library_empty") ||
                it.startsWith("voice_status_label_") ||
                it.startsWith("voice_native_session") ||
                it.startsWith("voice_mode_header") ||
                it.startsWith("settings_api_server_off") ||
                it.startsWith("settings_theme_swatch") ||
                it.startsWith("drawer_recent") ||
                it.startsWith("drawer_tools")
        }
        assertTrue("expected new keys to be present", newKeys.isNotEmpty())

        val sourceRoot = File(resDir().parentFile, "java")
        val sources = sourceRoot.walkTopDown()
            .filter { it.isFile && it.extension == "kt" }
            .joinToString("\n") { it.readText() }
        val unused = newKeys.filterNot { sources.contains("R.string.$it") }
        assertTrue("declared but never used: $unused", unused.isEmpty())
    }

    @Test
    fun everyLocaleFileIsWellFormed() {
        locales.forEach { locale ->
            val file = File(File(resDir(), locale), "strings.xml")
            val text = file.readText()
            assertEquals("$locale must open and close <resources>", 1, Regex("<resources>").findAll(text).count())
            assertEquals("$locale must close <resources>", 1, Regex("</resources>").findAll(text).count())
        }
    }

    @Test
    fun localesAreSubstantiallyTranslated() {
        val english = catalog("values")
        locales.drop(1).forEach { locale ->
            val translated = catalog(locale)
            val changed = english.keys.count { key ->
                val value = translated[key]
                value != null && value != english[key]
            }
            assertTrue(
                "$locale only differs from English on $changed of ${english.size} keys",
                changed > english.size / 2
            )
        }
    }
}
