package com.localllm.android.document

import android.content.Context
import android.net.Uri
import com.localllm.android.R
import com.localllm.android.i18n.AppStrings
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction
import java.util.Locale

/**
 * Converts a picked document into clean plain text.
 *
 * PDF text comes from PDFBox-Android (with ML Kit OCR fallback for scanned PDFs);
 * DOCX/XLSX/PPTX from [OfficeTextParser]; HTML is stripped to readable text; anything
 * else is decoded with BOM + UTF-8/CP949(EUC-KR)/UTF-16 detection so Korean/CJK text
 * files never turn into `\uFFFD` mojibake or trigger false "binary file" errors.
 */
class DocumentTextExtractor(private val context: Context) {

    private val pdfOcr = PdfOcrExtractor(context.applicationContext)

    suspend fun extract(uri: Uri, displayName: String, languagePreference: String = "system"): String {
        val extension = displayName.substringAfterLast('.', "").lowercase(Locale.ROOT)
        val mime = runCatching { context.contentResolver.getType(uri) }.getOrNull().orEmpty().lowercase(Locale.ROOT)

        return when {
            extension == "pdf" || mime == "application/pdf" -> extractPdf(uri, languagePreference)
            extension == "docx" || mime.contains("wordprocessingml") -> readBytes(uri).let(OfficeTextParser::parseDocx)
            extension == "xlsx" || mime.contains("spreadsheetml") -> readBytes(uri).let(OfficeTextParser::parseXlsx)
            extension == "pptx" || mime.contains("presentationml") -> readBytes(uri).let(OfficeTextParser::parsePptx)
            extension in LEGACY_OFFICE -> throw IOException(
                AppStrings.get(R.string.doc_error_unsupported, ".$extension")
            )
            extension == "html" || extension == "htm" || mime == "text/html" -> {
                stripHtml(decodeTextBytes(readBytes(uri), languagePreference))
            }
            else -> decodeTextBytes(readBytes(uri), languagePreference)
        }
    }

    private suspend fun extractPdf(uri: Uri, languagePreference: String): String {
        PDFBoxResourceLoader.init(context.applicationContext)
        var pageCount = 0
        val digitalText = open(uri).use { stream ->
            PDDocument.load(stream).use { document ->
                if (document.numberOfPages > MAX_PDF_PAGES) {
                    throw IOException(AppStrings.get(R.string.doc_error_pages, MAX_PDF_PAGES))
                }
                pageCount = document.numberOfPages
                PDFTextStripper().apply {
                    sortByPosition = true
                    startPage = 1
                    endPage = document.numberOfPages
                }.getText(document).replace('\u000C', '\n')
            }
        }

        if (!PdfOcrPolicy.needsOcr(digitalText, pageCount)) return digitalText

        val ocrText = try {
            pdfOcr.extract(uri, languagePreference, pageCount)
        } catch (t: Throwable) {
            if (digitalText.isNotBlank()) return digitalText
            throw IOException(AppStrings.get(R.string.doc_error_no_text), t)
        }

        val selected = PdfOcrPolicy.selectText(digitalText, ocrText)
        if (selected.isBlank()) throw IOException(AppStrings.get(R.string.doc_error_no_text))
        return selected
    }

    private fun readBytes(uri: Uri): ByteArray {
        val input = open(uri)
        return input.use { stream ->
            val out = ByteArrayOutputStream()
            val buffer = ByteArray(16 * 1024)
            while (out.size() <= MAX_TEXT_BYTES) {
                val read = stream.read(buffer)
                if (read < 0) break
                out.write(buffer, 0, read)
            }
            if (out.size() > MAX_TEXT_BYTES) {
                throw IOException(AppStrings.get(R.string.doc_error_too_large, MAX_TEXT_BYTES / (1024 * 1024)))
            }
            out.toByteArray()
        }
    }

    private fun open(uri: Uri): InputStream =
        context.contentResolver.openInputStream(uri)
            ?: throw IOException(AppStrings.get(R.string.doc_error_unreadable))

    companion object {
        private const val MAX_TEXT_BYTES = 12 * 1024 * 1024
        private const val MAX_PDF_PAGES = 400
        private val LEGACY_OFFICE = setOf("xls", "ppt", "doc")

        /**
         * Decodes raw text bytes into a clean Kotlin [String], respecting BOMs and
         * strictly validating UTF-8 before falling back to CP949/EUC-KR or GB18030.
         * Rejects true binary files cleanly instead of indexing `\uFFFD` garbage.
         */
        internal fun decodeTextBytes(
            bytes: ByteArray,
            languagePreference: String = "system"
        ): String {
            if (bytes.isEmpty()) return ""

            // 1. Explicit BOM detection (UTF-8 / UTF-16LE / UTF-16BE)
            if (bytes.size >= 3 &&
                bytes[0] == 0xEF.toByte() &&
                bytes[1] == 0xBB.toByte() &&
                bytes[2] == 0xBF.toByte()
            ) {
                return decodeStrict(bytes, 3, Charsets.UTF_8)
                    ?: String(bytes, 3, bytes.size - 3, Charsets.UTF_8)
            }
            if (bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xFE.toByte()) {
                return decodeStrict(bytes, 2, Charsets.UTF_16LE)
                    ?: String(bytes, 2, bytes.size - 2, Charsets.UTF_16LE)
            }
            if (bytes.size >= 2 && bytes[0] == 0xFE.toByte() && bytes[1] == 0xFF.toByte()) {
                return decodeStrict(bytes, 2, Charsets.UTF_16BE)
                    ?: String(bytes, 2, bytes.size - 2, Charsets.UTF_16BE)
            }

            // 2. BOM-less UTF-16 heuristic (e.g. Windows/Android exports with NUL every 2nd byte)
            if (bytes.size >= 4 && bytes.size % 2 == 0) {
                var evenNulls = 0
                var oddNulls = 0
                val sampleLimit = minOf(bytes.size, 1024)
                var idx = 0
                while (idx + 1 < sampleLimit) {
                    if (bytes[idx] == 0.toByte()) evenNulls++
                    if (bytes[idx + 1] == 0.toByte()) oddNulls++
                    idx += 2
                }
                val pairs = sampleLimit / 2
                if (oddNulls > pairs / 3 && evenNulls < pairs / 10) {
                    decodeStrict(bytes, 0, Charsets.UTF_16LE)?.let { decoded ->
                        if (isReadableText(decoded)) return decoded
                    }
                } else if (evenNulls > pairs / 3 && oddNulls < pairs / 10) {
                    decodeStrict(bytes, 0, Charsets.UTF_16BE)?.let { decoded ->
                        if (isReadableText(decoded)) return decoded
                    }
                }
            }

            // 3. Reject obvious binary files (NUL bytes in non-UTF-16 payload, or ZIP/PNG/ELF headers)
            val nullCount = bytes.count { it == 0.toByte() }
            if (nullCount > bytes.size / 20) {
                throw IOException(AppStrings.get(R.string.doc_error_binary))
            }

            // 4. Strict UTF-8 first
            decodeStrict(bytes, 0, Charsets.UTF_8)?.let { decoded ->
                if (looksBinaryControlHeavy(decoded)) {
                    throw IOException(AppStrings.get(R.string.doc_error_binary))
                }
                return decoded.removePrefix("\uFEFF")
            }

            // 5. Fallback encodings for non-UTF-8 text files (Korean CP949/EUC-KR, Chinese GB18030, etc.)
            val candidates = when (languagePreference.lowercase(Locale.ROOT)) {
                "zh" -> listOf("GB18030", "Big5", "x-windows-949", "EUC-KR", "MS932", "windows-1252")
                "ja" -> listOf("MS932", "x-windows-949", "GB18030", "windows-1252")
                else -> listOf("x-windows-949", "EUC-KR", "GB18030", "MS932", "windows-1252")
            }
            for (charsetName in candidates) {
                if (!Charset.isSupported(charsetName)) continue
                val decoded = decodeStrict(bytes, 0, Charset.forName(charsetName)) ?: continue
                if (isReadableText(decoded)) {
                    return decoded.removePrefix("\uFEFF")
                }
            }

            // 6. Last resort: lossy UTF-8 only if mostly valid characters, otherwise reject as binary
            val lossy = String(bytes, Charsets.UTF_8)
            val badChars = lossy.count { it == '\uFFFD' || isDisallowedControl(it) }
            if (lossy.isNotEmpty() && badChars > lossy.length / 8) {
                throw IOException(AppStrings.get(R.string.doc_error_binary))
            }
            return lossy.removePrefix("\uFEFF")
        }

        private fun decodeStrict(bytes: ByteArray, offset: Int, charset: Charset): String? = try {
            val decoder = charset.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
            decoder.decode(ByteBuffer.wrap(bytes, offset, bytes.size - offset)).toString()
        } catch (_: Throwable) {
            null
        }

        private fun isDisallowedControl(c: Char): Boolean {
            val code = c.code
            return (code < 0x20 && c != '\n' && c != '\r' && c != '\t') || code == 0x7F
        }

        private fun looksBinaryControlHeavy(text: String): Boolean {
            if (text.isEmpty()) return false
            val bad = text.count { it == '\uFFFD' || isDisallowedControl(it) }
            return bad > text.length / 10
        }

        private fun isReadableText(text: String): Boolean {
            if (text.isEmpty()) return true
            if (looksBinaryControlHeavy(text)) return false
            val good = text.count {
                it.isLetterOrDigit() || it.isWhitespace() || it in '!'..'~' || it in '\uAC00'..'\uD7A3' || it in '\u4E00'..'\u9FFF'
            }
            return good * 10 >= text.length * 7
        }

        internal fun stripHtml(html: String): String {
            val withoutScripts = html
                .replace(Regex("(?is)<script\\b[^>]*>.*?</script>"), " ")
                .replace(Regex("(?is)<style\\b[^>]*>.*?</style>"), " ")
                .replace(Regex("(?i)<(br|p|div|li|tr|h[1-6])\\b[^>]*>"), "\n")
                .replace(Regex("<[^>]+>"), " ")
            return withoutScripts
                .replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
        }
    }
}
