package com.localllm.android.document

/**
 * Splits extracted document text into retrieval-sized chunks.
 *
 * Paragraph boundaries are preferred; a paragraph longer than [maxChars] is cut at a
 * sentence boundary first and a word boundary second, so a single huge cell or a
 * PDF page without blank lines cannot produce an unbounded chunk. Consecutive chunks
 * share up to [overlap] characters so an answer that straddles a boundary can still
 * be retrieved. Every boundary respects UTF-16 surrogate pairs so chunks never emit
 * orphaned surrogates (which would corrupt UTF-8 passed across JNI to native engines).
 */
object DocumentChunker {
    const val DEFAULT_MAX_CHARS = 900
    const val DEFAULT_OVERLAP = 120

    fun chunk(
        text: String,
        maxChars: Int = DEFAULT_MAX_CHARS,
        overlap: Int = DEFAULT_OVERLAP
    ): List<String> {
        require(maxChars > 0) { "maxChars must be positive" }
        require(overlap in 0 until maxChars) { "overlap must be smaller than maxChars" }

        val normalized = normalize(text)
        if (normalized.isEmpty()) return emptyList()

        val paragraphs = normalized
            .split(Regex("\\n\\s*\\n"))
            .flatMap { splitLongParagraph(it.trim(), maxChars) }
            .filter { it.isNotBlank() }

        val chunks = mutableListOf<String>()
        val current = StringBuilder()
        for (paragraph in paragraphs) {
            if (current.isEmpty()) {
                current.append(paragraph)
                continue
            }
            if (current.length + 2 + paragraph.length <= maxChars) {
                current.append("\n\n").append(paragraph)
                continue
            }

            chunks += current.toString().trim()
            val tail = safeTakeLast(current, overlap).trimStart()
            current.setLength(0)
            if (tail.isNotEmpty() && tail.length + 2 + paragraph.length <= maxChars) {
                current.append(tail).append("\n\n").append(paragraph)
            } else {
                current.append(paragraph)
            }
        }
        if (current.isNotBlank()) chunks += current.toString().trim()

        return chunks.filter { it.isNotEmpty() }
    }

    internal fun normalize(text: String): String {
        val cleaned = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val c = text[i]
            when {
                Character.isHighSurrogate(c) -> {
                    if (i + 1 < text.length && Character.isLowSurrogate(text[i + 1])) {
                        cleaned.append(c).append(text[i + 1])
                        i += 2
                        continue
                    }
                }
                Character.isLowSurrogate(c) -> { /* drop lone low surrogate */ }
                c == '\uFEFF' || c == '\u00AD' || c == '\uFFFD' -> { /* strip BOM / soft-hyphen / replacement */ }
                c == '\r' -> cleaned.append('\n')
                c == '\n' -> cleaned.append('\n')
                c == '\t' || c == '\u000B' || c == '\u000C' -> cleaned.append(' ')
                c.code < 0x20 || c.code == 0x7F -> { /* strip control chars */ }
                else -> cleaned.append(c)
            }
            i++
        }
        return cleaned.toString()
            .replace(Regex(" {2,}"), " ")
            .replace(Regex("\\n{3,}"), "\n\n")
            .trim()
    }

    private fun safeTakeLast(sb: CharSequence, count: Int): String {
        if (count <= 0 || sb.isEmpty()) return ""
        var start = (sb.length - count).coerceAtLeast(0)
        if (start > 0 && start < sb.length && Character.isLowSurrogate(sb[start])) {
            start++
        }
        return sb.subSequence(start, sb.length).toString()
    }

    private fun splitLongParagraph(paragraph: String, maxChars: Int): List<String> {
        if (paragraph.length <= maxChars) return listOf(paragraph)
        val pieces = mutableListOf<String>()
        var rest = paragraph
        while (rest.length > maxChars) {
            var cut = rest.lastIndexOf(". ", maxChars)
            if (cut < maxChars / 2) cut = rest.lastIndexOf("? ", maxChars)
            if (cut < maxChars / 2) cut = rest.lastIndexOf("! ", maxChars)
            if (cut < maxChars / 2) cut = rest.lastIndexOf('\n', maxChars)
            if (cut < maxChars / 2) cut = rest.lastIndexOf(' ', maxChars)
            if (cut <= 0) cut = maxChars
            if (cut > 0 && cut < rest.length && Character.isLowSurrogate(rest[cut])) {
                cut--
            }
            if (cut <= 0) cut = minOf(maxChars, rest.length)
            pieces += rest.substring(0, cut).trim()
            rest = rest.substring(cut).trimStart()
        }
        if (rest.isNotBlank()) pieces += rest
        return pieces.filter { it.isNotBlank() }
    }
}
