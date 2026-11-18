package com.localllm.android.document

import org.junit.Assert.assertTrue
import org.junit.Test

class DocumentChunkerTest {

    @Test
    fun `long text is split into bounded chunks and keeps the tail`() {
        val text = (1..40).joinToString("\n\n") { "Paragraph $it " + "lorem ipsum ".repeat(10) }
        val chunks = DocumentChunker.chunk(text, maxChars = 300, overlap = 40)

        assertTrue("expected several chunks, got ${chunks.size}", chunks.size > 3)
        chunks.forEach { assertTrue("chunk too long: ${it.length}", it.length <= 300) }
        assertTrue(chunks.joinToString(" ").contains("Paragraph 40"))
    }

    @Test
    fun `a paragraph longer than the limit is cut without exceeding it`() {
        val chunks = DocumentChunker.chunk("word ".repeat(1000), maxChars = 200, overlap = 20)

        assertTrue(chunks.isNotEmpty())
        chunks.forEach { assertTrue("chunk too long: ${it.length}", it.length <= 200) }
    }

    @Test
    fun `blank text produces no chunks`() {
        assertTrue(DocumentChunker.chunk(" \n\n\t ").isEmpty())
        assertTrue(DocumentChunker.chunk("").isEmpty())
    }

    @Test
    fun `surrogate pairs and control characters never corrupt chunk boundaries`() {
        val emojis = "안녕😊".repeat(120)
        val chunks = DocumentChunker.chunk("\uFEFF$emojis\u0000\uFFFD", maxChars = 51, overlap = 11)
        assertTrue(chunks.isNotEmpty())
        chunks.forEach { chunk ->
            assertTrue("unexpected replacement/BOM/nul", !chunk.contains('\uFFFD') && !chunk.contains('\uFEFF') && !chunk.contains('\u0000'))
            for (i in chunk.indices) {
                val c = chunk[i]
                if (Character.isHighSurrogate(c)) {
                    assertTrue("unpaired high surrogate at $i", i + 1 < chunk.length && Character.isLowSurrogate(chunk[i + 1]))
                }
                if (Character.isLowSurrogate(c)) {
                    assertTrue("unpaired low surrogate at $i", i > 0 && Character.isHighSurrogate(chunk[i - 1]))
                }
            }
        }
    }
}
