package com.localllm.android.document

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.nio.charset.Charset

class DocumentIndexTest {

    @Test
    fun `decodeTextBytes handles UTF-8 BOM, UTF-16LE, and Korean CP949 without mojibake`() {
        val utf8Bom = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) +
            "안녕하세요 UTF-8 문서".toByteArray(Charsets.UTF_8)
        assertEquals("안녕하세요 UTF-8 문서", DocumentTextExtractor.decodeTextBytes(utf8Bom))

        val utf16LeBom = byteArrayOf(0xFF.toByte(), 0xFE.toByte()) +
            "Hello UTF-16 테스트".toByteArray(Charsets.UTF_16LE)
        assertEquals("Hello UTF-16 테스트", DocumentTextExtractor.decodeTextBytes(utf16LeBom))

        if (Charset.isSupported("x-windows-949")) {
            val cp949Bytes = "계약서 제1조: 한국어 CP949 인코딩 테스트입니다.".toByteArray(Charset.forName("x-windows-949"))
            val decoded = DocumentTextExtractor.decodeTextBytes(cp949Bytes, "ko")
            assertFalse("CP949 should not decode with replacement chars", decoded.contains('\uFFFD'))
            assertTrue("decoded=$decoded", decoded.contains("계약서 제1조: 한국어 CP949"))
        }
    }

    @Test(expected = IOException::class)
    fun `decodeTextBytes rejects binary NUL-heavy payloads`() {
        val binary = byteArrayOf(0x50, 0x4B, 0x00, 0x00, 0x03, 0x04, 0x00, 0x00, 0x14, 0x00)
        DocumentTextExtractor.decodeTextBytes(binary)
    }

    @Test
    fun `formatSelectedContext finds short 2-char Korean terms and never drops summary queries`() {
        val candidates = listOf(
            DocumentIndex.PlainCandidate(
                docId = "docA",
                docName = "contract.txt",
                ordinal = 0,
                total = 3,
                text = "이 문서는 2026년 신규 프로젝트 개요와 참여 인원 구성을 설명합니다. ".repeat(15),
                embeddingScore = 0.01f
            ),
            DocumentIndex.PlainCandidate(
                docId = "docA",
                docName = "contract.txt",
                ordinal = 1,
                total = 3,
                text = ("일반 조항 및 기타 안내 사항입니다. ".repeat(20)) + "핵심 계약 기간은 24개월입니다.",
                embeddingScore = 0.03f
            )
        )

        // Short 2-char query ("계약") must match chunk 2 even when embedding score < 0.08
        val keywordResult = DocumentIndex.formatSelectedContext("계약 알려줘", candidates, maxChunks = 2, maxChars = 2000)
        assertNotNull(keywordResult)
        assertTrue(keywordResult!!, keywordResult.contains("핵심 계약 기간은 24개월"))

        // Summary query with zero lexical overlap must fall back to opening chunk instead of null
        val summaryResult = DocumentIndex.formatSelectedContext("이 파일 요약해줘", candidates, maxChunks = 2, maxChars = 2000)
        assertNotNull(summaryResult)
        assertTrue(summaryResult!!, summaryResult.contains("contract.txt (1/3)"))
    }

    @Test
    fun `formatSelectedContext distributes fallback across multiple selected documents`() {
        val candidates = listOf(
            DocumentIndex.PlainCandidate("d1", "first.txt", 0, 2, "First file opening content.", 0f),
            DocumentIndex.PlainCandidate("d1", "first.txt", 1, 2, "First file second chunk.", 0f),
            DocumentIndex.PlainCandidate("d2", "second.txt", 0, 2, "Second file opening content.", 0f)
        )

        val result = DocumentIndex.formatSelectedContext("요약", candidates, maxChunks = 2, maxChars = 2000)!!
        assertTrue(result, result.contains("first.txt (1/2)"))
        assertTrue(result, result.contains("second.txt (1/2)"))
    }
}
