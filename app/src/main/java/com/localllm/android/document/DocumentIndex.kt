package com.localllm.android.document

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import java.util.Locale
import java.util.PriorityQueue

/**
 * End-to-end document RAG: extract, chunk, embed, persist, retrieve.
 *
 * Retrieval combines hashed n-gram cosine similarity with exact case-insensitive
 * keyword overlap on candidate chunks. When the user turns on files for a chat and
 * asks a general/summary prompt (e.g. "summarize this") that shares no terms with the
 * text, opening chunks from the selected documents are still included instead of
 * returning null and starving the model.
 */
class DocumentIndex(context: Context) {

    private val extractor = DocumentTextExtractor(context.applicationContext)
    private val store = DocumentStore(context.applicationContext)
    private val embedder = HashedTextEmbedder()

    data class IngestResult(
        val documentName: String,
        val chunkCount: Int,
        val documentId: String
    )

    suspend fun ingest(
        uri: Uri,
        displayName: String,
        languagePreference: String = "system"
    ): IngestResult = withContext(Dispatchers.IO) {
        val text = extractor.extract(uri, displayName, languagePreference)
        val chunks = DocumentChunker.chunk(text)
        val docId = documentId(uri, displayName)
        if (chunks.isEmpty()) return@withContext IngestResult(displayName, 0, docId)

        store.replaceDocument(
            docId = docId,
            docName = displayName,
            chunks = chunks.map { DocumentStore.ChunkInput(it, embedder.embed(it)) }
        )
        IngestResult(displayName, chunks.size, docId)
    }

    /**
     * Returns a prompt-ready block of the most relevant excerpts from [documentIds],
     * or null only when no documents are selected or none have readable chunks.
     */
    suspend fun buildContext(
        query: String,
        documentIds: Set<String>,
        maxChunks: Int = MAX_CONTEXT_CHUNKS,
        maxChars: Int = MAX_CONTEXT_CHARS
    ): String? = withContext(Dispatchers.IO) {
        if (query.isBlank() || documentIds.isEmpty() || maxChunks <= 0 || maxChars <= 0) {
            return@withContext null
        }

        val queryVector = embedder.embed(query)
        val topByEmbedding = PriorityQueue<Pair<DocumentStore.StoredChunk, Float>>(
            compareBy { it.second }
        )
        val openingByDoc = LinkedHashMap<String, MutableList<DocumentStore.StoredChunk>>()

        store.forEachForSearch(documentIds) { chunk ->
            if (chunk.ordinal < FALLBACK_CHUNKS_PER_DOC) {
                openingByDoc.getOrPut(chunk.docId) { mutableListOf() }.add(chunk)
            }
            val score = if (queryVector.all { it == 0f }) 0f else embedder.similarity(queryVector, chunk.embedding)
            if (topByEmbedding.size < MAX_RANKED_CHUNKS) {
                topByEmbedding.add(chunk to score)
            } else {
                val lowest = topByEmbedding.peek()?.second ?: -1f
                if (score > lowest) {
                    topByEmbedding.poll()
                    topByEmbedding.add(chunk to score)
                }
            }
        }

        if (topByEmbedding.isEmpty() && openingByDoc.isEmpty()) return@withContext null

        val seenKeys = HashSet<Pair<String, Int>>()
        val candidates = mutableListOf<PlainCandidate>()

        fun consider(chunk: DocumentStore.StoredChunk, sim: Float) {
            val key = chunk.docId to chunk.ordinal
            if (!seenKeys.add(key)) return
            val text = store.decryptText(chunk) ?: return
            val name = store.decryptName(chunk)
            candidates += PlainCandidate(
                docId = chunk.docId,
                docName = name,
                ordinal = chunk.ordinal,
                total = chunk.total,
                text = text,
                embeddingScore = sim.coerceAtLeast(0f)
            )
        }

        topByEmbedding.sortedByDescending { it.second }.forEach { (chunk, sim) ->
            consider(chunk, sim)
        }
        openingByDoc.values.flatten().sortedBy { it.ordinal }.forEach { chunk ->
            val sim = if (queryVector.all { it == 0f }) 0f else embedder.similarity(queryVector, chunk.embedding)
            consider(chunk, sim)
        }

        if (candidates.isEmpty()) return@withContext null
        formatSelectedContext(query, candidates, maxChunks, maxChars)
    }

    suspend fun documents(): List<DocumentSummary> = withContext(Dispatchers.IO) {
        store.listDocuments()
    }

    suspend fun chunksForDocument(documentId: String): List<DocumentChunkPreview> = withContext(Dispatchers.IO) {
        store.chunksForDocument(documentId)
    }

    suspend fun stats(): DocumentStats = withContext(Dispatchers.IO) { store.stats() }

    suspend fun deleteDocument(documentId: String) = withContext(Dispatchers.IO) {
        store.deleteDocument(documentId)
    }

    suspend fun clear() = withContext(Dispatchers.IO) { store.clear() }

    private fun documentId(uri: Uri, displayName: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest((uri.toString() + "|" + displayName).toByteArray())
        return digest.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    internal data class PlainCandidate(
        val docId: String,
        val docName: String,
        val ordinal: Int,
        val total: Int,
        val text: String,
        val embeddingScore: Float
    )

    companion object {
        private const val MAX_CONTEXT_CHUNKS = 4
        private const val MAX_CONTEXT_CHARS = 4000
        private const val MIN_COMBINED_SCORE = 0.045f
        private const val MAX_RANKED_CHUNKS = 48
        private const val FALLBACK_CHUNKS_PER_DOC = 2
        private val TERM_REGEX = Regex("[\\p{L}\\p{N}_]+")

        /**
         * Pure ranking + formatting step, extracted for deterministic unit testing.
         */
        internal fun formatSelectedContext(
            query: String,
            candidates: List<PlainCandidate>,
            maxChunks: Int,
            maxChars: Int
        ): String? {
            if (candidates.isEmpty() || maxChunks <= 0 || maxChars <= 0) return null

            val terms = extractQueryTerms(query)
            val normalizedQuery = query.trim().lowercase(Locale.ROOT)

            val scored = candidates.map { candidate ->
                val lexical = lexicalScore(normalizedQuery, terms, candidate.docName, candidate.text)
                val combined = candidate.embeddingScore * 0.45f + lexical * 0.55f
                Triple(candidate, lexical, combined)
            }

            val relevant = scored
                .filter { (_, lexical, combined) -> lexical > 0f || combined >= MIN_COMBINED_SCORE }
                .sortedWith(
                    compareByDescending<Triple<PlainCandidate, Float, Float>> { it.third }
                        .thenBy { it.first.ordinal }
                )
                .map { it.first }

            // General/summary queries ("요약해줘", "explain this file") share few words with
            // the document text: fall back to opening chunks across selected files so the
            // model always sees the attached content instead of getting null.
            val ordered = relevant.ifEmpty {
                roundRobinByDocument(candidates)
            }

            val builder = StringBuilder()
            var used = 0
            var included = 0
            for (chunk in ordered) {
                if (included >= maxChunks) break
                val rawText = chunk.text.trim()
                if (rawText.isEmpty()) continue
                val remaining = maxChars - used
                if (included > 0 && remaining < 120) break
                val excerpt = if (rawText.length > remaining) {
                    safePrefix(rawText, remaining.coerceAtLeast(1))
                } else {
                    rawText
                }
                if (excerpt.isEmpty()) continue
                builder.append('[').append(included + 1).append("] ")
                    .append(chunk.docName)
                    .append(" (").append(chunk.ordinal + 1).append('/').append(chunk.total).append(")\n")
                    .append(excerpt).append("\n\n")
                used += excerpt.length
                included++
            }
            return if (builder.isEmpty()) null else builder.toString().trim()
        }

        private fun roundRobinByDocument(candidates: List<PlainCandidate>): List<PlainCandidate> {
            val queues = LinkedHashMap<String, ArrayDeque<PlainCandidate>>()
            candidates.sortedBy { it.ordinal }.forEach { candidate ->
                queues.getOrPut(candidate.docId) { ArrayDeque() }.addLast(candidate)
            }
            val result = mutableListOf<PlainCandidate>()
            while (queues.values.any { it.isNotEmpty() }) {
                for (q in queues.values) {
                    if (q.isNotEmpty()) result += q.removeFirst()
                }
            }
            return result
        }

        private fun extractQueryTerms(query: String): List<String> =
            TERM_REGEX.findAll(query.lowercase(Locale.ROOT))
                .map { it.value }
                .filter { it.length >= 2 || isCjkOrHangul(it.first()) }
                .distinct()
                .toList()

        private fun isCjkOrHangul(c: Char): Boolean =
            c in '\uAC00'..'\uD7A3' || c in '\u4E00'..'\u9FFF' || c in '\u3040'..'\u30FF'

        private fun lexicalScore(
            normalizedQuery: String,
            terms: List<String>,
            docName: String,
            text: String
        ): Float {
            if (terms.isEmpty()) return 0f
            val haystack = (docName + "\n" + text).lowercase(Locale.ROOT)
            val matched = terms.count { term -> haystack.contains(term) }
            val termCoverage = matched.toFloat() / terms.size.toFloat()
            val phraseBonus = if (normalizedQuery.length >= 3 && haystack.contains(normalizedQuery)) 0.35f else 0f
            return (termCoverage * 0.65f + phraseBonus).coerceAtMost(1.0f)
        }

        private fun safePrefix(text: String, maxChars: Int): String {
            if (text.length <= maxChars) return text
            var cut = maxChars
            if (cut > 0 && Character.isLowSurrogate(text[cut])) cut--
            return text.substring(0, cut).trimEnd()
        }
    }
}
