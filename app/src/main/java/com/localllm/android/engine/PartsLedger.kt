package com.localllm.android.engine

import java.security.MessageDigest

/**
 * Identity of the download a set of `.part_*` files belongs to.
 *
 * The segment worker resumes a part file by skipping its length in the next HTTP Range
 * request. That is only safe when the bytes already on disk came from the *same* source:
 * with the previous size-only sidecar, parts written from a different URL (a switched
 * mirror, an edited custom model URL, a re-published file of the same length) were
 * resumed as if they belonged here, and the merged file was silently corrupt. A host
 * that publishes no SHA-256 never caught it, and one that does burned a full re-download.
 *
 * The sidecar therefore records the total size *and* a digest of the declared source URL
 * (never the redirected URL — object-storage signatures change on every request, so
 * hashing that would disable resume entirely). A size-only or unreadable sidecar is
 * treated as unknown, which discards the parts: re-downloading is recoverable, resuming
 * someone else's bytes is not.
 */
internal data class PartsLedger(val totalBytes: Long, val sourceDigest: String) {

    fun encode(): String = "$VERSION:$totalBytes:$sourceDigest"

    companion object {
        const val VERSION = "v2"

        private val DIGEST_HEX = Regex("[0-9a-f]{32}")

        /** Stable, non-reversible fingerprint of the source a part file was fetched from. */
        fun sourceDigest(sourceKey: String): String = MessageDigest.getInstance("SHA-256")
            .digest(sourceKey.trim().toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it.toInt() and 0xFF) }
            .take(32)

        /** Null for a missing, malformed, or legacy (size-only) sidecar. */
        fun parse(raw: String?): PartsLedger? {
            val parts = raw?.trim().orEmpty().split(':')
            if (parts.size != 3 || parts[0] != VERSION) return null
            val totalBytes = parts[1].toLongOrNull() ?: return null
            if (totalBytes <= 0L) return null
            if (!DIGEST_HEX.matches(parts[2])) return null
            return PartsLedger(totalBytes, parts[2])
        }

        /** True when the sidecar describes exactly this download. */
        fun matches(ledger: PartsLedger?, totalBytes: Long, sourceKey: String): Boolean =
            ledger != null &&
                ledger.totalBytes == totalBytes &&
                ledger.sourceDigest == sourceDigest(sourceKey)

        /**
         * True when every existing part still fits inside its segment, so appending at
         * its length resumes at the right byte. An oversized part (a previous run with a
         * different segment layout) would otherwise fail every retry identically.
         */
        fun resumable(partLengths: List<Long>, segmentTotals: LongArray): Boolean {
            if (partLengths.size != segmentTotals.size) return false
            return partLengths.withIndex().all { (index, length) ->
                length >= 0L && length <= segmentTotals[index]
            }
        }
    }
}
