package com.localllm.android.ui.controllers

import android.os.SystemClock
import com.localllm.android.engine.McpToolProtocol
import com.localllm.android.model.ToolCallTemplate

/** Throttles expensive whole-answer visibility parsing to actual publication frames. */
internal class TurnStreamPresenter(
    private val template: ToolCallTemplate,
    private val toolAware: Boolean,
    private val intervalMs: Long = DEFAULT_INTERVAL_MS,
    private val clock: () -> Long = { SystemClock.uptimeMillis() }
) {
    private var lastPublishAt = -1L
    private var completedVisibleText: String? = null

    /** Null means the current accumulated answer is not due for UI publication. */
    fun onChunk(rawText: () -> String, isComplete: Boolean): String? {
        val now = clock()
        val due = isComplete || lastPublishAt < 0L || now - lastPublishAt >= intervalMs
        if (!due) return null
        lastPublishAt = now
        val currentText = rawText()
        if (!toolAware) {
            if (isComplete) completedVisibleText = McpToolProtocol.visibleText(currentText, template)
            return currentText
        }
        val visible = McpToolProtocol.visibleText(currentText, template)
        if (isComplete) {
            completedVisibleText = visible
            return visible
        }
        return McpToolProtocol.stableVisiblePrefix(visible)
    }

    /** Exact final display text, also used to persist an interrupted partial answer. */
    fun finalText(rawText: String): String =
        completedVisibleText ?: McpToolProtocol.visibleText(rawText, template)

    companion object {
        const val DEFAULT_INTERVAL_MS = 45L
    }
}
