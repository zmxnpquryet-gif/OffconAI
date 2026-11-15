package com.localllm.android.ui.controllers

import com.localllm.android.model.ChatMessage
import com.localllm.android.model.MessageRole
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

object ConversationExporter {

    fun formatMarkdown(
        title: String,
        modelId: String,
        messages: List<ChatMessage>
    ): String {
        val sb = StringBuilder()
        sb.append("# OffconAI Conversation Export\n\n")
        val effectiveTitle = title.ifBlank { "Conversation" }
        sb.append("> **Title**: ").append(effectiveTitle).append("\n")
        if (modelId.isNotBlank()) {
            sb.append("> **Model**: ").append(modelId).append("\n")
        }
        sb.append("\n---\n\n")

        for (msg in messages) {
            when (msg.role) {
                MessageRole.USER -> {
                    sb.append("### 👤 User\n\n")
                    if (msg.attachment != null) {
                        sb.append("📎 *[Attachment: ").append(msg.attachment.fileName).append("]*\n\n")
                    }
                    sb.append(msg.content.trim()).append("\n\n")
                }
                MessageRole.ASSISTANT -> {
                    sb.append("### 🤖 OffconAI\n\n")
                    if (!msg.reasoning.isNullOrBlank()) {
                        sb.append("<details><summary>💭 Reasoning</summary>\n\n")
                        sb.append(msg.reasoning.trim())
                        sb.append("\n\n</details>\n\n")
                    }
                    sb.append(msg.content.trim()).append("\n\n")
                    if (msg.tps > 0f) {
                        sb.append("> *[Throughput: ")
                            .append(String.format(Locale.US, "%.1f", msg.tps))
                            .append(" t/s]*\n\n")
                    }
                }
                MessageRole.TOOL -> {
                    sb.append("> 🛠️ **Tool Result**\n```\n")
                    sb.append(msg.content.trim())
                    sb.append("\n```\n\n")
                }
                MessageRole.SYSTEM -> {
                    sb.append("> ⚙️ **System**: ").append(msg.content.trim()).append("\n\n")
                }
            }
        }
        return sb.toString()
    }

    fun formatJson(
        conversationId: String,
        title: String,
        modelId: String,
        messages: List<ChatMessage>
    ): String {
        val root = JSONObject()
        root.put("version", "2.0.0")
        root.put("conversation_id", conversationId)
        root.put("title", title.ifBlank { "Conversation" })
        root.put("model_id", modelId)
        root.put("total_messages", messages.size)

        val array = JSONArray()
        for (msg in messages) {
            val obj = JSONObject()
            obj.put("id", msg.id)
            obj.put("role", msg.role.name.lowercase(Locale.US))
            obj.put("content", msg.content)
            if (!msg.reasoning.isNullOrBlank()) {
                obj.put("reasoning", msg.reasoning)
            }
            obj.put("timestamp", msg.timestamp)
            if (msg.tps > 0f) {
                obj.put("tps", msg.tps.toDouble())
            }
            if (msg.promptSpeed > 0f) {
                obj.put("prompt_speed", msg.promptSpeed.toDouble())
            }
            if (msg.contextTokens > 0) {
                obj.put("context_tokens", msg.contextTokens)
            }
            if (msg.attachment != null) {
                val att = JSONObject()
                att.put("file_name", msg.attachment.fileName)
                att.put("mime_type", msg.attachment.mimeType)
                att.put("is_image", msg.attachment.isImage)
                obj.put("attachment", att)
            }
            array.put(obj)
        }
        root.put("messages", array)
        return root.toString(2)
    }
}
