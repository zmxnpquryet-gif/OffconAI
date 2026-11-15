package com.localllm.android.ui.controllers

import com.localllm.android.model.ChatAttachment
import com.localllm.android.model.ChatMessage
import com.localllm.android.model.MessageRole
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ConversationExporterTest {

    @Test
    fun `formatMarkdown renders all roles attachments reasoning and throughput`() {
        val messages = listOf(
            ChatMessage(
                conversationId = "conv-1",
                role = MessageRole.SYSTEM,
                content = "You are a helpful assistant."
            ),
            ChatMessage(
                conversationId = "conv-1",
                role = MessageRole.USER,
                content = "Analyze this diagram",
                attachment = ChatAttachment(
                    uriString = "content://media/1",
                    mimeType = "image/png",
                    fileName = "diagram.png",
                    isImage = true
                )
            ),
            ChatMessage(
                conversationId = "conv-1",
                role = MessageRole.ASSISTANT,
                content = "Here is the architectural analysis.",
                reasoning = "Inspecting nodes and edges...",
                tps = 38.5f
            ),
            ChatMessage(
                conversationId = "conv-1",
                role = MessageRole.TOOL,
                content = "{\"status\": \"ok\", \"count\": 4}"
            )
        )

        val md = ConversationExporter.formatMarkdown(
            title = "Architecture Chat",
            modelId = "qwen-3.5-7b",
            messages = messages
        )

        assertTrue(md.startsWith("# OffconAI Conversation Export"))
        assertTrue(md.contains("> **Title**: Architecture Chat"))
        assertTrue(md.contains("> **Model**: qwen-3.5-7b"))
        assertTrue(md.contains("> ⚙️ **System**: You are a helpful assistant."))
        assertTrue(md.contains("### 👤 User"))
        assertTrue(md.contains("📎 *[Attachment: diagram.png]*"))
        assertTrue(md.contains("Analyze this diagram"))
        assertTrue(md.contains("### 🤖 OffconAI"))
        assertTrue(md.contains("<details><summary>💭 Reasoning</summary>"))
        assertTrue(md.contains("Inspecting nodes and edges..."))
        assertTrue(md.contains("Here is the architectural analysis."))
        assertTrue(md.contains("> *[Throughput: 38.5 t/s]*"))
        assertTrue(md.contains("> 🛠️ **Tool Result**"))
        assertTrue(md.contains("{\"status\": \"ok\", \"count\": 4}"))
    }

    @Test
    fun `formatMarkdown with empty messages still produces valid header`() {
        val md = ConversationExporter.formatMarkdown(
            title = "",
            modelId = "",
            messages = emptyList()
        )
        assertTrue(md.contains("# OffconAI Conversation Export"))
        assertTrue(md.contains("> **Title**: Conversation"))
        assertFalse(md.contains("> **Model**:"))
    }

    @Test
    fun `formatJson produces structured payload with all metadata`() {
        val messages = listOf(
            ChatMessage(
                conversationId = "conv-99",
                role = MessageRole.USER,
                content = "What is 2+2?",
                timestamp = 1000L
            ),
            ChatMessage(
                conversationId = "conv-99",
                role = MessageRole.ASSISTANT,
                content = "4",
                reasoning = "Basic arithmetic",
                timestamp = 2000L,
                tps = 45.2f,
                promptSpeed = 120.0f,
                contextTokens = 64
            )
        )

        val jsonStr = ConversationExporter.formatJson(
            conversationId = "conv-99",
            title = "Math Query",
            modelId = "deepseek-r1-8b",
            messages = messages
        )

        val root = JSONObject(jsonStr)
        assertEquals("2.0.0", root.getString("version"))
        assertEquals("conv-99", root.getString("conversation_id"))
        assertEquals("Math Query", root.getString("title"))
        assertEquals("deepseek-r1-8b", root.getString("model_id"))
        assertEquals(2, root.getInt("total_messages"))

        val array = root.getJSONArray("messages")
        assertEquals(2, array.length())

        val msg0 = array.getJSONObject(0)
        assertEquals("user", msg0.getString("role"))
        assertEquals("What is 2+2?", msg0.getString("content"))
        assertEquals(1000L, msg0.getLong("timestamp"))
        assertFalse(msg0.has("reasoning"))

        val msg1 = array.getJSONObject(1)
        assertEquals("assistant", msg1.getString("role"))
        assertEquals("4", msg1.getString("content"))
        assertEquals("Basic arithmetic", msg1.getString("reasoning"))
        assertEquals(45.2, msg1.getDouble("tps"), 0.01)
        assertEquals(120.0, msg1.getDouble("prompt_speed"), 0.01)
        assertEquals(64, msg1.getInt("context_tokens"))
    }
}
