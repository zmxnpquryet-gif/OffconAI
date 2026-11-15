package com.localllm.android.ui.chat

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.localllm.android.R
import com.localllm.android.model.ChatAttachment
import com.localllm.android.model.LlmModel
import com.localllm.android.model.ReasoningEffortTier
import com.localllm.android.ui.glass.GCompactIconButton
import com.localllm.android.ui.glass.GIcon
import com.localllm.android.ui.glass.GIcons
import com.localllm.android.ui.glass.GLinearProgress
import com.localllm.android.ui.glass.GOptionRow
import com.localllm.android.ui.glass.GSheet
import com.localllm.android.ui.glass.GSlider
import com.localllm.android.ui.glass.GSwitch
import com.localllm.android.ui.glass.GText
import com.localllm.android.ui.glass.GlassTheme
import java.util.Locale
import kotlin.math.roundToInt

@Composable
fun ChatInputBar(
    inputText: String,
    onInputTextChanged: (String) -> Unit,
    onSendMessage: () -> Unit,
    onStopGeneration: () -> Unit,
    isGenerating: Boolean,
    activeModel: LlmModel?,
    pendingAttachment: ChatAttachment?,
    onSetAttachment: (ChatAttachment?) -> Unit,
    reasoningEffort: Float,
    onReasoningEffortChanged: (Float) -> Unit,
    onStartVoiceInput: () -> Unit,
    onOpenVoiceMode: () -> Unit,
    onDocumentPicked: (android.net.Uri, String) -> Unit = { _, _ -> },
    indexedDocuments: List<com.localllm.android.document.DocumentSummary> = emptyList(),
    activeDocumentIds: Set<String> = emptySet(),
    onToggleDocument: (String) -> Unit = {},
    onDeleteDocument: (String) -> Unit = {},
    isModelLoading: Boolean = false,
    modelLoadingProgress: Float = 0f,
    modelLoadingStage: String = "",
    currentTps: Float = 0f,
    promptSpeed: Float = 0f,
    contextTokens: Int = 0,
    isVoiceListening: Boolean = false,
    isPreparingImage: Boolean = false,
    modifier: Modifier = Modifier
) {
    var showAttachmentSheet by remember { mutableStateOf(false) }
    var showReasoningSlider by remember { mutableStateOf(false) }
    var pickerLaunching by remember { mutableStateOf(false) }
    val pickerGuard = remember { PickerLaunchGuard() }
    val context = LocalContext.current
    val isDarkCanvas = GlassTheme.colors.background.luminance() < 0.5f
    val composerLabel = stringResource(R.string.chat_input_hint)

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        pickerGuard.onResult()
        pickerLaunching = false
        if (uri != null) {
            onSetAttachment(
                ChatAttachment(
                    uriString = uri.toString(),
                    mimeType = "image/*",
                    fileName = uri.lastPathSegment ?: "image.jpg",
                    isImage = true
                )
            )
        }
        showAttachmentSheet = false
    }

    // Document Picker contract: extracts and indexes the file for retrieval.
    val documentPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        pickerGuard.onResult()
        pickerLaunching = false
        if (uri != null) {
            val displayName = runCatching {
                context.contentResolver.query(
                    uri,
                    arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null
                )?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                }
            }.getOrNull() ?: uri.lastPathSegment ?: "document"
            onDocumentPicked(uri, displayName)
        }
        showAttachmentSheet = false
    }

    fun launchImagePicker() {
        if (!pickerGuard.tryLaunch("image")) return
        pickerLaunching = true
        runCatching {
            imagePicker.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }.onFailure {
            pickerGuard.onResult()
            pickerLaunching = false
            android.widget.Toast.makeText(
                context,
                context.getString(R.string.chat_picker_unavailable),
                android.widget.Toast.LENGTH_SHORT
            ).show()
        }
    }

    fun launchDocumentPicker() {
        if (!pickerGuard.tryLaunch("document")) return
        pickerLaunching = true
        runCatching {
            documentPicker.launch(arrayOf("text/*", "application/pdf", "*/*"))
        }.onFailure {
            pickerGuard.onResult()
            pickerLaunching = false
            android.widget.Toast.makeText(
                context,
                context.getString(R.string.chat_picker_unavailable),
                android.widget.Toast.LENGTH_SHORT
            ).show()
        }
    }

    val canSend = inputText.isNotBlank() || pendingAttachment != null
    val sendEnabled = isGenerating || canSend

    fun submit() {
        if (canSend) onSendMessage()
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .imePadding()
            .padding(horizontal = 16.dp, vertical = 8.dp)
    ) {
        // 1. Reasoning Effort quick pill if model supports reasoning (e.g. DeepSeek-R1)
        if (activeModel?.supportsReasoning == true) {
            val tierLabel = stringResource(ReasoningEffortTier.of(reasoningEffort).labelRes)
            val effortValue = String.format(Locale.US, "%.1f", reasoningEffort)
            val effortSummary = stringResource(R.string.chat_reasoning_effort_with_value, tierLabel, effortValue)
            val effortStateDescription = stringResource(
                if (showReasoningSlider) R.string.chat_state_expanded else R.string.chat_state_collapsed
            )
            Row(
                modifier = Modifier
                    .padding(bottom = 6.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(GlassTheme.colors.secondary)
                    .semantics {
                        contentDescription = effortSummary
                        stateDescription = effortStateDescription
                    }
                    .clickable(role = Role.Button) { showReasoningSlider = !showReasoningSlider }
                    .padding(horizontal = 10.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                GIcon(
                    imageVector = GIcons.Psychology,
                    contentDescription = null,
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                GText(
                    text = stringResource(R.string.chat_reasoning_effort_label, effortSummary),
                    style = GlassTheme.type.labelSmall,
                    color = GlassTheme.colors.onSurface
                )
            }

            AnimatedVisibility(visible = showReasoningSlider) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(GlassTheme.colors.surface)
                        .border(1.dp, GlassTheme.colors.outline, RoundedCornerShape(12.dp))
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    GText(
                        text = stringResource(R.string.chat_reasoning_slider_title),
                        style = GlassTheme.type.labelSmall,
                        color = GlassTheme.colors.onSurfaceVariant
                    )
                    GSlider(
                        value = reasoningEffort,
                        onValueChange = onReasoningEffortChanged,
                        valueRange = 0f..1f,
                        steps = 3
                    )
                }
            }
        }

        // 2. Pending Attachment Preview
        if (pendingAttachment != null) {
            Box(
                modifier = Modifier
                    .padding(bottom = 8.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(GlassTheme.colors.surface)
                    .border(1.dp, GlassTheme.colors.outline, RoundedCornerShape(12.dp))
                    .padding(8.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (pendingAttachment.isImage) {
                        AsyncImage(
                            model = pendingAttachment.uriString,
                            contentDescription = null,
                            modifier = Modifier
                                .size(48.dp)
                                .clip(RoundedCornerShape(6.dp))
                        )
                    } else {
                        GIcon(
                            imageVector = GIcons.Description,
                            contentDescription = null,
                            tint = GlassTheme.colors.onSurfaceVariant,
                            modifier = Modifier.size(32.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    Column(modifier = Modifier.weight(1f)) {
                        GText(
                            text = pendingAttachment.fileName,
                            style = GlassTheme.type.bodySmall,
                            maxLines = 1
                        )
                        GText(
                            text = stringResource(
                                if (pendingAttachment.isImage) R.string.chat_attachment_image
                                else R.string.chat_attachment_file
                            ),
                            fontSize = 11.sp,
                            color = GlassTheme.colors.onSurfaceVariant
                        )
                    }

                    GCompactIconButton(
                        onClick = { onSetAttachment(null) },
                        modifier = Modifier.size(28.dp)
                    ) {
                        GIcon(
                            imageVector = GIcons.Close,
                            contentDescription = stringResource(R.string.chat_attachment_remove),
                            tint = GlassTheme.colors.onSurfaceVariant,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
        }

        // 3. Quiet composer capsule
        val capsuleShape = RoundedCornerShape(20.dp)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .shadow(
                    elevation = if (isDarkCanvas) 0.dp else 1.dp,
                    shape = capsuleShape,
                    ambientColor = Color.Black.copy(alpha = 0.04f),
                    spotColor = Color.Black.copy(alpha = 0.06f)
                )
                .clip(capsuleShape)
                .background(GlassTheme.colors.surface)
                .border(1.dp, GlassTheme.colors.outline, capsuleShape)
                .padding(start = 6.dp, end = 6.dp, top = 6.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Attachment Icon Button
            GCompactIconButton(
                onClick = { showAttachmentSheet = true },
                modifier = Modifier.size(38.dp)
            ) {
                GIcon(
                    imageVector = GIcons.Add,
                    contentDescription = stringResource(R.string.attach_file),
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(21.dp)
                )
            }

            // Hands-free (live) voice mode trigger
            GCompactIconButton(
                onClick = onOpenVoiceMode,
                modifier = Modifier.size(38.dp)
            ) {
                GIcon(
                    imageVector = GIcons.VoiceMode,
                    contentDescription = stringResource(R.string.nav_voice_mode),
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(20.dp)
                )
            }

            // Main Text Input Field
            Box(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 6.dp, vertical = 6.dp),
                contentAlignment = Alignment.CenterStart
            ) {
                if (inputText.isEmpty()) {
                    GText(
                        text = stringResource(R.string.chat_input_hint),
                        style = GlassTheme.type.bodyLarge,
                        color = GlassTheme.colors.onSurfaceVariant
                    )
                }
                BasicTextField(
                    value = inputText,
                    onValueChange = onInputTextChanged,
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics { contentDescription = composerLabel },
                    textStyle = GlassTheme.type.bodyLarge.copy(color = GlassTheme.colors.onSurface),
                    cursorBrush = SolidColor(GlassTheme.colors.primary),
                    maxLines = 5,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Text,
                        capitalization = KeyboardCapitalization.Sentences,
                        imeAction = ImeAction.Send
                    ),
                    keyboardActions = KeyboardActions(
                        onSend = {
                            if (isGenerating) onStopGeneration() else submit()
                        }
                    )
                )
            }

            // Mic Icon Button (dictation)
            val voiceInputLabel = stringResource(
                if (isVoiceListening) R.string.voice_stop_dictation else R.string.voice_input
            )
            GCompactIconButton(
                onClick = onStartVoiceInput,
                modifier = Modifier.size(38.dp)
            ) {
                GIcon(
                    imageVector = if (isVoiceListening) GIcons.MicOff else GIcons.Mic,
                    contentDescription = voiceInputLabel,
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(21.dp)
                )
            }

            Spacer(modifier = Modifier.width(4.dp))

            // Send / Stop Action Button
            val sendDescription = stringResource(
                if (isGenerating) R.string.stop_generation else R.string.send
            )
            val sendStateDescription = stringResource(
                when {
                    isGenerating -> R.string.chat_state_streaming
                    canSend -> R.string.chat_state_ready_to_send
                    else -> R.string.chat_send_disabled_hint
                }
            )
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .semantics {
                        contentDescription = sendDescription
                        stateDescription = sendStateDescription
                    }
                    .clickable(enabled = sendEnabled, role = Role.Button) {
                        if (isGenerating) onStopGeneration() else submit()
                    },
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                isGenerating -> GlassTheme.colors.primary
                                canSend -> GlassTheme.colors.primary
                                else -> GlassTheme.colors.surfaceVariant
                            }
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    if (isGenerating) {
                        GIcon(
                            imageVector = GIcons.Stop,
                            contentDescription = null,
                            tint = GlassTheme.colors.onPrimary,
                            modifier = Modifier.size(15.dp)
                        )
                    } else {
                        GIcon(
                            imageVector = GIcons.ArrowUp,
                            contentDescription = null,
                            tint = if (canSend) GlassTheme.colors.onPrimary else GlassTheme.colors.onSurfaceVariant,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }
        }

        // Streaming / loading status strip below the capsule
        val status = ComposerStatusResolver.resolve(
            isModelLoading = isModelLoading,
            modelLoadingProgress = modelLoadingProgress,
            isGenerating = isGenerating,
            tokensPerSecond = currentTps,
            promptTokensPerSecond = promptSpeed,
            contextTokens = contextTokens,
            pendingImageInput = isPreparingImage
        )
        val statusText = when (status?.phase) {
            ComposerPhase.MODEL_LOADING -> status.progress?.let {
                stringResource(
                    R.string.chat_status_loading_progress,
                    modelLoadingStage.ifBlank { stringResource(R.string.model_loading_default) },
                    (it * 100).roundToInt()
                )
            } ?: modelLoadingStage.ifBlank { stringResource(R.string.model_loading_default) }
            ComposerPhase.PREPARING_IMAGE -> stringResource(R.string.chat_status_preparing_image)
            ComposerPhase.GENERATING -> stringResource(
                R.string.chat_status_metrics,
                String.format(Locale.US, "%.1f", status.tokensPerSecond),
                status.contextTokens
            )
            ComposerPhase.PREFILLING -> stringResource(
                R.string.chat_status_prefill,
                String.format(Locale.US, "%.1f", status.promptTokensPerSecond)
            )
            ComposerPhase.WAITING_FIRST_TOKEN -> stringResource(R.string.chat_status_generating)
            null -> null
        }
        if (statusText != null) {
            Spacer(modifier = Modifier.height(8.dp))
            if (status?.phase == ComposerPhase.MODEL_LOADING) {
                GLinearProgress(
                    progress = status.progress,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 24.dp),
                    height = 3.dp
                )
                Spacer(modifier = Modifier.height(6.dp))
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = statusText
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(4.dp)
                        .clip(RoundedCornerShape(1.dp))
                        .background(GlassTheme.colors.onSurfaceVariant)
                )
                Spacer(modifier = Modifier.width(8.dp))
                GText(
                    text = statusText,
                    style = GlassTheme.type.labelSmall.copy(
                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                    ),
                    color = GlassTheme.colors.onSurfaceVariant,
                    fontSize = 10.sp
                )
            }
        }
    }

    // Attachment Bottom Sheet
    GSheet(
        visible = showAttachmentSheet,
        onDismissRequest = { showAttachmentSheet = false }
    ) {
        GText(
            text = stringResource(R.string.chat_attach_sheet_title),
            style = GlassTheme.type.titleMedium,
            color = GlassTheme.colors.onSurface
        )

        if (pickerLaunching) {
            GText(
                text = stringResource(R.string.chat_attach_opening),
                modifier = Modifier.padding(top = 8.dp),
                style = GlassTheme.type.bodySmall,
                color = GlassTheme.colors.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        GOptionRow(
            onClick = ::launchImagePicker,
            enabled = !pickerLaunching
        ) {
            GIcon(
                imageVector = GIcons.Add,
                contentDescription = null,
                tint = GlassTheme.colors.onSurfaceVariant,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                GText(
                    text = stringResource(R.string.chat_attach_image),
                    style = GlassTheme.type.bodyLarge,
                    color = GlassTheme.colors.onSurface
                )
                GText(
                    text = stringResource(R.string.chat_attach_image_mmproj_hint),
                    style = GlassTheme.type.bodySmall,
                    color = GlassTheme.colors.onSurfaceVariant
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        GOptionRow(
            onClick = ::launchDocumentPicker,
            enabled = !pickerLaunching
        ) {
            GIcon(
                imageVector = GIcons.Description,
                contentDescription = null,
                tint = GlassTheme.colors.onSurfaceVariant,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                GText(
                    text = stringResource(R.string.chat_attach_document),
                    style = GlassTheme.type.bodyLarge,
                    color = GlassTheme.colors.onSurface
                )
                GText(
                    text = stringResource(R.string.chat_attach_document_hint),
                    style = GlassTheme.type.bodySmall,
                    color = GlassTheme.colors.onSurfaceVariant
                )
            }
        }

        if (indexedDocuments.isNotEmpty()) {
            Spacer(modifier = Modifier.height(16.dp))
            GText(
                text = stringResource(R.string.chat_library_title),
                style = GlassTheme.type.titleSmall,
                color = GlassTheme.colors.onSurface
            )
            Spacer(modifier = Modifier.height(4.dp))
            GText(
                text = stringResource(R.string.chat_library_hint),
                style = GlassTheme.type.bodySmall,
                color = GlassTheme.colors.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))
            indexedDocuments.forEach { document ->
                val included = document.id in activeDocumentIds
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .clickable { onToggleDocument(document.id) }
                        .padding(horizontal = 8.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        GText(
                            text = document.name,
                            style = GlassTheme.type.bodyMedium,
                            color = GlassTheme.colors.onSurface,
                            maxLines = 1
                        )
                        GText(
                            text = stringResource(R.string.chat_library_chunks, document.chunks),
                            style = GlassTheme.type.labelSmall,
                            color = GlassTheme.colors.onSurfaceVariant
                        )
                    }
                    GCompactIconButton(
                        onClick = { onDeleteDocument(document.id) },
                        modifier = Modifier.size(32.dp)
                    ) {
                        GIcon(
                            imageVector = GIcons.Close,
                            contentDescription = stringResource(R.string.chat_library_delete),
                            tint = GlassTheme.colors.onSurfaceVariant,
                            modifier = Modifier.size(15.dp)
                        )
                    }
                    GSwitch(
                        checked = included,
                        onCheckedChange = { onToggleDocument(document.id) },
                        modifier = Modifier.semantics { contentDescription = document.name }
                    )
                }
            }
        } else {
            Spacer(modifier = Modifier.height(16.dp))
            GText(
                text = stringResource(R.string.chat_library_empty),
                style = GlassTheme.type.bodySmall,
                color = GlassTheme.colors.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}
