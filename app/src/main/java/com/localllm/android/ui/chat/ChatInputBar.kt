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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.toggleable
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
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
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

private const val PICKER_IMAGE = "image"
private const val PICKER_DOCUMENT = "document"

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
    /** A turn that carries an image is running and no token has arrived yet. */
    isPreparingImage: Boolean = false,
    modifier: Modifier = Modifier
) {
    var showAttachmentSheet by remember { mutableStateOf(false) }
    var showReasoningSlider by remember { mutableStateOf(false) }
    var pickerLaunching by remember { mutableStateOf(false) }
    val pickerGuard = remember { PickerLaunchGuard() }
    val context = LocalContext.current
    val isDarkCanvas = GlassTheme.colors.background.luminance() < 0.5f
    val focusRequester = remember { FocusRequester() }
    var composerFocused by remember { mutableStateOf(false) }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        // Every result — including cancel — releases the single-flight guard.
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
        if (!pickerGuard.tryLaunch(PICKER_IMAGE)) return
        pickerLaunching = true
        runCatching {
            imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }.onFailure {
            // No picker activity (or the launch was rejected): never leave the sheet stuck.
            pickerGuard.onResult()
            pickerLaunching = false
        }
    }

    fun launchDocumentPicker() {
        if (!pickerGuard.tryLaunch(PICKER_DOCUMENT)) return
        pickerLaunching = true
        runCatching {
            documentPicker.launch(arrayOf("text/*", "application/pdf", "*/*"))
        }.onFailure {
            pickerGuard.onResult()
            pickerLaunching = false
        }
    }

    val canSend = inputText.isNotBlank() || pendingAttachment != null
    val sendEnabled = isGenerating || canSend

    fun submit() {
        if (!canSend) return
        onSendMessage()
        // Keep the keyboard up for the next turn instead of dropping focus on send.
        if (composerFocused) focusRequester.requestFocus()
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
            val tier = ReasoningEffortTier.of(reasoningEffort)
            val tierLabel = stringResource(tier.labelRes)
            val pillText = stringResource(
                R.string.chat_reasoning_effort_label,
                stringResource(
                    R.string.chat_reasoning_effort_with_value,
                    tierLabel,
                    String.format(Locale.US, "%.1f", reasoningEffort)
                )
            )
            val pillState = stringResource(
                if (showReasoningSlider) R.string.chat_state_expanded else R.string.chat_state_collapsed
            )
            val pillDescription = stringResource(R.string.chat_reasoning_effort_expand)
            // The pill keeps its compact look; the 48dp row around it is the touch target.
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .padding(bottom = 2.dp),
                contentAlignment = Alignment.CenterStart
            ) {
                Row(
                    modifier = Modifier
                        .clip(RoundedCornerShape(12.dp))
                        .background(GlassTheme.colors.secondary)
                        .semantics {
                            contentDescription = pillDescription
                            stateDescription = pillState
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
                        text = pillText,
                        style = GlassTheme.type.labelSmall,
                        color = GlassTheme.colors.onSurface
                    )
                }
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
                    .fillMaxWidth()
                    .padding(bottom = 8.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(GlassTheme.colors.surface)
                    .border(1.dp, GlassTheme.colors.outline, RoundedCornerShape(12.dp))
                    .padding(start = 8.dp, top = 8.dp, bottom = 8.dp, end = 4.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (pendingAttachment.isImage) {
                        AsyncImage(
                            model = pendingAttachment.uriString,
                            contentDescription = stringResource(R.string.chat_attached_image_desc),
                            modifier = Modifier
                                .size(48.dp)
                                .clip(RoundedCornerShape(6.dp))
                        )
                    } else {
                        GIcon(
                            imageVector = GIcons.Description,
                            contentDescription = stringResource(R.string.chat_attachment_file),
                            tint = GlassTheme.colors.onSurfaceVariant,
                            modifier = Modifier.size(32.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    Column(modifier = Modifier.weight(1f)) {
                        GText(
                            text = pendingAttachment.fileName,
                            style = GlassTheme.type.bodySmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        GText(
                            text = stringResource(
                                if (pendingAttachment.isImage) R.string.chat_attachment_image
                                else R.string.chat_attachment_file
                            ),
                            fontSize = 11.sp,
                            color = GlassTheme.colors.onSurfaceVariant,
                            maxLines = 1
                        )
                    }

                    GCompactIconButton(onClick = { onSetAttachment(null) }) {
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
        val capsuleBorder = if (composerFocused) GlassTheme.colors.primary.copy(alpha = 0.55f)
        else GlassTheme.colors.outline
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
                .border(1.dp, capsuleBorder, capsuleShape)
                .padding(start = 4.dp, end = 4.dp, top = 2.dp, bottom = 2.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Attachment Icon Button (48dp touch area, unchanged glyph size)
            GCompactIconButton(onClick = { showAttachmentSheet = true }) {
                GIcon(
                    imageVector = GIcons.Add,
                    contentDescription = stringResource(R.string.attach_file),
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(21.dp)
                )
            }

            // Hands-free (live) voice mode trigger
            GCompactIconButton(onClick = onOpenVoiceMode) {
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
                        .focusRequester(focusRequester)
                        .onFocusChanged { composerFocused = it.isFocused },
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
            GCompactIconButton(onClick = onStartVoiceInput) {
                GIcon(
                    imageVector = GIcons.Mic,
                    contentDescription = stringResource(R.string.voice_input),
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(21.dp)
                )
            }

            Spacer(modifier = Modifier.width(2.dp))

            // Send / Stop Action Button: 40dp visual inside a 48dp touch target.
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
                    .clip(CircleShape)
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
                            if (sendEnabled) GlassTheme.colors.primary
                            else GlassTheme.colors.surfaceVariant
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
                            tint = if (sendEnabled) GlassTheme.colors.onPrimary
                            else GlassTheme.colors.onSurfaceVariant,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }
        }

        // 4. Streaming / loading status strip below the capsule
        val status = ComposerStatusResolver.resolve(
            isModelLoading = isModelLoading,
            modelLoadingProgress = modelLoadingProgress,
            isGenerating = isGenerating,
            tokensPerSecond = currentTps,
            promptTokensPerSecond = promptSpeed,
            contextTokens = contextTokens,
            pendingImageInput = isPreparingImage
        )
        if (status != null) {
            Spacer(modifier = Modifier.height(8.dp))
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    modifier = Modifier.semantics {
                        // Announce real state changes, but never the per-chunk token rate.
                        if (status.phase == ComposerPhase.MODEL_LOADING ||
                            status.phase == ComposerPhase.PREPARING_IMAGE
                        ) {
                            liveRegion = LiveRegionMode.Polite
                        }
                    },
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
                        text = composerStatusText(status, modelLoadingStage),
                        style = GlassTheme.type.labelSmall.copy(fontFamily = FontFamily.Monospace),
                        color = GlassTheme.colors.onSurfaceVariant,
                        fontSize = 10.sp,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                if (status.phase == ComposerPhase.MODEL_LOADING) {
                    Spacer(modifier = Modifier.height(4.dp))
                    GLinearProgress(
                        progress = status.progress,
                        modifier = Modifier
                            .fillMaxWidth(0.6f)
                            .height(2.dp)
                            .clip(RoundedCornerShape(1.dp)),
                        color = GlassTheme.colors.primary,
                        trackColor = GlassTheme.colors.surfaceVariant,
                        height = 2.dp
                    )
                }
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

        Spacer(modifier = Modifier.height(12.dp))

        GOptionRow(
            onClick = { launchImagePicker() },
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

        Spacer(modifier = Modifier.height(12.dp))

        GOptionRow(
            onClick = { launchDocumentPicker() },
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

        if (pickerLaunching) {
            Spacer(modifier = Modifier.height(8.dp))
            GText(
                text = stringResource(R.string.chat_attach_opening),
                style = GlassTheme.type.labelSmall,
                color = GlassTheme.colors.primary
            )
        }

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

        if (indexedDocuments.isEmpty()) {
            GText(
                text = stringResource(R.string.chat_library_empty),
                style = GlassTheme.type.bodySmall,
                color = GlassTheme.colors.onSurfaceVariant
            )
        } else {
            indexedDocuments.forEach { document ->
                DocumentLibraryRow(
                    name = document.name,
                    chunks = document.chunks,
                    included = document.id in activeDocumentIds,
                    onToggle = { onToggleDocument(document.id) },
                    onDelete = { onDeleteDocument(document.id) }
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
    }
}

/**
 * One indexed document.
 *
 * The row is the single toggleable node (Role.Switch) and the trailing [GSwitch] is a
 * non-interactive indicator, so the row tap and the switch never fight over the same tap.
 * The delete action stays a separate 48dp button with its own label, and the file name
 * ellipsizes inside the weighted column instead of pushing the controls off-screen.
 */
@Composable
private fun DocumentLibraryRow(
    name: String,
    chunks: Int,
    included: Boolean,
    onToggle: () -> Unit,
    onDelete: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .toggleable(value = included, role = Role.Switch, onValueChange = { onToggle() })
            .padding(start = 8.dp, end = 2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            GText(
                text = name,
                style = GlassTheme.type.bodyMedium,
                color = GlassTheme.colors.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            GText(
                text = stringResource(R.string.chat_library_chunks, chunks),
                style = GlassTheme.type.labelSmall,
                color = GlassTheme.colors.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        GCompactIconButton(onClick = onDelete) {
            GIcon(
                imageVector = GIcons.Close,
                contentDescription = stringResource(R.string.chat_library_delete),
                tint = GlassTheme.colors.onSurfaceVariant,
                modifier = Modifier.size(15.dp)
            )
        }
        GSwitch(checked = included, onCheckedChange = null)
    }
}

/** Localized status copy for one resolved [ComposerStatus]. */
@Composable
private fun composerStatusText(status: ComposerStatus, stage: String): String = when (status.phase) {
    ComposerPhase.MODEL_LOADING -> {
        val stageText = stage.ifBlank { stringResource(R.string.model_loading_default) }
        val progress = status.progress
        if (progress == null) {
            stageText
        } else {
            stringResource(
                R.string.chat_status_loading_progress,
                stageText,
                (progress * 100f).roundToInt()
            )
        }
    }
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
}
