package com.localllm.android.ui.chat

import android.Manifest
import android.content.ClipData
import android.content.Intent
import android.net.Uri
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.DragInteraction
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.localllm.android.R
import com.localllm.android.model.MessageRole
import com.localllm.android.ui.AppScreen
import com.localllm.android.ui.MainViewModel
import com.localllm.android.ui.drawer.ChatDrawer
import com.localllm.android.ui.glass.GDrawer
import com.localllm.android.ui.glass.GScaffold
import com.localllm.android.ui.glass.GlassTheme
import com.localllm.android.voice.InteractiveVoiceState
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import androidx.compose.runtime.withFrameNanos

@Composable
fun ChatScreen(
    viewModel: MainViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var drawerOpen by remember { mutableStateOf(false) }
    BackHandler(enabled = drawerOpen) { drawerOpen = false }

    val conversations by viewModel.conversations.collectAsStateWithLifecycle()
    val currentConvId by viewModel.currentConversationId.collectAsStateWithLifecycle()
    val messages by viewModel.messages.collectAsStateWithLifecycle()
    val settings by viewModel.settings.collectAsStateWithLifecycle()
    val activeModel by viewModel.activeModel.collectAsStateWithLifecycle()
    val allModels by viewModel.models.collectAsStateWithLifecycle()
    val isGenerating by viewModel.isGenerating.collectAsStateWithLifecycle()
    val streamingMessage by viewModel.streamingMessage.collectAsStateWithLifecycle()
    val pendingAttachment by viewModel.pendingAttachment.collectAsStateWithLifecycle()
    val voiceState by viewModel.voiceManager.voiceState.collectAsStateWithLifecycle()
    val engineStatus by viewModel.engineStatusMessage.collectAsStateWithLifecycle()
    val isModelLoading by viewModel.isModelLoading.collectAsStateWithLifecycle()
    val modelLoadingProgress by viewModel.modelLoadingProgress.collectAsStateWithLifecycle()
    val modelLoadingStage by viewModel.modelLoadingStage.collectAsStateWithLifecycle()
    val isApiModeEnabled by viewModel.isApiModeEnabled.collectAsStateWithLifecycle()
    val apiPort by viewModel.apiServerPort.collectAsStateWithLifecycle()
    val documentLibrary by viewModel.documentLibrary.collectAsStateWithLifecycle()
    val activeDocumentIds by viewModel.activeDocumentIds.collectAsStateWithLifecycle()
    val appUpdate by viewModel.appUpdate.collectAsStateWithLifecycle()
    val pendingTool by viewModel.pendingToolApproval.collectAsStateWithLifecycle()

    var inputText by remember { mutableStateOf("") }
    var showStatusDetailDialog by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()

    val followState = remember { ChatFollowState() }
    var following by remember { mutableStateOf(true) }
    fun dispatchScrollEvent(event: ChatScrollEvent): Boolean {
        val shouldScroll = followState.onEvent(event)
        following = followState.isFollowing
        return shouldScroll
    }

    val density = LocalDensity.current
    val bottomTolerancePx = remember(density) { with(density) { 12.dp.roundToPx() } }
    val isScrolledToBottom by remember(bottomTolerancePx) {
        derivedStateOf {
            val info = listState.layoutInfo
            val last = info.visibleItemsInfo.lastOrNull()
            ChatScrollPolicy.isAtBottom(
                lastVisibleIndex = last?.index ?: -1,
                lastVisibleEndOffset = (last?.offset ?: 0) + (last?.size ?: 0),
                totalItems = info.totalItemsCount,
                viewportEndOffset = info.viewportEndOffset,
                tolerancePx = bottomTolerancePx
            )
        }
    }

    suspend fun scrollTranscriptToEnd() {
        withFrameNanos { }
        val totalCount = listState.layoutInfo.totalItemsCount
        if (totalCount > 0) listState.scrollToItem(totalCount - 1)
    }

    LaunchedEffect(listState) {
        listState.interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is DragInteraction.Start -> dispatchScrollEvent(ChatScrollEvent.USER_DRAG_STARTED)
                is DragInteraction.Stop, is DragInteraction.Cancel -> dispatchScrollEvent(
                    if (isScrolledToBottom) ChatScrollEvent.USER_DRAG_ENDED_AT_BOTTOM
                    else ChatScrollEvent.USER_DRAG_ENDED_ABOVE_BOTTOM
                )
            }
        }
    }

    LaunchedEffect(listState) {
        snapshotFlow { listState.isScrollInProgress to isScrolledToBottom }
            .distinctUntilChanged()
            .collect { (inProgress, atBottom) ->
                if (!inProgress && atBottom) dispatchScrollEvent(ChatScrollEvent.SETTLED_AT_BOTTOM)
            }
    }

    LaunchedEffect(streamingMessage?.content?.length) {
        if (streamingMessage != null && dispatchScrollEvent(ChatScrollEvent.STREAMING_UPDATE)) {
            scrollTranscriptToEnd()
        }
    }

    LaunchedEffect(messages.lastOrNull()?.id) {
        if (messages.lastOrNull()?.role == MessageRole.USER) {
            dispatchScrollEvent(ChatScrollEvent.PROMPT_SUBMITTED)
            scrollTranscriptToEnd()
        }
    }

    val pendingImageTurn = messages.lastOrNull { it.role == MessageRole.USER }?.attachment?.isImage == true
    val isPreparingImage = isGenerating && pendingImageTurn && streamingMessage?.content.isNullOrBlank()

    val micPermissionDeniedMsg = stringResource(R.string.chat_voice_mic_permission_required)
    val micPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.voiceManager.startListening(
                onResult = { recognizedText -> inputText = recognizedText },
                onError = { err -> Toast.makeText(context, err, Toast.LENGTH_SHORT).show() }
            )
        } else {
            Toast.makeText(context, micPermissionDeniedMsg, Toast.LENGTH_SHORT).show()
        }
    }

    val contextUsageTokens = streamingMessage?.contextTokens ?: messages.lastOrNull()?.contextTokens ?: 0

    fun writeExport(uri: Uri, format: String, displayName: String, mimeType: String) {
        Toast.makeText(context, context.getString(R.string.chat_export_saving), Toast.LENGTH_SHORT).show()
        viewModel.exportCurrentConversation(uri, format, streamingMessage) { result ->
            result.fold(
                onSuccess = {
                    Toast.makeText(
                        context,
                        context.getString(R.string.chat_export_success, displayName),
                        Toast.LENGTH_LONG
                    ).show()
                    val shareIntent = Intent(Intent.ACTION_SEND).apply {
                        type = mimeType
                        putExtra(Intent.EXTRA_STREAM, uri)
                        clipData = ClipData.newUri(context.contentResolver, displayName, uri)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    runCatching {
                        context.startActivity(
                            Intent.createChooser(
                                shareIntent,
                                context.getString(R.string.chat_export_share_title)
                            )
                        )
                    }
                },
                onFailure = { error ->
                    Toast.makeText(
                        context,
                        context.getString(
                            R.string.chat_export_failed,
                            error.localizedMessage ?: context.getString(R.string.chat_export_write_failed)
                        ),
                        Toast.LENGTH_LONG
                    ).show()
                }
            )
        }
    }

    val markdownExportLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/markdown")
    ) { uri ->
        if (uri != null) writeExport(uri, "md", "offcon_conversation.md", "text/markdown")
    }
    val jsonExportLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/json")
    ) { uri ->
        if (uri != null) writeExport(uri, "json", "offcon_conversation.json", "application/json")
    }

    GDrawer(
        open = drawerOpen,
        onClose = { drawerOpen = false },
        drawerContent = {
            ChatDrawer(
                conversations = conversations,
                currentConversationId = currentConvId,
                onSelectConversation = { id ->
                    viewModel.selectConversation(id)
                    drawerOpen = false
                },
                onNewChat = {
                    viewModel.createNewConversation()
                    drawerOpen = false
                },
                onDeleteConversation = viewModel::deleteConversation,
                onRenameConversation = viewModel::renameConversation,
                onOpenModelManager = {
                    viewModel.navigateTo(AppScreen.MODELS)
                    drawerOpen = false
                },
                onOpenVoiceMode = {
                    viewModel.navigateTo(AppScreen.VOICE_MODE)
                    drawerOpen = false
                },
                onOpenApiMode = {
                    viewModel.navigateTo(AppScreen.API_MODE)
                    drawerOpen = false
                },
                onOpenSettings = {
                    viewModel.navigateTo(AppScreen.SETTINGS)
                    drawerOpen = false
                }
            )
        },
        modifier = modifier.fillMaxSize()
    ) {
        GScaffold(
            topBar = {
                ChatTopBar(
                    activeModel = activeModel,
                    allModels = allModels,
                    currentRuntime = settings.runtime,
                    isMtpOn = settings.enableMtp,
                    onOpenDrawer = { drawerOpen = true },
                    onNewChat = viewModel::createNewConversation,
                    onSelectModel = viewModel::selectModel,
                    onOpenSettings = { viewModel.navigateTo(AppScreen.SETTINGS) },
                    onSwitchRuntime = viewModel::switchRuntime,
                    onOpenModelManager = { viewModel.navigateTo(AppScreen.MODELS) },
                    onOpenApiMode = { viewModel.navigateTo(AppScreen.API_MODE) },
                    onExportMarkdown = {
                        if (messages.isEmpty() && streamingMessage == null) {
                            Toast.makeText(context, context.getString(R.string.chat_export_empty), Toast.LENGTH_SHORT).show()
                        } else {
                            markdownExportLauncher.launch("offcon_conversation.md")
                        }
                    },
                    onExportJson = {
                        if (messages.isEmpty() && streamingMessage == null) {
                            Toast.makeText(context, context.getString(R.string.chat_export_empty), Toast.LENGTH_SHORT).show()
                        } else {
                            jsonExportLauncher.launch("offcon_conversation.json")
                        }
                    },
                    contextUsageTokens = contextUsageTokens
                )
            },
            bottomBar = {
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Box(modifier = Modifier.fillMaxWidth().widthIn(max = 680.dp)) {
                        ChatInputBar(
                            inputText = inputText,
                            onInputTextChanged = { inputText = it },
                            onSendMessage = {
                                val text = inputText
                                if (viewModel.sendMessage(text)) {
                                    inputText = ""
                                    dispatchScrollEvent(ChatScrollEvent.PROMPT_SUBMITTED)
                                    scope.launch { scrollTranscriptToEnd() }
                                }
                            },
                            onStopGeneration = viewModel::stopGeneration,
                            isGenerating = isGenerating,
                            activeModel = activeModel,
                            pendingAttachment = pendingAttachment,
                            onSetAttachment = viewModel::setPendingAttachment,
                            reasoningEffort = settings.reasoningEffort,
                            onReasoningEffortChanged = viewModel::setReasoningEffort,
                            onStartVoiceInput = {
                                if (voiceState == InteractiveVoiceState.LISTENING) {
                                    viewModel.voiceManager.stopListening()
                                } else {
                                    val hasMic = ContextCompat.checkSelfPermission(
                                        context,
                                        Manifest.permission.RECORD_AUDIO
                                    ) == PackageManager.PERMISSION_GRANTED
                                    if (hasMic) {
                                        viewModel.voiceManager.startListening(
                                            onResult = { text -> inputText = text },
                                            onError = { err -> Toast.makeText(context, err, Toast.LENGTH_SHORT).show() }
                                        )
                                    } else {
                                        micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                    }
                                }
                            },
                            onOpenVoiceMode = { viewModel.navigateTo(AppScreen.VOICE_MODE) },
                            onDocumentPicked = viewModel::importDocument,
                            indexedDocuments = documentLibrary,
                            activeDocumentIds = activeDocumentIds,
                            onToggleDocument = viewModel::toggleDocumentSelection,
                            onDeleteDocument = viewModel::deleteIndexedDocument,
                            isModelLoading = isModelLoading,
                            modelLoadingProgress = modelLoadingProgress,
                            modelLoadingStage = modelLoadingStage,
                            currentTps = streamingMessage?.tps ?: 0f,
                            promptSpeed = streamingMessage?.promptSpeed ?: 0f,
                            contextTokens = contextUsageTokens,
                            isVoiceListening = voiceState == InteractiveVoiceState.LISTENING,
                            isPreparingImage = isPreparingImage
                        )
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(GlassTheme.colors.background)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .widthIn(max = 680.dp)
                        .align(Alignment.TopCenter)
                        .fillMaxHeight()
                ) {
                    ModelLoadingBanner(
                        visible = isModelLoading,
                        stage = modelLoadingStage,
                        progress = modelLoadingProgress
                    )

                    ApiModeBanner(
                        visible = isApiModeEnabled,
                        apiPort = apiPort,
                        onOpenApiMode = { viewModel.navigateTo(AppScreen.API_MODE) }
                    )

                    appUpdate?.let { update ->
                        UpdateBanner(
                            update = update,
                            onDismiss = viewModel::dismissAppUpdate
                        )
                    }

                    EngineStatusStrip(
                        status = engineStatus,
                        onShowDetails = { showStatusDetailDialog = true }
                    )

                    pendingTool?.let { call ->
                        McpToolApprovalDialog(
                            call = call,
                            onResolve = viewModel::resolveToolApproval
                        )
                    }

                    if (showStatusDetailDialog && !engineStatus.isNullOrBlank()) {
                        EngineStatusDetailDialog(
                            context = context,
                            status = engineStatus!!,
                            onDismiss = { showStatusDetailDialog = false }
                        )
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                    ) {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize()
                        ) {
                            if (messages.isEmpty() && streamingMessage == null) {
                                item {
                                    EmptyChatPlaceholder(
                                        activeModel = activeModel,
                                        contextWindow = settings.contextWindow,
                                        onOpenModelManager = { viewModel.navigateTo(AppScreen.MODELS) },
                                        onSampleClick = { sampleText ->
                                            if (activeModel == null) {
                                                viewModel.navigateTo(AppScreen.MODELS)
                                            } else {
                                                viewModel.sendMessage(sampleText)
                                            }
                                        }
                                    )
                                }
                            } else {
                                items(messages, key = { it.id }) { msg ->
                                    ChatMessageItem(
                                        message = msg,
                                        showMetrics = settings.showPerformanceMetrics,
                                        isMtpOn = settings.enableMtp && (activeModel?.supportsMtp == true),
                                        reasoningEffortLabel = settings.reasoningEffortLabel,
                                        onSpeak = viewModel.voiceManager::speak
                                    )
                                }

                                streamingMessage?.let { activeStream ->
                                    item(key = "streaming_current_message") {
                                        ChatMessageItem(
                                            message = activeStream,
                                            showMetrics = settings.showPerformanceMetrics,
                                            isMtpOn = settings.enableMtp && (activeModel?.supportsMtp == true),
                                            reasoningEffortLabel = settings.reasoningEffortLabel,
                                            onSpeak = viewModel.voiceManager::speak
                                        )
                                    }
                                }

                                item(key = "chat_bottom_anchor") {
                                    Spacer(modifier = Modifier.fillMaxWidth().height(28.dp))
                                }
                            }
                        }

                        val jumpVisible = !following && (streamingMessage != null || isGenerating)
                        val jumpAlpha by animateFloatAsState(
                            targetValue = if (jumpVisible) 1f else 0f,
                            label = "jumpAlpha"
                        )
                        if (jumpAlpha > 0.01f) {
                            Box(
                                modifier = Modifier
                                    .align(Alignment.BottomCenter)
                                    .padding(bottom = 12.dp)
                                    .alpha(jumpAlpha)
                            ) {
                                JumpToLatestButton(
                                    onClick = {
                                        dispatchScrollEvent(ChatScrollEvent.JUMP_TO_LATEST)
                                        scope.launch { scrollTranscriptToEnd() }
                                    }
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
