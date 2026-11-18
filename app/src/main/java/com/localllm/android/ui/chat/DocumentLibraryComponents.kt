package com.localllm.android.ui.chat

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.localllm.android.R
import com.localllm.android.document.DocumentChunkPreview
import com.localllm.android.document.DocumentSummary
import com.localllm.android.ui.glass.GCompactIconButton
import com.localllm.android.ui.glass.GIcon
import com.localllm.android.ui.glass.GIcons
import com.localllm.android.ui.glass.GSwitch
import com.localllm.android.ui.glass.GText
import com.localllm.android.ui.glass.GlassTheme

/**
 * Grouped-by-file document library with an expandable per-file chunk dropdown.
 *
 * Used in both the Chat Attachment Sheet (with a per-conversation include switch)
 * and Settings -> Document Library (RAG), so users can inspect every indexed chunk
 * under its source file instead of seeing an opaque chunk count.
 */
@Composable
internal fun DocumentLibraryGroupedList(
    documents: List<DocumentSummary>,
    chunksByDocId: Map<String, List<DocumentChunkPreview>>,
    onExpandDocument: (String) -> Unit,
    onDeleteDocument: (String) -> Unit,
    activeDocumentIds: Set<String> = emptySet(),
    onToggleDocument: ((String) -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        documents.forEach { document ->
            DocumentFileDropdownCard(
                document = document,
                chunks = chunksByDocId[document.id],
                included = document.id in activeDocumentIds,
                showIncludeSwitch = onToggleDocument != null,
                onToggleInclude = { onToggleDocument?.invoke(document.id) },
                onExpand = { onExpandDocument(document.id) },
                onDelete = { onDeleteDocument(document.id) }
            )
        }
    }
}

@Composable
private fun DocumentFileDropdownCard(
    document: DocumentSummary,
    chunks: List<DocumentChunkPreview>?,
    included: Boolean,
    showIncludeSwitch: Boolean,
    onToggleInclude: () -> Unit,
    onExpand: () -> Unit,
    onDelete: () -> Unit
) {
    var expanded by remember(document.id) { mutableStateOf(false) }

    LaunchedEffect(expanded, document.id, document.chunks) {
        if (expanded) onExpand()
    }

    val shape = RoundedCornerShape(12.dp)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(GlassTheme.colors.surfaceVariant.copy(alpha = 0.45f))
            .border(1.dp, GlassTheme.colors.outline, shape)
    ) {
        val rowModifier = if (showIncludeSwitch) {
            Modifier
                .fillMaxWidth()
                .toggleable(value = included, role = Role.Switch, onValueChange = { onToggleInclude() })
                .padding(start = 10.dp, end = 4.dp, top = 4.dp, bottom = 4.dp)
        } else {
            Modifier
                .fillMaxWidth()
                .clickable { expanded = !expanded }
                .padding(start = 10.dp, end = 4.dp, top = 4.dp, bottom = 4.dp)
        }

        Row(
            modifier = rowModifier,
            verticalAlignment = Alignment.CenterVertically
        ) {
            GIcon(
                imageVector = GIcons.Description,
                contentDescription = null,
                tint = if (showIncludeSwitch && included) GlassTheme.colors.primary else GlassTheme.colors.onSurfaceVariant,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Column(modifier = Modifier.weight(1f)) {
                GText(
                    text = document.name,
                    style = GlassTheme.type.bodyMedium,
                    color = GlassTheme.colors.onSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                GText(
                    text = stringResource(R.string.chat_library_chunks, document.chunks),
                    style = GlassTheme.type.labelSmall,
                    color = GlassTheme.colors.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            GCompactIconButton(onClick = { expanded = !expanded }) {
                GIcon(
                    imageVector = if (expanded) GIcons.ArrowUp else GIcons.ArrowDown,
                    contentDescription = stringResource(
                        if (expanded) R.string.chat_library_chunks_toggle_collapse
                        else R.string.chat_library_chunks_toggle_expand
                    ),
                    tint = GlassTheme.colors.onSurfaceVariant,
                    modifier = Modifier.size(18.dp)
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
            if (showIncludeSwitch) {
                GSwitch(checked = included, onCheckedChange = null)
            }
        }

        AnimatedVisibility(visible = expanded) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(GlassTheme.colors.surface)
                    .padding(horizontal = 10.dp, vertical = 8.dp)
            ) {
                if (chunks == null) {
                    GText(
                        text = stringResource(R.string.chat_library_chunks_loading),
                        style = GlassTheme.type.labelSmall,
                        color = GlassTheme.colors.onSurfaceVariant
                    )
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 260.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        chunks.forEach { chunk ->
                            DocumentChunkDetailCard(chunk = chunk)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DocumentChunkDetailCard(chunk: DocumentChunkPreview) {
    var showFull by remember(chunk.ordinal, chunk.text.length) { mutableStateOf(false) }
    val shape = RoundedCornerShape(8.dp)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(GlassTheme.colors.surfaceVariant.copy(alpha = 0.55f))
            .border(1.dp, GlassTheme.colors.outline, shape)
            .clickable { showFull = !showFull }
            .padding(horizontal = 10.dp, vertical = 8.dp)
    ) {
        GText(
            text = stringResource(
                R.string.chat_library_chunk_header,
                chunk.ordinal + 1,
                chunk.total.coerceAtLeast(1),
                chunk.text.length
            ),
            style = GlassTheme.type.labelSmall.copy(fontFamily = FontFamily.Monospace),
            color = GlassTheme.colors.primary,
            fontSize = 11.sp
        )
        Spacer(modifier = Modifier.height(4.dp))
        GText(
            text = chunk.text,
            style = GlassTheme.type.bodySmall,
            color = GlassTheme.colors.onSurface,
            maxLines = if (showFull) Int.MAX_VALUE else 4,
            overflow = TextOverflow.Ellipsis
        )
    }
}
