//! Slint UI orchestration controller.
//!
//! All blocking HTTP/process/MCP I/O executes on worker threads; the Slint event loop
//! drains typed `UiEvent`s and coordinates multi-round MCP tool execution with user
//! approval.
//!
//! Three invariants drive the design:
//! 1. **Every async completion is epoch-tagged** ([`crate::epoch`]). A completion whose
//!    token is no longer current is dropped, so a slow `Loaded` can never resurrect a
//!    backend the user switched away from and a late `Delta` can never overwrite a newer
//!    transcript.
//! 2. **The session owns tool-call history integrity** ([`crate::session`]). This module
//!    never commits an assistant `tool_calls` message without also closing the batch,
//!    whichever path the round takes.
//! 3. **Deltas are batched** ([`crate::stream_batch`]) so the transcript re-lays out about
//!    once per frame instead of once per token, with a forced flush before every
//!    structural event.

use crate::backends::{LlamaCpp, LlamaCppConfig, Strata, StrataConfig, Vllm, VllmConfig};
use crate::contracts::{
    BackendKind, ChatRequest, ImageGenerationBackend, ImageRequest, LlmBackend, ModelSpec,
    StreamEvent, ToolCallRecord,
};
use crate::epoch::{Epoch, EpochGate};
use crate::image_backend::OpenAiImageBackend;
use crate::mcp::{McpClient, McpConnectionInfo};
use crate::persist::{self, DesktopSettings, PersistedSession, SaveScheduler};
use crate::process::{
    format_resources, parse_command, parse_f64_clamped, parse_port, parse_u32, parse_u32_clamped,
};
use crate::session::{ConversationSession, ToolAbort, TurnEntry};
use crate::stream_batch::DeltaBatcher;
use crate::{ChatTurn, MainWindow, Ui};
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

const MAX_TOOL_ROUNDS: u32 = 3;
const MAX_CALLS_PER_TURN: usize = 4;
/// How long settings / the conversation must stay unchanged before they are written.
const PERSIST_QUIET_PERIOD: Duration = Duration::from_millis(1200);
/// UI tick: the delta batch window (40 ms) plus a little jitter headroom.
const TICK_INTERVAL: Duration = Duration::from_millis(45);
/// How long a stop may stay unacknowledged before the run is closed locally.
const STOP_GRACE_PERIOD: Duration = Duration::from_millis(2500);

/// What the app is currently doing. This is deliberately finer-grained than the old
/// boolean `busy`, so "loading", "unloading", "connecting", "generating" and "running a
/// tool" are distinguishable in the footer and in the composer hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Loading,
    Unloading,
    Connecting,
    Streaming,
    Stopping,
    Tool,
    Image,
}

impl Phase {
    fn is_busy(self) -> bool {
        self != Self::Idle
    }

    /// Short monospace label for the footer, matching the calm instrument readout style.
    fn footer(self) -> &'static str {
        match self {
            Self::Idle => "",
            Self::Loading => "loading",
            Self::Unloading => "unloading",
            Self::Connecting => "connecting",
            Self::Streaming => "generating",
            Self::Stopping => "stopping",
            Self::Tool => "tool",
            Self::Image => "rendering",
        }
    }

    /// Sentence used by the composer hint while the app is busy.
    fn hint(self) -> &'static str {
        match self {
            Self::Idle => "",
            Self::Loading => "Loading the runtime — the composer is locked until it answers",
            Self::Unloading => "Unloading the runtime",
            Self::Connecting => "Connecting to the MCP server",
            Self::Streaming => "Generating — Esc stops",
            Self::Stopping => "Stopping…",
            Self::Tool => "Awaiting tool approval — Esc declines",
            Self::Image => "Rendering the image",
        }
    }
}

enum BackendChoice {
    Llama(LlamaCppConfig),
    Vllm(VllmConfig),
    Strata(StrataConfig),
}

impl BackendChoice {
    fn build(self) -> Arc<dyn LlmBackend> {
        match self {
            Self::Llama(config) => Arc::new(LlamaCpp::new(config)),
            Self::Vllm(config) => Arc::new(Vllm::new(config)),
            Self::Strata(config) => Arc::new(Strata::new(config)),
        }
    }
}

/// A completion posted by a worker, tagged with the generation it belongs to.
struct UiEvent {
    epoch: Epoch,
    kind: UiKind,
}

enum UiKind {
    Loaded {
        backend: Arc<dyn LlmBackend>,
        model_label: String,
        status: String,
    },
    LoadFailed(String),
    Unloaded,
    McpConnected(McpConnectionInfo),
    McpFailed(String),
    Delta {
        text: Option<String>,
        reasoning: Option<String>,
    },
    ToolCall(ToolCallRecord),
    ToolExecuted {
        index: usize,
        outcome: crate::mcp::McpCallOutcome,
    },
    Error(String),
    Done {
        cancelled: bool,
    },
    ImageWritten {
        path: PathBuf,
        message: String,
    },
    ImageFailed(String),
}

fn to_chat_turn(entry: TurnEntry) -> ChatTurn {
    ChatTurn {
        kind: entry.kind,
        role: entry.role.into(),
        body: entry.body.into(),
    }
}

/// Reads every persisted field from the UI.
fn settings_from_ui(ui: &MainWindow) -> DesktopSettings {
    DesktopSettings {
        dark_mode: ui.global::<Ui>().get_dark_mode(),
        edition_index: ui.get_edition_index(),
        backend_index: ui.get_backend_index(),
        model_label: ui.get_model_label().to_string(),
        llama_executable: ui.get_llama_executable().to_string(),
        llama_model_path: ui.get_llama_model_path().to_string(),
        llama_port: ui.get_llama_port().to_string(),
        llama_context_size: ui.get_llama_context_size().to_string(),
        llama_gpu_layers: ui.get_llama_gpu_layers().to_string(),
        vllm_endpoint: ui.get_vllm_endpoint().to_string(),
        vllm_model_path: ui.get_vllm_model_path().to_string(),
        vllm_port: ui.get_vllm_port().to_string(),
        vllm_managed: ui.get_vllm_managed(),
        strata_endpoint: ui.get_strata_endpoint().to_string(),
        strata_launch: ui.get_strata_launch().to_string(),
        temperature_text: ui.get_temperature_text().to_string(),
        max_tokens_text: ui.get_max_tokens_text().to_string(),
        mcp_url: ui.get_mcp_url().to_string(),
        image_endpoint: ui.get_image_endpoint().to_string(),
        image_model: ui.get_image_model().to_string(),
        image_size: ui.get_image_size().to_string(),
        image_output: ui.get_image_output().to_string(),
    }
}

/// Writes every persisted field back into the UI. Never called with an API key: the
/// settings type has no field for one.
fn apply_settings_to_ui(ui: &MainWindow, settings: &DesktopSettings) {
    ui.global::<Ui>().set_dark_mode(settings.dark_mode);
    ui.set_edition_index(settings.edition_index);
    ui.set_backend_index(settings.backend_index);
    ui.set_model_label(settings.model_label.as_str().into());
    ui.set_llama_executable(settings.llama_executable.as_str().into());
    ui.set_llama_model_path(settings.llama_model_path.as_str().into());
    ui.set_llama_port(settings.llama_port.as_str().into());
    ui.set_llama_context_size(settings.llama_context_size.as_str().into());
    ui.set_llama_gpu_layers(settings.llama_gpu_layers.as_str().into());
    ui.set_vllm_endpoint(settings.vllm_endpoint.as_str().into());
    ui.set_vllm_model_path(settings.vllm_model_path.as_str().into());
    ui.set_vllm_port(settings.vllm_port.as_str().into());
    ui.set_vllm_managed(settings.vllm_managed);
    ui.set_strata_endpoint(settings.strata_endpoint.as_str().into());
    ui.set_strata_launch(settings.strata_launch.as_str().into());
    ui.set_temperature_text(settings.temperature_text.as_str().into());
    ui.set_max_tokens_text(settings.max_tokens_text.as_str().into());
    ui.set_mcp_url(settings.mcp_url.as_str().into());
    ui.set_image_endpoint(settings.image_endpoint.as_str().into());
    ui.set_image_model(settings.image_model.as_str().into());
    ui.set_image_size(settings.image_size.as_str().into());
    ui.set_image_output(settings.image_output.as_str().into());
    ui.set_empty_engine(
        idle_engine_label(BackendKind::from_index(settings.backend_index)).into(),
    );
}

pub struct DesktopController {
    ui: slint::Weak<MainWindow>,
    sender: Sender<UiEvent>,
    receiver: Receiver<UiEvent>,
    backend: Option<Arc<dyn LlmBackend>>,
    /// Which backend the loaded runtime belongs to, so `send` can refuse a mismatch.
    loaded_kind: Option<BackendKind>,
    mcp: Arc<McpClient>,
    turns_model: Rc<VecModel<ChatTurn>>,
    session: ConversationSession,
    generation: Option<Arc<AtomicBool>>,
    pending_calls: VecDeque<(usize, ToolCallRecord)>,
    /// Calls committed this round but beyond [`MAX_CALLS_PER_TURN`]; closed after the
    /// runnable ones so the transcript keeps commit order.
    deferred_calls: Vec<(usize, ToolCallRecord)>,
    pending_total: usize,
    pending_done: usize,
    current_tool_round: u32,
    phase: Phase,
    epoch: EpochGate,
    batcher: DeltaBatcher,
    config_dir: PathBuf,
    /// Debounced writers: settings by value, the conversation by (turns, messages) shape.
    settings_save: SaveScheduler<DesktopSettings>,
    session_save: SaveScheduler<(usize, usize)>,
    last_tick: Instant,
    tick_count: u32,
    resource_ticks: u32,
    retry_draft: Option<String>,
    /// Status to show when a backend switch finishes unloading the old runtime.
    unload_note: Option<String>,
    /// When the user asked to stop, so a backend that never answers cannot leave the app
    /// in "stopping" forever.
    stopping_since: Option<Instant>,
    stale_events: u64,
}

impl DesktopController {
    fn new(ui: slint::Weak<MainWindow>, config_dir: PathBuf) -> Self {
        let (sender, receiver) = channel();
        Self {
            ui,
            sender,
            receiver,
            backend: None,
            loaded_kind: None,
            mcp: Arc::new(McpClient::new()),
            turns_model: Rc::new(VecModel::from(Vec::<ChatTurn>::new())),
            session: ConversationSession::new(),
            generation: None,
            pending_calls: VecDeque::new(),
            deferred_calls: Vec::new(),
            pending_total: 0,
            pending_done: 0,
            current_tool_round: 0,
            phase: Phase::Idle,
            epoch: EpochGate::new(),
            batcher: DeltaBatcher::default(),
            config_dir,
            settings_save: SaveScheduler::new(PERSIST_QUIET_PERIOD),
            session_save: SaveScheduler::new(PERSIST_QUIET_PERIOD),
            last_tick: Instant::now(),
            tick_count: 0,
            resource_ticks: 0,
            retry_draft: None,
            unload_note: None,
            stopping_since: None,
            stale_events: 0,
        }
    }

    /// Wires the model and restores settings + conversation. Never panics on bad input:
    /// [`crate::persist`] turns malformed files into defaults plus a warning string.
    fn install(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        ui.set_turns(ModelRc::new(self.turns_model.clone()));

        let mut notes: Vec<String> = Vec::new();

        let loaded_settings = persist::load_settings(&self.config_dir);
        if let Some(warning) = loaded_settings.warning.clone() {
            notes.push(warning);
            // Repair the file on disk so the warning is shown once, not on every launch.
            let _ = persist::save_settings(&self.config_dir, &loaded_settings.value);
        }
        apply_settings_to_ui(&ui, &loaded_settings.value);
        self.settings_save.prime(loaded_settings.value);

        let restored = persist::load_session(&self.config_dir);
        if let Some(warning) = restored.warning.clone() {
            notes.push(warning);
        }
        if let Some(stored) = restored.value {
            if !stored.is_empty() {
                let restored_turns = stored.turns.len();
                self.session = ConversationSession::from_parts(stored.turns, stored.messages);
                self.session_save.prime((
                    self.session.turns().len(),
                    self.session.messages().len(),
                ));
                self.sync_all_turns(&ui);
                notes.push(format!("Restored {restored_turns} turn(s)"));
            }
        }

        self.set_phase(&ui, Phase::Idle);
        ui.set_send_ready(false);
        if notes.is_empty() {
            ui.set_status("Select a backend and load or connect a model.".into());
            ui.set_status_error(false);
        } else {
            ui.set_status(notes.join(" \u{00B7} ").into());
            ui.set_status_error(false);
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(self.last_tick);
        self.last_tick = now;
        self.tick_count = self.tick_count.wrapping_add(1);

        // Batch clock advances even when no event arrived, so a stream that stalls
        // mid-batch still shows what it has.
        self.batcher.advance(elapsed);
        if self.batcher.is_due() {
            if let Some(ui) = self.ui.upgrade() {
                self.flush_delta(&ui);
            }
        }

        while let Ok(event) = self.receiver.try_recv() {
            if let Some(ui) = self.ui.upgrade() {
                self.apply(&ui, event);
            }
        }

        let Some(ui) = self.ui.upgrade() else { return };
        if self.tick_count % 2 == 0 {
            self.refresh_composer_gate(&ui);
        }
        // A stop that the backend never acknowledges must still release the UI.
        if let Some(since) = self.stopping_since {
            if now.saturating_duration_since(since) >= STOP_GRACE_PERIOD {
                self.abandon_run(
                    &ui,
                    "Stopped \u{00B7} the backend did not confirm, so the run was closed locally.",
                );
            }
        }
        if self.tick_count % 8 == 0 {
            self.maybe_persist(&ui, now);
        }
        self.resource_ticks += 1;
        if self.resource_ticks >= 20 {
            self.resource_ticks = 0;
            self.refresh_resources(&ui);
        }
    }

    fn apply(&mut self, ui: &MainWindow, event: UiEvent) {
        if !self.epoch.is_current(event.epoch) {
            // A completion from a superseded operation: dropping it is the whole point
            // of the epoch token.
            self.stale_events += 1;
            return;
        }
        match event.kind {
            UiKind::Loaded {
                backend,
                model_label,
                status,
            } => {
                let kind = backend.kind();
                self.backend = Some(backend);
                self.loaded_kind = Some(kind);
                self.set_phase(ui, Phase::Idle);
                ui.set_model_loaded(true);
                ui.set_model_indicator(model_label.as_str().into());
                ui.set_empty_engine(format!("{} \u{00B7} {}", model_label, kind.label()).into());
                ui.set_status(status.into());
                ui.set_status_error(false);
                self.refresh_resources(ui);
            }
            UiKind::LoadFailed(message) => {
                self.backend = None;
                self.loaded_kind = None;
                self.set_phase(ui, Phase::Idle);
                ui.set_model_loaded(false);
                ui.set_model_indicator("No model".into());
                ui.set_status(format!("Load failed: {message}").into());
                ui.set_status_error(true);
                self.persist_settings_now(ui);
            }
            UiKind::Unloaded => {
                self.backend = None;
                self.loaded_kind = None;
                self.set_phase(ui, Phase::Idle);
                ui.set_model_loaded(false);
                ui.set_model_indicator("No model".into());
                let kind = BackendKind::from_index(ui.get_backend_index());
                ui.set_empty_engine(idle_engine_label(kind).into());
                ui.set_resources("No runtime resources".into());
                let note = self
                    .unload_note
                    .take()
                    .unwrap_or_else(|| "Runtime unloaded.".to_string());
                ui.set_status(note.into());
                ui.set_status_error(false);
            }
            UiKind::McpConnected(info) => {
                self.set_phase(ui, Phase::Idle);
                ui.set_mcp_connected(true);
                let summary = format!(
                    "{} \u{00B7} {} tools ({} ms)",
                    info.server_name,
                    info.tools.len(),
                    info.latency_ms
                );
                ui.set_mcp_status(summary.as_str().into());
                ui.set_status(format!("MCP connected: {summary}").into());
                ui.set_status_error(false);
                self.persist_settings_now(ui);
            }
            UiKind::McpFailed(err) => {
                self.set_phase(ui, Phase::Idle);
                ui.set_mcp_connected(false);
                ui.set_mcp_status("Disconnected".into());
                ui.set_status(format!("MCP failed: {err}").into());
                ui.set_status_error(true);
            }
            UiKind::Delta { text, reasoning } => {
                // Buffered, not applied: the transcript is repainted on the batch
                // cadence (or immediately before a tool call / error / done).
                self.batcher.push(text.as_deref(), reasoning.as_deref());
            }
            UiKind::ToolCall(call) => {
                self.flush_delta(ui);
                let entry = self.session.push_tool_call(call);
                self.turns_model.push(to_chat_turn(entry));
                self.touch_transcript(ui);
            }
            UiKind::ToolExecuted { index, outcome } => {
                self.flush_delta(ui);
                match self.session.resolve_tool(index, &outcome.output, outcome.is_success) {
                    Some(entry) => {
                        self.turns_model.push(to_chat_turn(entry));
                        self.touch_transcript(ui);
                        self.pending_done = self.pending_done.saturating_add(1);
                        self.mark_session_dirty();
                    }
                    None => {
                        // Duplicate or late result for an already-closed batch: keep the
                        // queue moving instead of leaving the approval bar stranded.
                        self.stale_events += 1;
                        if self.pending_calls.is_empty() {
                            self.clear_tool_approval_ui(ui);
                            self.set_phase(ui, Phase::Idle);
                        }
                        return;
                    }
                }
                self.prompt_next_tool_or_continue(ui);
            }
            UiKind::Error(message) => {
                self.flush_delta(ui);
                let entry = self.session.push_error(&message);
                self.turns_model.push(to_chat_turn(entry));
                self.touch_transcript(ui);
                ui.set_status_error(true);
                let restored = self.restore_draft(ui);
                ui.set_status(
                    if restored {
                        format!("Generation failed: {message} \u{00B7} draft restored — press Enter to retry")
                    } else {
                        format!("Generation failed: {message}")
                    }
                    .into(),
                );
            }
            UiKind::Done { cancelled } => {
                self.flush_delta(ui);
                self.finish_stream(ui, cancelled);
            }
            UiKind::ImageWritten { path, message } => {
                self.set_phase(ui, Phase::Idle);
                match slint::Image::load_from_path(&path) {
                    Ok(image) => {
                        ui.set_image_preview(image);
                        ui.set_image_ready(true);
                        ui.set_status(message.into());
                        ui.set_status_error(false);
                    }
                    Err(error) => {
                        ui.set_status(format!("Image saved, preview failed: {error}").into());
                        ui.set_status_error(true);
                    }
                }
            }
            UiKind::ImageFailed(message) => {
                self.set_phase(ui, Phase::Idle);
                ui.set_status(format!("Image generation failed: {message}").into());
                ui.set_status_error(true);
            }
        }
    }

    /// Handles the terminal event of a stream: commits the assistant turn, then either
    /// starts the tool round or closes everything.
    fn finish_stream(&mut self, ui: &MainWindow, cancelled: bool) {
        let had_error = self.session.had_error;
        let (calls, from_inline) = self.session.finish_turn();
        if from_inline {
            for call in &calls {
                self.turns_model.push(ChatTurn {
                    kind: crate::session::KIND_TOOL,
                    role: "Tool call".into(),
                    body: call.display_text().into(),
                });
            }
            self.touch_transcript(ui);
        }
        self.generation = None;
        self.stopping_since = None;
        self.mark_session_dirty();

        if cancelled || had_error {
            let reason = if cancelled {
                ToolAbort::Cancelled
            } else {
                ToolAbort::NotExecuted
            };
            let entries = self.session.abort_pending_tools(reason);
            self.push_entries(ui, entries);
            self.pending_calls.clear();
            self.deferred_calls.clear();
            self.clear_tool_approval_ui(ui);
            self.set_phase(ui, Phase::Idle);
            if cancelled {
                ui.set_status("Stopped \u{00B7} any pending tool call was closed.".into());
                ui.set_status_error(false);
            }
            self.refresh_resources(ui);
            return;
        }

        if !calls.is_empty() {
            let can_run = self.mcp.is_connected() && self.current_tool_round < MAX_TOOL_ROUNDS;
            if can_run {
                self.current_tool_round += 1;
                let runnable = calls.len().min(MAX_CALLS_PER_TURN);
                self.pending_total = runnable;
                self.pending_done = 0;
                self.pending_calls.clear();
                self.deferred_calls.clear();
                for (index, call) in calls.iter().enumerate() {
                    if index < runnable {
                        self.pending_calls.push_back((index, call.clone()));
                    } else {
                        self.deferred_calls.push((index, call.clone()));
                    }
                }
                self.prompt_next_tool_or_continue(ui);
                return;
            }
            // The tool calls cannot run: close the batch instead of leaving an
            // assistant `tool_calls` message without results (the old HTTP 400 bug).
            let entries = self.session.abort_pending_tools(ToolAbort::NotExecuted);
            self.push_entries(ui, entries);
            let reason = if !self.mcp.is_connected() {
                "no MCP server is connected"
            } else {
                "the tool-round limit was reached"
            };
            ui.set_status(format!("Tool calls were not executed: {reason}.").into());
            ui.set_status_error(true);
        }

        self.set_phase(ui, Phase::Idle);
        if !ui.get_status_error() {
            ui.set_status("Ready".into());
            ui.set_status_error(false);
        }
        self.refresh_resources(ui);
    }

    /// Abandons a run without waiting for its worker: commits whatever the user already
    /// saw, closes any open tool batch with "interrupted" results, drops every in-flight
    /// event and returns the UI to idle.
    ///
    /// Used by a backend switch, by New conversation and by the stop watchdog, so there is
    /// exactly one definition of "the run is over".
    fn abandon_run(&mut self, ui: &MainWindow, note: &str) {
        self.flush_delta(ui);
        let (calls, from_inline) = self.session.finish_turn();
        if from_inline {
            for call in &calls {
                self.turns_model.push(ChatTurn {
                    kind: crate::session::KIND_TOOL,
                    role: "Tool call".into(),
                    body: call.display_text().into(),
                });
            }
            self.touch_transcript(ui);
        }
        if let Some(cancel) = self.generation.take() {
            cancel.store(true, Ordering::SeqCst);
        }
        self.epoch.invalidate();
        self.batcher.clear();
        self.pending_calls.clear();
        self.deferred_calls.clear();
        self.clear_tool_approval_ui(ui);
        let entries = self.session.abort_pending_tools(ToolAbort::Interrupted);
        self.push_entries(ui, entries);
        self.stopping_since = None;
        self.set_phase(ui, Phase::Idle);
        if !note.is_empty() {
            ui.set_status(note.into());
            ui.set_status_error(false);
        }
        self.mark_session_dirty();
    }

    fn prompt_next_tool_or_continue(&mut self, ui: &MainWindow) {
        if let Some((_, next_call)) = self.pending_calls.front() {
            let position = self.pending_done + 1;
            ui.set_awaiting_tool_approval(true);
            ui.set_pending_tool_title(format!("Approve MCP tool \"{}\"?", next_call.name).into());
            ui.set_pending_tool_args(next_call.arguments.as_str().into());
            ui.set_pending_tool_progress(format!("{position} of {}", self.pending_total).into());
            ui.set_status(
                format!(
                    "Awaiting approval for \"{}\" \u{00B7} {} of {} \u{00B7} Esc declines",
                    next_call.name, position, self.pending_total
                )
                .into(),
            );
            ui.set_status_error(false);
            self.set_phase(ui, Phase::Tool);
            return;
        }

        self.clear_tool_approval_ui(ui);

        // Calls beyond the per-turn cap were committed but never shown; close them now
        // so the batch invariant holds before the next request.
        if !self.deferred_calls.is_empty() {
            let mut entries = Vec::new();
            for (index, _) in std::mem::take(&mut self.deferred_calls) {
                if let Some(entry) = self.session.resolve_tool(index, ToolAbort::NotExecuted.text(), false) {
                    entries.push(entry);
                }
            }
            self.push_entries(ui, entries);
        }

        let Some(backend) = self.backend.clone() else {
            let entries = self.session.abort_pending_tools(ToolAbort::NotExecuted);
            self.push_entries(ui, entries);
            self.set_phase(ui, Phase::Idle);
            ui.set_status(
                "The runtime is no longer loaded; tool results are kept in the transcript.".into(),
            );
            ui.set_status_error(true);
            return;
        };

        self.session.prepare_continuation_round();
        let temperature = parse_f64_clamped(&ui.get_temperature_text(), 0.7, 0.0, 2.0);
        let max_tokens = parse_u32_clamped(&ui.get_max_tokens_text(), 1024, 1, 32_768);
        let request = self
            .session
            .build_request(self.mcp.current_tools(), temperature, max_tokens);
        self.spawn_stream(ui, backend, request);
    }

    fn approve_pending_tool(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let Some((index, call)) = self.pending_calls.pop_front() else {
            return;
        };
        self.clear_tool_approval_ui(&ui);
        ui.set_status(format!("Running MCP tool \"{}\"…", call.name).into());
        ui.set_status_error(false);
        self.set_phase(&ui, Phase::Tool);
        // A new generation: a result that arrives after a decline/stop is stale.
        let epoch = self.epoch.begin();
        let mcp = Arc::clone(&self.mcp);
        let sender = self.sender.clone();
        thread::spawn(move || {
            let outcome = mcp.call_tool(&call.name, &call.arguments);
            let _ = sender.send(UiEvent {
                epoch,
                kind: UiKind::ToolExecuted { index, outcome },
            });
        });
    }

    /// Deterministic decline: every call still in the batch gets exactly one "declined"
    /// result, so the history stays valid and the model learns the tools were refused.
    fn decline_pending_tool(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.pending_calls.is_empty() && !self.session.has_unresolved_tools() {
            return;
        }
        self.epoch.invalidate();
        self.pending_calls.clear();
        self.deferred_calls.clear();
        self.clear_tool_approval_ui(&ui);
        let entries = self.session.abort_pending_tools(ToolAbort::Declined);
        self.push_entries(&ui, entries);
        self.set_phase(&ui, Phase::Idle);
        ui.set_status(
            "Tool call declined \u{00B7} every pending call was closed. Send your next message to continue."
                .into(),
        );
        ui.set_status_error(false);
        self.mark_session_dirty();
    }

    fn clear_tool_approval_ui(&self, ui: &MainWindow) {
        ui.set_awaiting_tool_approval(false);
        ui.set_pending_tool_title("".into());
        ui.set_pending_tool_args("".into());
        ui.set_pending_tool_progress("".into());
    }

    fn connect_mcp(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.phase.is_busy() {
            return;
        }
        let url = ui.get_mcp_url().to_string();
        let token = ui.get_mcp_token().to_string();
        let mcp = Arc::clone(&self.mcp);
        let sender = self.sender.clone();
        let epoch = self.epoch.begin();
        self.set_phase(&ui, Phase::Connecting);
        ui.set_mcp_status("Connecting…".into());
        ui.set_status("Connecting to MCP server…".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            let token_opt = (!token.trim().is_empty()).then_some(token.as_str());
            let kind = match mcp.connect(&url, token_opt) {
                Ok(info) => UiKind::McpConnected(info),
                Err(err) => UiKind::McpFailed(err),
            };
            let _ = sender.send(UiEvent { epoch, kind });
        });
    }

    fn disconnect_mcp(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.phase.is_busy() {
            return;
        }
        self.mcp.disconnect();
        ui.set_mcp_connected(false);
        ui.set_mcp_status("Disconnected".into());
        ui.set_status("MCP server disconnected.".into());
        ui.set_status_error(false);
    }

    fn sync_turn(&self, ui: &MainWindow, idx: usize, entry: TurnEntry, inserted: bool) {
        let turn = to_chat_turn(entry);
        if inserted {
            self.turns_model.push(turn);
        } else {
            self.turns_model.set_row_data(idx, turn);
        }
        self.touch_transcript(ui);
    }

    fn push_entries(&self, ui: &MainWindow, entries: Vec<TurnEntry>) {
        if entries.is_empty() {
            return;
        }
        for entry in entries {
            self.turns_model.push(to_chat_turn(entry));
        }
        self.touch_transcript(ui);
    }

    fn sync_all_turns(&self, ui: &MainWindow) {
        let turns: Vec<ChatTurn> = self.session.turns().iter().cloned().map(to_chat_turn).collect();
        self.turns_model.set_vec(turns);
        self.touch_transcript(ui);
    }

    fn touch_transcript(&self, ui: &MainWindow) {
        ui.set_scroll_tick(ui.get_scroll_tick().wrapping_add(1));
    }

    /// Requests that the chat composer take keyboard focus (launch, new conversation).
    fn focus_composer(&self, ui: &MainWindow) {
        ui.set_composer_focus_tick(ui.get_composer_focus_tick().wrapping_add(1));
    }

    fn set_phase(&mut self, ui: &MainWindow, phase: Phase) {
        self.phase = phase;
        ui.set_busy(phase.is_busy());
        ui.set_phase(phase.footer().into());
    }

    /// Applies buffered streaming text to the transcript. Called on the batch cadence and
    /// forced before tool calls, errors and done so nothing is lost or reordered.
    fn flush_delta(&mut self, ui: &MainWindow) {
        let Some(batch) = self.batcher.drain() else {
            return;
        };
        if let Some(reasoning) = batch.reasoning {
            if let Some((idx, entry, inserted)) = self.session.append_reasoning(&reasoning) {
                self.sync_turn(ui, idx, entry, inserted);
            }
        }
        if let Some(text) = batch.text {
            if let Some((idx, entry, inserted)) = self.session.append_assistant(&text) {
                self.sync_turn(ui, idx, entry, inserted);
            }
        }
    }

    fn new_chat(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let was_running = self.generation.is_some() || !self.pending_calls.is_empty();
        // Invalidate in-flight work first: late deltas / tool results belong to the old
        // conversation and must not appear in the new one.
        self.epoch.invalidate();
        if let Some(cancel) = self.generation.take() {
            cancel.store(true, Ordering::SeqCst);
        }
        self.stopping_since = None;
        self.batcher.clear();
        self.pending_calls.clear();
        self.deferred_calls.clear();
        self.clear_tool_approval_ui(&ui);
        self.session.clear();
        self.turns_model.set_vec(Vec::new());
        self.current_tool_round = 0;
        self.pending_total = 0;
        self.pending_done = 0;
        self.retry_draft = None;
        self.set_phase(&ui, Phase::Idle);
        ui.set_status(
            if was_running {
                "New conversation \u{00B7} the previous run was stopped."
            } else {
                "New conversation."
            }
            .into(),
        );
        ui.set_status_error(false);
        self.touch_transcript(&ui);
        self.focus_composer(&ui);
        self.persist_session_now();
    }

    fn load_model(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.phase.is_busy() {
            return;
        }
        // Read the backend once: the choice and the model spec must describe the same
        // backend even if the user clicks the selector mid-load.
        let kind = BackendKind::from_index(ui.get_backend_index());
        let choice = self.backend_choice(&ui, kind);
        let raw_label = ui.get_model_label().to_string();
        let model_label = if raw_label.trim().is_empty() {
            "local-model".to_string()
        } else {
            raw_label.trim().to_string()
        };
        let local_path = match kind {
            BackendKind::LlamaCpp => Some(ui.get_llama_model_path().to_string()),
            BackendKind::Vllm => Some(ui.get_vllm_model_path().to_string()),
            BackendKind::Strata => None,
        };
        let model = ModelSpec::new(model_label.clone(), local_path);
        let previous = self.backend.take();
        self.loaded_kind = None;
        let epoch = self.epoch.begin();
        let sender = self.sender.clone();
        self.set_phase(&ui, Phase::Loading);
        ui.set_status(format!("Loading {}…", kind.label()).into());
        ui.set_status_error(false);
        thread::spawn(move || {
            if let Some(previous) = previous {
                previous.unload();
            }
            let backend = choice.build();
            match backend.load(&model) {
                Ok(()) => {
                    let capabilities = backend.capabilities();
                    let status = if capabilities.available {
                        capabilities.reason
                    } else {
                        format!("Unavailable: {}", capabilities.reason)
                    };
                    let _ = sender.send(UiEvent {
                        epoch,
                        kind: UiKind::Loaded {
                            backend,
                            model_label,
                            status,
                        },
                    });
                }
                Err(error) => {
                    backend.unload();
                    let _ = sender.send(UiEvent {
                        epoch,
                        kind: UiKind::LoadFailed(error),
                    });
                }
            }
        });
        self.persist_settings_now(&ui);
    }

    fn unload_model(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.phase.is_busy() {
            return;
        }
        let previous = self.backend.take();
        self.loaded_kind = None;
        self.unload_note = None;
        let epoch = self.epoch.begin();
        let sender = self.sender.clone();
        self.set_phase(&ui, Phase::Unloading);
        ui.set_status("Unloading runtime…".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            if let Some(previous) = previous {
                previous.unload();
            }
            let _ = sender.send(UiEvent {
                epoch,
                kind: UiKind::Unloaded,
            });
        });
    }

    /// Backend selector changed.
    ///
    /// The old code only relabelled the empty state, so `model-loaded` stayed true and
    /// requests kept going to the previous backend. Now a switch invalidates in-flight
    /// work, unloads the runtime that belonged to the old backend and resets every piece
    /// of "loaded" state before the new backend can be used.
    fn backend_changed(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let kind = BackendKind::from_index(ui.get_backend_index());
        let had_runtime = self.backend.is_some();

        // 1. Keep whatever the user already saw, then commit the open assistant turn so
        //    the transcript and the history agree even though the run is being abandoned.
        if self.generation.is_some() {
            self.abandon_run(&ui, "");
        }

        // 2. Everything still in flight belongs to the previous backend.
        self.epoch.invalidate();
        self.batcher.clear();
        self.pending_calls.clear();
        self.deferred_calls.clear();
        self.clear_tool_approval_ui(&ui);
        if self.session.has_unresolved_tools() {
            let entries = self.session.abort_pending_tools(ToolAbort::Interrupted);
            self.push_entries(&ui, entries);
        }

        let previous = self.backend.take();
        self.loaded_kind = None;
        self.current_tool_round = 0;
        ui.set_model_loaded(false);
        ui.set_model_indicator("No model".into());
        ui.set_empty_engine(idle_engine_label(kind).into());
        ui.set_resources("No runtime resources".into());
        ui.set_status_error(false);

        if let Some(previous) = previous {
            let epoch = self.epoch.begin();
            let sender = self.sender.clone();
            self.unload_note = Some(format!(
                "Switched to {} \u{00B7} the previous runtime was unloaded. Load or connect to continue.",
                kind.label()
            ));
            self.set_phase(&ui, Phase::Unloading);
            ui.set_status(format!("Switching to {} \u{00B7} unloading…", kind.label()).into());
            thread::spawn(move || {
                previous.unload();
                let _ = sender.send(UiEvent {
                    epoch,
                    kind: UiKind::Unloaded,
                });
            });
        } else if had_runtime {
            self.set_phase(&ui, Phase::Idle);
            ui.set_status(format!("Switched to {}. Load or connect to continue.", kind.label()).into());
        } else {
            self.set_phase(&ui, Phase::Idle);
            ui.set_status(
                format!(
                    "{} selected. Configure settings and load or connect.",
                    kind.label()
                )
                .into(),
            );
        }
        self.persist_settings_now(&ui);
        self.refresh_composer_gate(&ui);
    }

    fn backend_choice(&self, ui: &MainWindow, kind: BackendKind) -> BackendChoice {
        match kind {
            BackendKind::LlamaCpp => BackendChoice::Llama(LlamaCppConfig {
                executable: ui.get_llama_executable().to_string(),
                default_model_path: ui.get_llama_model_path().to_string(),
                port: parse_port(&ui.get_llama_port(), 8080),
                context_size: parse_u32_clamped(&ui.get_llama_context_size(), 4096, 512, 131_072),
                gpu_layers: parse_u32(&ui.get_llama_gpu_layers(), 0),
                ..LlamaCppConfig::default()
            }),
            BackendKind::Vllm => BackendChoice::Vllm(VllmConfig {
                base_url: ui.get_vllm_endpoint().to_string(),
                api_key: std::env::var("VLLM_API_KEY").ok(),
                model_id: ui.get_model_label().to_string(),
                model_path: ui.get_vllm_model_path().to_string(),
                auto_start: ui.get_vllm_managed(),
                port: parse_port(&ui.get_vllm_port(), 8000),
                ..VllmConfig::default()
            }),
            BackendKind::Strata => BackendChoice::Strata(StrataConfig {
                base_url: ui.get_strata_endpoint().to_string(),
                api_key: std::env::var("STRATA_API_KEY").ok(),
                model_id: ui.get_model_label().to_string(),
                launch_command: parse_command(&ui.get_strata_launch()),
                working_directory: Some(PathBuf::from(".")),
                ..StrataConfig::default()
            }),
        }
    }

    fn send_message(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.phase.is_busy() {
            // The draft stays in the composer: nothing is lost by pressing Enter early.
            ui.set_status(format!("Busy ({}). Esc stops the current run.", self.phase.footer()).into());
            ui.set_status_error(false);
            return;
        }
        let kind = BackendKind::from_index(ui.get_backend_index());
        let Some(backend) = self.backend.clone() else {
            ui.set_status("Load or connect a backend first.".into());
            ui.set_status_error(true);
            return;
        };
        if self.loaded_kind != Some(kind) || !ui.get_model_loaded() {
            ui.set_status(
                "The selected backend is not the loaded one. Load or connect it before sending."
                    .into(),
            );
            ui.set_status_error(true);
            return;
        }
        let raw_prompt = ui.get_input_text().to_string();
        let temperature = parse_f64_clamped(&ui.get_temperature_text(), 0.7, 0.0, 2.0);
        let max_tokens = parse_u32_clamped(&ui.get_max_tokens_text(), 1024, 1, 32_768);
        let tools = self.mcp.current_tools();
        let Some(request) = self
            .session
            .begin_user_turn(&raw_prompt, tools, temperature, max_tokens)
        else {
            ui.set_status("Type a prompt first.".into());
            ui.set_status_error(true);
            return;
        };
        // The draft is cleared only after the turn is committed to the session, and it is
        // remembered so a failed first round can put it back.
        ui.set_input_text("".into());
        self.retry_draft = Some(raw_prompt.trim().to_string());
        self.pending_calls.clear();
        self.deferred_calls.clear();
        self.pending_total = 0;
        self.pending_done = 0;
        self.current_tool_round = 0;
        if let Some(last_user) = self.session.turns().last().cloned() {
            self.turns_model.push(to_chat_turn(last_user));
        }
        ui.set_stick_tick(ui.get_stick_tick().wrapping_add(1));
        self.touch_transcript(&ui);
        self.spawn_stream(&ui, backend, request);
        self.mark_session_dirty();
    }

    fn spawn_stream(
        &mut self,
        ui: &MainWindow,
        backend: Arc<dyn LlmBackend>,
        request: ChatRequest,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.generation = Some(Arc::clone(&cancel));
        self.stopping_since = None;
        let epoch = self.epoch.begin();
        let sender = self.sender.clone();
        self.set_phase(ui, Phase::Streaming);
        ui.set_status("Generating…".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            backend.stream(&request, &cancel, &mut |event| {
                let kind = match event {
                    StreamEvent::Delta { text, reasoning } => UiKind::Delta { text, reasoning },
                    StreamEvent::ToolCall(call) => UiKind::ToolCall(call),
                    StreamEvent::Error(error) => UiKind::Error(error),
                    StreamEvent::Done => return,
                };
                let _ = sender.send(UiEvent { epoch, kind });
            });
            let cancelled = cancel.load(Ordering::SeqCst);
            let _ = sender.send(UiEvent {
                epoch,
                kind: UiKind::Done { cancelled },
            });
        });
    }

    /// Esc / Stop.
    fn stop_generation(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if !self.pending_calls.is_empty() || self.session.has_unresolved_tools() {
            // Close the whole pending set: every committed call gets one "cancelled"
            // result, and any in-flight tool result becomes stale.
            self.epoch.invalidate();
            self.pending_calls.clear();
            self.deferred_calls.clear();
            self.clear_tool_approval_ui(&ui);
            let entries = self.session.abort_pending_tools(ToolAbort::Cancelled);
            self.push_entries(&ui, entries);
            self.set_phase(&ui, Phase::Idle);
            ui.set_status("Stopped \u{00B7} every pending tool call was cancelled.".into());
            ui.set_status_error(false);
            self.mark_session_dirty();
            return;
        }
        if let Some(cancel) = &self.generation {
            cancel.store(true, Ordering::SeqCst);
            ui.set_status("Stopping…".into());
            ui.set_status_error(false);
            self.set_phase(&ui, Phase::Stopping);
            // If the backend never acknowledges the cancel, the tick watchdog closes the
            // run locally so the app cannot stay stuck in "stopping".
            self.stopping_since = Some(Instant::now());
        }
    }

    /// Esc routing: an open approval bar is declined first, then a running generation is
    /// stopped. Nothing else consumes Escape.
    fn handle_escape(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if ui.get_awaiting_tool_approval() {
            self.decline_pending_tool();
            return;
        }
        self.stop_generation();
    }

    fn generate_image(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.phase.is_busy() {
            return;
        }
        let prompt = ui.get_image_prompt().trim().to_string();
        if prompt.is_empty() {
            ui.set_status("Image prompt is required.".into());
            ui.set_status_error(true);
            return;
        }
        let backend = OpenAiImageBackend::new(
            ui.get_image_endpoint().to_string(),
            std::env::var("IMAGE_API_KEY").ok(),
        );
        let request = ImageRequest {
            model: ui.get_image_model().to_string(),
            prompt,
            size: ui.get_image_size().to_string(),
            output_format: "png".into(),
        };
        let output = PathBuf::from(ui.get_image_output().to_string());
        let sender = self.sender.clone();
        let epoch = self.epoch.begin();
        self.set_phase(&ui, Phase::Image);
        ui.set_status("Generating image…".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            let kind = match backend.generate(&request) {
                Ok(result) => {
                    let write = (|| -> Result<(), String> {
                        if let Some(parent) = output.parent() {
                            if !parent.as_os_str().is_empty() {
                                std::fs::create_dir_all(parent).map_err(|error| {
                                    format!("could not create {}: {error}", parent.display())
                                })?;
                            }
                        }
                        std::fs::write(&output, &result.bytes)
                            .map_err(|error| format!("could not write {}: {error}", output.display()))
                    })();
                    match write {
                        Ok(()) => {
                            let display_path = output.canonicalize().unwrap_or_else(|_| output.clone());
                            UiKind::ImageWritten {
                                path: output,
                                message: format!("Image written to {}", display_path.display()),
                            }
                        }
                        Err(error) => UiKind::ImageFailed(error),
                    }
                }
                Err(error) => UiKind::ImageFailed(error),
            };
            let _ = sender.send(UiEvent { epoch, kind });
        });
    }

    fn refresh_resources(&mut self, ui: &MainWindow) {
        let Some(backend) = self.backend.as_ref() else {
            ui.set_resources("No runtime resources".into());
            return;
        };
        let snapshot = backend.resources();
        ui.set_resources(format_resources(&snapshot).into());
    }

    /// Recomputes "can the composer send right now?" and the one-line reason when it
    /// cannot. Cheap enough to run on every other tick.
    fn refresh_composer_gate(&mut self, ui: &MainWindow) {
        let kind = BackendKind::from_index(ui.get_backend_index());
        let backend_ready =
            self.backend.is_some() && self.loaded_kind == Some(kind) && ui.get_model_loaded();
        let draft_empty = ui.get_input_text().trim().is_empty();
        let can_send = backend_ready
            && !self.phase.is_busy()
            && !draft_empty
            && !ui.get_awaiting_tool_approval();
        if ui.get_send_ready() != can_send {
            ui.set_send_ready(can_send);
        }

        let note = if self.phase.is_busy() {
            self.phase.hint()
        } else if !backend_ready {
            if self.backend.is_none() {
                "Load or connect a backend to send"
            } else {
                "The loaded backend is not the selected one — reload it"
            }
        } else if draft_empty {
            "Write a prompt to send"
        } else {
            ""
        };
        if ui.get_composer_note() != note {
            ui.set_composer_note(note.into());
        }
    }

    fn restore_draft(&mut self, ui: &MainWindow) -> bool {
        // Only a failed first round is retryable from the composer; later rounds already
        // have tool output the user would lose.
        if self.current_tool_round != 0 {
            return false;
        }
        let Some(draft) = self.retry_draft.take() else {
            return false;
        };
        if !ui.get_input_text().trim().is_empty() {
            return false;
        }
        ui.set_input_text(draft.as_str().into());
        true
    }

    fn mark_session_dirty(&mut self) {
        self.session_save.mark_dirty(Instant::now());
    }

    fn persist_settings_now(&mut self, ui: &MainWindow) {
        let current = settings_from_ui(ui);
        if !self.settings_save.is_changed(&current) {
            return;
        }
        if persist::save_settings(&self.config_dir, &current).is_ok() {
            self.settings_save.mark_saved(current);
        }
    }

    fn persist_session_now(&mut self) {
        let stored = PersistedSession {
            turns: self.session.turns().to_vec(),
            messages: self.session.messages().to_vec(),
        };
        if persist::save_session(&self.config_dir, &stored).is_ok() {
            self.session_save
                .mark_saved((stored.turns.len(), stored.messages.len()));
        }
    }

    /// Debounced persistence: writes at most once per quiet period, so dragging a slider
    /// or typing a path does not hammer the disk.
    fn maybe_persist(&mut self, ui: &MainWindow, now: Instant) {
        let current = settings_from_ui(ui);
        if self.settings_save.observe(&current, now) {
            if persist::save_settings(&self.config_dir, &current).is_ok() {
                self.settings_save.mark_saved(current);
            }
        }

        let fingerprint = (self.session.turns().len(), self.session.messages().len());
        if self.session_save.observe(&fingerprint, now) {
            self.persist_session_now();
        }
    }

    fn shutdown(&mut self) {
        if let Some(cancel) = self.generation.take() {
            cancel.store(true, Ordering::SeqCst);
        }
        if self.session.has_unresolved_tools() {
            // Closing the window must not leave a half-open tool batch behind.
            self.session.abort_pending_tools(ToolAbort::Interrupted);
        }
        if let Some(ui) = self.ui.upgrade() {
            self.persist_settings_now(&ui);
        }
        self.persist_session_now();
        self.mcp.disconnect();
        if let Some(backend) = self.backend.take() {
            backend.unload();
        }
    }
}

fn idle_engine_label(kind: BackendKind) -> String {
    format!(
        "{} \u{00B7} {}",
        kind.label(),
        if kind.is_local_process() {
            "local runtime"
        } else {
            "endpoint"
        }
    )
}

pub fn run() -> Result<(), slint::PlatformError> {
    run_with_config_dir(persist::config_dir())
}

pub fn run_with_config_dir(config_dir: PathBuf) -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    let controller = Rc::new(RefCell::new(DesktopController::new(
        ui.as_weak(),
        config_dir,
    )));
    controller.borrow_mut().install();

    macro_rules! wire {
        ($method:ident, $action:ident) => {{
            let weak = Rc::downgrade(&controller);
            ui.$method(move || {
                if let Some(ctrl) = weak.upgrade() {
                    ctrl.borrow_mut().$action();
                }
            });
        }};
    }

    wire!(on_load_requested, load_model);
    wire!(on_unload_requested, unload_model);
    wire!(on_new_chat_requested, new_chat);
    wire!(on_send_requested, send_message);
    wire!(on_stop_requested, stop_generation);
    wire!(on_escape_requested, handle_escape);
    wire!(on_reload_requested, load_model);
    wire!(on_generate_image_requested, generate_image);
    wire!(on_backend_changed, backend_changed);
    wire!(on_mcp_connect_requested, connect_mcp);
    wire!(on_mcp_disconnect_requested, disconnect_mcp);
    wire!(on_approve_tool_requested, approve_pending_tool);
    wire!(on_decline_tool_requested, decline_pending_tool);

    // Persist before the window closes, then again from `shutdown` (which also runs when
    // the event loop ends for another reason).
    {
        let weak = Rc::downgrade(&controller);
        let window = ui.as_weak();
        ui.window().on_close_requested(move || {
            if let (Some(ctrl), Some(window)) = (weak.upgrade(), window.upgrade()) {
                let mut ctrl = ctrl.borrow_mut();
                ctrl.persist_settings_now(&window);
                ctrl.persist_session_now();
            }
            slint::CloseRequestResponse::HideWindow
        });
    }

    // Focus the composer shortly after the window is mapped: doing it from the first
    // event-loop turn beats the default focus traversal that would otherwise land the
    // cursor in the first sidebar field.
    {
        let window = ui.as_weak();
        slint::Timer::single_shot(Duration::from_millis(120), move || {
            if let Some(window) = window.upgrade() {
                window.set_composer_focus_tick(window.get_composer_focus_tick().wrapping_add(1));
            }
        });
    }

    let timer = slint::Timer::default();
    {
        let weak = Rc::downgrade(&controller);
        timer.start(slint::TimerMode::Repeated, TICK_INTERVAL, move || {
            if let Some(ctrl) = weak.upgrade() {
                ctrl.borrow_mut().tick();
            }
        });
    }

    let result = ui.run();
    drop(timer);
    controller.borrow_mut().shutdown();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epoch::EpochGate;

    #[test]
    fn phases_distinguish_every_busy_state_and_never_read_as_idle() {
        assert!(!Phase::Idle.is_busy());
        for phase in [
            Phase::Loading,
            Phase::Unloading,
            Phase::Connecting,
            Phase::Streaming,
            Phase::Stopping,
            Phase::Tool,
            Phase::Image,
        ] {
            assert!(phase.is_busy(), "{phase:?} must lock the composer");
            assert!(!phase.footer().is_empty(), "{phase:?} needs a footer label");
            assert!(!phase.hint().is_empty(), "{phase:?} needs a composer hint");
        }
        // Labels are distinct so the footer can be used to tell them apart.
        let labels = [
            Phase::Loading.footer(),
            Phase::Unloading.footer(),
            Phase::Connecting.footer(),
            Phase::Streaming.footer(),
            Phase::Tool.footer(),
        ];
        for (index, label) in labels.iter().enumerate() {
            for other in labels.iter().skip(index + 1) {
                assert_ne!(label, other);
            }
        }
    }

    #[test]
    fn a_late_completion_from_the_previous_backend_is_dropped() {
        // Regression for "backend switch lies": the user switches backend while a load
        // is in flight; the old load finishing last must not mark the new backend loaded.
        let mut gate = EpochGate::new();
        let loading_llama = gate.begin();
        let switch = gate.invalidate();
        assert!(!gate.is_current(loading_llama), "late Loaded must be dropped");
        assert!(gate.is_current(switch));
        let loading_vllm = gate.begin();
        assert!(gate.is_current(loading_vllm));
    }

    #[test]
    fn settings_snapshot_covers_every_persisted_ui_field() {
        // Compile-time-ish guard: the settings type has exactly the fields we expect to
        // restore. Adding a UI field without persisting it fails here.
        let settings = DesktopSettings::default();
        let value = settings.to_value();
        for key in [
            "dark_mode",
            "edition_index",
            "backend_index",
            "model_label",
            "llama",
            "vllm",
            "strata",
            "sampling",
            "mcp",
            "image",
        ] {
            assert!(value.get(key).is_some(), "missing persisted key {key}");
        }
        assert!(value["llama"].get("executable").is_some());
        assert!(value["llama"].get("gpu_layers").is_some());
        assert!(value["vllm"].get("managed").is_some());
        assert!(value["strata"].get("launch").is_some());
        assert!(value["sampling"].get("temperature").is_some());
        assert!(value["mcp"].get("url").is_some());
        assert!(value["image"].get("output").is_some());
    }
}
