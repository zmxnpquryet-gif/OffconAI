//! Slint UI orchestration controller.
//! All blocking HTTP/process/MCP I/O executes on worker threads; the Slint event loop
//! drains typed `UiEvent`s and coordinates multi-round MCP tool execution with user approval.
//!
//! The controller also owns the built-in model catalog: selection, in-app HF download
//! with progress, and load-then-go chaining so a fresh install is truly one click.

use crate::backends::{LlamaCpp, LlamaCppConfig, Strata, StrataConfig, Vllm, VllmConfig};
use crate::catalog::{self, CatalogModel};
use crate::contracts::{
    BackendKind, ChatRequest, ImageGenerationBackend, ImageRequest, LlmBackend, ModelSpec,
    StreamEvent, ToolCallRecord,
};
use crate::epoch::EpochGate;
use crate::image_backend::OpenAiImageBackend;
use crate::mcp::{McpCallOutcome, McpClient, McpConnectionInfo};
use crate::persist::{load_settings, save_settings, Settings};
use crate::process::{
    app_data_dir, format_bytes, format_resources, parse_command, parse_f64_clamped, parse_port,
    parse_u32, parse_u32_clamped, sanitize_path,
};
use crate::session::{ConversationSession, TurnEntry};
use crate::{ChatTurn, MainWindow};
use slint::{ComponentHandle, ModelRc, VecModel};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const MAX_TOOL_ROUNDS: u32 = 3;
const MAX_CALLS_PER_TURN: usize = 4;
const STOP_WATCHDOG_MS: u64 = 2500;

#[derive(Default)]
struct DeltaBatcher {
    reasoning: String,
    text: String,
}

impl DeltaBatcher {
    fn push(&mut self, text: Option<String>, reasoning: Option<String>) {
        if let Some(r) = reasoning {
            self.reasoning.push_str(&r);
        }
        if let Some(t) = text {
            self.text.push_str(&t);
        }
    }

    fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.text.is_empty()
    }

    fn take(&mut self) -> (String, String) {
        (
            std::mem::take(&mut self.reasoning),
            std::mem::take(&mut self.text),
        )
    }

    fn clear(&mut self) {
        self.reasoning.clear();
        self.text.clear();
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

/// One row of the model picker for the active engine.
#[derive(Clone)]
enum CatalogEntry {
    Catalog(&'static CatalogModel),
    Local { label: String, path: String },
    Gateway,
}

impl CatalogEntry {
    fn label(&self, installed: bool) -> String {
        match self {
            Self::Catalog(model) => model.list_label(installed),
            Self::Local { label, .. } => format!("{label}  \u{2713} local"),
            Self::Gateway => "Gateway-managed model".to_string(),
        }
    }
}

enum UiEvent {
    Loaded {
        epoch: u64,
        backend: Arc<dyn LlmBackend>,
        model_label: String,
        status: String,
    },
    LoadFailed {
        epoch: u64,
        message: String,
    },
    Unloaded {
        epoch: u64,
    },
    McpConnected {
        epoch: u64,
        info: McpConnectionInfo,
    },
    McpFailed {
        epoch: u64,
        error: String,
    },
    Delta {
        epoch: u64,
        text: Option<String>,
        reasoning: Option<String>,
    },
    ToolCall {
        epoch: u64,
        call: ToolCallRecord,
    },
    ToolExecuted {
        epoch: u64,
        call: ToolCallRecord,
        index: usize,
        outcome: McpCallOutcome,
    },
    Error {
        epoch: u64,
        message: String,
    },
    Done {
        epoch: u64,
        cancelled: bool,
    },
    ImageWritten {
        epoch: u64,
        path: PathBuf,
        message: String,
    },
    ImageFailed {
        epoch: u64,
        message: String,
    },
    ImageTestDone {
        epoch: u64,
        ok: bool,
        message: String,
    },
    DownloadProgress {
        epoch: u64,
        received: u64,
        total: u64,
        display: String,
    },
    DownloadDone {
        epoch: u64,
        path: PathBuf,
        display: String,
    },
    DownloadFailed {
        epoch: u64,
        display: String,
        message: String,
    },
}

fn to_chat_turn(entry: TurnEntry) -> ChatTurn {
    ChatTurn {
        kind: entry.kind,
        role: entry.role.into(),
        body: entry.body.into(),
    }
}

pub struct DesktopController {
    ui: slint::Weak<MainWindow>,
    sender: Sender<UiEvent>,
    receiver: Receiver<UiEvent>,
    backend: Option<Arc<dyn LlmBackend>>,
    loaded_kind: Option<BackendKind>,
    mcp: Arc<McpClient>,
    turns_model: Rc<VecModel<ChatTurn>>,
    session: ConversationSession,
    generation: Option<Arc<AtomicBool>>,
    stopping_since: Option<Instant>,
    pending_calls: VecDeque<(usize, ToolCallRecord)>,
    current_tool_round: u32,
    busy: bool,
    resource_ticks: u32,
    catalog: Vec<CatalogEntry>,
    catalog_key: String,
    auto_label: String,
    downloading: bool,
    pending_auto_load: bool,
    runtime_epoch: EpochGate,
    turn_epoch: EpochGate,
    mcp_epoch: EpochGate,
    image_epoch: EpochGate,
    delta_batch: DeltaBatcher,
}

impl DesktopController {
    fn new(ui: slint::Weak<MainWindow>) -> Self {
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
            stopping_since: None,
            pending_calls: VecDeque::new(),
            current_tool_round: 0,
            busy: false,
            resource_ticks: 0,
            catalog: Vec::new(),
            catalog_key: String::new(),
            auto_label: String::new(),
            downloading: false,
            pending_auto_load: false,
            runtime_epoch: EpochGate::default(),
            turn_epoch: EpochGate::default(),
            mcp_epoch: EpochGate::default(),
            image_epoch: EpochGate::default(),
            delta_batch: DeltaBatcher::default(),
        }
    }

    fn install(&self) {
        if let Some(ui) = self.ui.upgrade() {
            ui.set_turns(ModelRc::new(self.turns_model.clone()));
        }
    }

    fn tick(&mut self) {
        while let Ok(event) = self.receiver.try_recv() {
            self.apply(event);
        }
        if let Some(ui) = self.ui.upgrade() {
            self.flush_deltas(&ui);
            if let Some(since) = self.stopping_since {
                if since.elapsed() >= Duration::from_millis(STOP_WATCHDOG_MS) {
                    self.abandon_run(&ui, "Stopped (backend did not respond to cancel).");
                }
            }
        }
        self.resource_ticks += 1;
        if self.resource_ticks >= 20 {
            self.resource_ticks = 0;
            self.refresh_resources();
        }
    }

    fn flush_deltas(&mut self, ui: &MainWindow) {
        if self.delta_batch.is_empty() {
            return;
        }
        let (reasoning, text) = self.delta_batch.take();
        if !reasoning.is_empty() {
            if let Some((idx, entry, inserted)) = self.session.append_reasoning(&reasoning) {
                self.sync_turn(ui, idx, entry, inserted);
            }
        }
        if !text.is_empty() {
            if let Some((idx, entry, inserted)) = self.session.append_assistant(&text) {
                self.sync_turn(ui, idx, entry, inserted);
            }
        }
    }

    fn abort_remaining_tools(&mut self, ui: &MainWindow, reason: &str) {
        let aborted = self.session.abort_pending_tools(reason);
        if !aborted.is_empty() {
            for entry in aborted {
                self.turns_model.push(to_chat_turn(entry));
            }
            self.touch_transcript(ui);
        }
    }

    fn abandon_run(&mut self, ui: &MainWindow, status: &str) {
        self.turn_epoch.begin();
        self.stopping_since = None;
        if let Some(cancel) = self.generation.take() {
            cancel.store(true, Ordering::SeqCst);
        }
        self.flush_deltas(ui);
        let _ = self.session.finish_turn();
        self.pending_calls.clear();
        self.clear_tool_approval_ui(ui);
        self.abort_remaining_tools(ui, "stopped");
        self.set_busy(ui, false);
        ui.set_status(status.into());
        ui.set_status_error(false);
        self.refresh_resources();
    }

    fn apply(&mut self, event: UiEvent) {
        let Some(ui) = self.ui.upgrade() else { return };
        match event {
            UiEvent::Loaded {
                epoch,
                backend,
                model_label,
                status,
            } => {
                if !self.runtime_epoch.is_current(epoch) {
                    backend.unload();
                    return;
                }
                let kind = backend.kind();
                self.backend = Some(backend);
                self.loaded_kind = Some(kind);
                self.set_busy(&ui, false);
                ui.set_model_loaded(true);
                ui.set_model_indicator(model_label.clone().into());
                ui.set_empty_engine(format!("{} \u{00B7} {}", model_label, kind.label()).into());
                ui.set_status(status.into());
                ui.set_status_error(false);
                self.refresh_resources();
                self.save_now(&ui);
            }
            UiEvent::LoadFailed { epoch, message } => {
                if !self.runtime_epoch.is_current(epoch) {
                    return;
                }
                self.backend = None;
                self.loaded_kind = None;
                self.set_busy(&ui, false);
                ui.set_model_loaded(false);
                ui.set_status(format!("Load failed: {message}").into());
                ui.set_status_error(true);
            }
            UiEvent::Unloaded { epoch } => {
                if !self.runtime_epoch.is_current(epoch) {
                    return;
                }
                self.backend = None;
                self.loaded_kind = None;
                self.set_busy(&ui, false);
                ui.set_model_loaded(false);
                ui.set_model_indicator("No model".into());
                let kind = BackendKind::from_index(ui.get_backend_index());
                ui.set_empty_engine(idle_engine_label(kind).into());
                ui.set_resources("No runtime resources".into());
                ui.set_status("Runtime unloaded.".into());
                ui.set_status_error(false);
                self.save_now(&ui);
            }
            UiEvent::McpConnected { epoch, info } => {
                if !self.mcp_epoch.is_current(epoch) {
                    return;
                }
                self.set_busy(&ui, false);
                ui.set_mcp_connected(true);
                let summary = format!(
                    "{} \u{00B7} {} tools ({} ms)",
                    info.server_name,
                    info.tools.len(),
                    info.latency_ms
                );
                ui.set_mcp_status(summary.clone().into());
                ui.set_status(format!("MCP connected: {summary}").into());
                ui.set_status_error(false);
            }
            UiEvent::McpFailed { epoch, error } => {
                if !self.mcp_epoch.is_current(epoch) {
                    return;
                }
                self.set_busy(&ui, false);
                ui.set_mcp_connected(false);
                ui.set_mcp_status("Disconnected".into());
                ui.set_status(format!("MCP failed: {error}").into());
                ui.set_status_error(true);
            }
            UiEvent::Delta {
                epoch,
                text,
                reasoning,
            } => {
                if !self.turn_epoch.is_current(epoch) {
                    return;
                }
                self.delta_batch.push(text, reasoning);
            }
            UiEvent::ToolCall { epoch, call } => {
                if !self.turn_epoch.is_current(epoch) {
                    return;
                }
                self.flush_deltas(&ui);
                let entry = self.session.push_tool_call(call);
                self.turns_model.push(to_chat_turn(entry));
                self.touch_transcript(&ui);
            }
            UiEvent::ToolExecuted {
                epoch,
                call,
                index,
                outcome,
            } => {
                if !self.turn_epoch.is_current(epoch) {
                    return;
                }
                if let Some(entry) =
                    self.session
                        .resolve_tool(&call, index, &outcome.output, outcome.is_success)
                {
                    self.turns_model.push(to_chat_turn(entry));
                    self.touch_transcript(&ui);
                    self.prompt_next_tool_or_continue(&ui);
                }
            }
            UiEvent::Error { epoch, message } => {
                if !self.turn_epoch.is_current(epoch) {
                    return;
                }
                self.flush_deltas(&ui);
                let entry = self.session.push_error(&message);
                self.turns_model.push(to_chat_turn(entry));
                self.touch_transcript(&ui);
                ui.set_status(format!("Generation failed: {message}").into());
                ui.set_status_error(true);
            }
            UiEvent::Done { epoch, cancelled } => {
                if !self.turn_epoch.is_current(epoch) {
                    return;
                }
                self.flush_deltas(&ui);
                self.stopping_since = None;
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
                    self.touch_transcript(&ui);
                }
                self.generation = None;

                if cancelled || had_error {
                    self.pending_calls.clear();
                    self.clear_tool_approval_ui(&ui);
                    self.abort_remaining_tools(
                        &ui,
                        if cancelled { "cancelled" } else { "stream error" },
                    );
                    self.set_busy(&ui, false);
                    if cancelled {
                        ui.set_status("Stopped.".into());
                        ui.set_status_error(false);
                    }
                    self.refresh_resources();
                    return;
                }

                if !calls.is_empty() {
                    if !self.mcp.is_connected() {
                        self.abort_remaining_tools(&ui, "MCP not connected");
                        self.set_busy(&ui, false);
                        ui.set_status("Ready (MCP not connected for tool calls)".into());
                        ui.set_status_error(false);
                        self.refresh_resources();
                        return;
                    }
                    if self.current_tool_round >= MAX_TOOL_ROUNDS {
                        self.abort_remaining_tools(&ui, "tool round limit reached");
                        self.set_busy(&ui, false);
                        ui.set_status("Ready (tool round limit reached)".into());
                        ui.set_status_error(false);
                        self.refresh_resources();
                        return;
                    }
                    self.current_tool_round += 1;
                    for (idx, call) in calls.into_iter().take(MAX_CALLS_PER_TURN).enumerate() {
                        self.pending_calls.push_back((idx, call));
                    }
                    self.prompt_next_tool_or_continue(&ui);
                    return;
                }

                self.set_busy(&ui, false);
                ui.set_status("Ready".into());
                ui.set_status_error(false);
                self.refresh_resources();
            }
            UiEvent::ImageWritten {
                epoch,
                path,
                message,
            } => {
                if !self.image_epoch.is_current(epoch) {
                    return;
                }
                if ui.get_image_output().trim().is_empty() {
                    ui.set_image_output(path.to_string_lossy().to_string().into());
                }
                self.set_busy(&ui, false);
                self.save_now(&ui);
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
            UiEvent::ImageFailed { epoch, message } => {
                if !self.image_epoch.is_current(epoch) {
                    return;
                }
                self.set_busy(&ui, false);
                ui.set_status(format!("Image generation failed: {message}").into());
                ui.set_status_error(true);
            }
            UiEvent::ImageTestDone { epoch, ok, message } => {
                if !self.image_epoch.is_current(epoch) {
                    return;
                }
                self.set_busy(&ui, false);
                ui.set_status(message.clone().into());
                ui.set_status_error(!ok);
                self.save_now(&ui);
            }
            UiEvent::DownloadProgress {
                epoch,
                received,
                total,
                display,
            } => {
                if !self.runtime_epoch.is_current(epoch) {
                    return;
                }
                let ratio = if total > 0 {
                    (received as f32 / total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                ui.set_download_active(true);
                ui.set_download_progress(ratio);
                ui.set_download_status(
                    format!(
                        "Downloading {} \u{00B7} {} / {} ({:.0}%)",
                        display,
                        format_bytes(received),
                        format_bytes(total),
                        ratio * 100.0
                    )
                    .into(),
                );
                ui.set_status(format!("Downloading {display}...").into());
                ui.set_status_error(false);
            }
            UiEvent::DownloadDone {
                epoch,
                path,
                display,
            } => {
                if !self.runtime_epoch.is_current(epoch) {
                    return;
                }
                self.downloading = false;
                ui.set_download_active(false);
                self.refresh_catalog(&ui);
                self.save_now(&ui);
                if self.pending_auto_load {
                    self.pending_auto_load = false;
                    // Clear busy so load_model() proceeds directly into engine launch.
                    self.set_busy(&ui, false);
                    ui.set_status(
                        format!("Downloaded {display} \u{2014} starting the engine...").into(),
                    );
                    self.load_model();
                } else {
                    self.set_busy(&ui, false);
                    ui.set_status(
                        format!("Model downloaded: {}", path.display()).into(),
                    );
                    ui.set_status_error(false);
                }
            }
            UiEvent::DownloadFailed {
                epoch,
                display,
                message,
            } => {
                if !self.runtime_epoch.is_current(epoch) {
                    return;
                }
                self.downloading = false;
                self.pending_auto_load = false;
                ui.set_download_active(false);
                self.set_busy(&ui, false);
                ui.set_status(format!("Download of {display} failed: {message}").into());
                ui.set_status_error(true);
            }
        }
    }

    fn prompt_next_tool_or_continue(&mut self, ui: &MainWindow) {
        if let Some((_, next_call)) = self.pending_calls.front() {
            ui.set_awaiting_tool_approval(true);
            ui.set_pending_tool_title(format!("Approve MCP tool \"{}\"?", next_call.name).into());
            ui.set_pending_tool_args(next_call.arguments.clone().into());
            ui.set_status(format!("Awaiting approval for tool \"{}\"...", next_call.name).into());
            ui.set_status_error(false);
            return;
        }

        self.clear_tool_approval_ui(ui);
        // Any excess calls beyond MAX_CALLS_PER_TURN stay in pending_batch and must
        // be closed cleanly before sending the continuation request.
        self.abort_remaining_tools(ui, "per-turn call limit");
        let Some(backend) = self.backend.clone() else {
            self.set_busy(ui, false);
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
        ui.set_status(format!("Running MCP tool \"{}\"...", call.name).into());
        ui.set_status_error(false);
        let epoch = self.turn_epoch.current();
        let mcp = self.mcp.clone();
        let sender = self.sender.clone();
        thread::spawn(move || {
            let outcome = mcp.call_tool(&call.name, &call.arguments);
            let _ = sender.send(UiEvent::ToolExecuted {
                epoch,
                call,
                index,
                outcome,
            });
        });
    }

    fn decline_pending_tool(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let Some((index, call)) = self.pending_calls.pop_front() else {
            return;
        };
        self.pending_calls.clear();
        self.clear_tool_approval_ui(&ui);
        if let Some(entry) =
            self.session
                .resolve_tool(&call, index, "The user declined to run this tool.", false)
        {
            self.turns_model.push(to_chat_turn(entry));
            self.touch_transcript(&ui);
        }
        self.abort_remaining_tools(&ui, "batch declined");
        self.set_busy(&ui, false);
        ui.set_status("Tool call declined.".into());
        ui.set_status_error(false);
    }

    fn clear_tool_approval_ui(&self, ui: &MainWindow) {
        ui.set_awaiting_tool_approval(false);
        ui.set_pending_tool_title("".into());
        ui.set_pending_tool_args("".into());
    }

    fn connect_mcp(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy {
            return;
        }
        let url = ui.get_mcp_url().to_string();
        let token = ui.get_mcp_token().to_string();
        let epoch = self.mcp_epoch.begin();
        let mcp = self.mcp.clone();
        let sender = self.sender.clone();
        self.set_busy(&ui, true);
        ui.set_mcp_status("Connecting...".into());
        ui.set_status("Connecting to MCP server...".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            let token_opt = (!token.trim().is_empty()).then_some(token.as_str());
            match mcp.connect(&url, token_opt) {
                Ok(info) => {
                    let _ = sender.send(UiEvent::McpConnected { epoch, info });
                }
                Err(error) => {
                    let _ = sender.send(UiEvent::McpFailed { epoch, error });
                }
            }
        });
    }

    fn disconnect_mcp(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy {
            return;
        }
        self.mcp_epoch.begin();
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
            use slint::Model;
            self.turns_model.set_row_data(idx, turn);
        }
        self.touch_transcript(ui);
    }

    fn touch_transcript(&self, ui: &MainWindow) {
        ui.set_scroll_tick(ui.get_scroll_tick().wrapping_add(1));
    }

    fn set_busy(&mut self, ui: &MainWindow, busy: bool) {
        self.busy = busy;
        ui.set_busy(busy);
    }

    fn new_chat(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy {
            return;
        }
        self.turn_epoch.begin();
        self.delta_batch.clear();
        self.pending_calls.clear();
        self.clear_tool_approval_ui(&ui);
        self.session.clear();
        self.turns_model.set_vec(Vec::new());
        ui.set_status("Started a new conversation.".into());
        ui.set_status_error(false);
    }

    // ----- catalog & settings -------------------------------------------------

    fn selected_entry(&self, ui: &MainWindow) -> Option<CatalogEntry> {
        let index = ui.get_catalog_index() as usize;
        self.catalog.get(index).cloned()
    }

    fn refresh_catalog(&mut self, ui: &MainWindow) {
        let kind = BackendKind::from_index(ui.get_backend_index());
        let mut entries: Vec<CatalogEntry> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        match kind {
            BackendKind::LlamaCpp => {
                for model in catalog::models() {
                    let installed = catalog::is_installed(model);
                    labels.push(model.list_label(installed));
                    entries.push(CatalogEntry::Catalog(model));
                }
                for (name, path) in catalog::extra_local_models() {
                    labels.push(CatalogEntry::Local {
                        label: name.clone(),
                        path: path.clone(),
                    }
                    .label(true));
                    entries.push(CatalogEntry::Local { label: name, path });
                }
                let restored = if !self.catalog_key.is_empty() {
                    catalog::find_by_key(&self.catalog_key).and_then(|model| {
                        entries
                            .iter()
                            .position(|entry| matches!(entry, CatalogEntry::Catalog(m) if std::ptr::eq(*m, model)))
                    })
                } else {
                    None
                };
                ui.set_catalog_index(restored.map(|i| i as i32).unwrap_or(0));
            }
            BackendKind::Vllm => {
                let current = ui.get_vllm_model_path().to_string();
                for model in catalog::models() {
                    labels.push(model.display.to_string());
                    entries.push(CatalogEntry::Catalog(model));
                }
                if !current.trim().is_empty()
                    && !catalog::models().iter().any(|model| model.repo == current.trim())
                {
                    labels.insert(0, format!("{current}  (current)"));
                    entries.insert(0, CatalogEntry::Local { label: current.clone(), path: current });
                }
                let restored = entries.iter().position(|entry| match entry {
                    CatalogEntry::Local { path, .. } => *path == ui.get_vllm_model_path().to_string(),
                    _ => false,
                });
                ui.set_catalog_index(restored.map(|i| i as i32).unwrap_or(0));
            }
            BackendKind::Strata => {
                labels.push("Gateway-managed model".to_string());
                entries.push(CatalogEntry::Gateway);
                ui.set_catalog_index(0);
            }
        }
        self.catalog = entries;
        ui.set_catalog_labels(ModelRc::new(VecModel::from(
            labels
                .into_iter()
                .map(slint::SharedString::from)
                .collect::<Vec<_>>(),
        )));
        self.sync_catalog_detail(ui);
    }

    fn sync_catalog_detail(&mut self, ui: &MainWindow) {
        let note = match self.selected_entry(ui) {
            Some(CatalogEntry::Catalog(model)) => {
                format!("{} \u{00B7} {}", model.note_text(), model.note)
            }
            Some(CatalogEntry::Local { path, .. }) => {
                let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
                format!("local file \u{00B7} {}", format_bytes(size))
            }
            Some(CatalogEntry::Gateway) | None => {
                catalog::engine_guide(BackendKind::from_index(ui.get_backend_index()))
                    .recommendation
                    .to_string()
            }
        };
        ui.set_catalog_note(note.into());

        // Auto-derive the served model label unless the user typed their own.
        let auto = match self.selected_entry(ui) {
            Some(CatalogEntry::Catalog(model)) => model.display.to_string(),
            Some(CatalogEntry::Local { label, .. }) => label
                .trim_end_matches(".gguf")
                .to_string(),
            _ => String::new(),
        };
        let current = ui.get_model_label().to_string();
        if !auto.is_empty() && (current.trim().is_empty() || current == self.auto_label) {
            ui.set_model_label(auto.clone().into());
        }
        if !auto.is_empty() {
            self.auto_label = auto;
        }

        // Auto-tune the context window for catalog models (Advanced keeps control).
        if let Some(CatalogEntry::Catalog(model)) = self.selected_entry(ui) {
            if BackendKind::from_index(ui.get_backend_index()) == BackendKind::LlamaCpp {
                ui.set_llama_context_size(model.context.to_string().into());
            }
        }
    }

    fn set_engine_info(&self, ui: &MainWindow) {
        let guide = catalog::engine_guide(BackendKind::from_index(ui.get_backend_index()));
        ui.set_engine_info(
            format!("{}\n\n{}\n\n{}", guide.tagline, guide.strengths, guide.recommendation).into(),
        );
    }

    fn backend_changed(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let kind = BackendKind::from_index(ui.get_backend_index());
        let switching_away =
            self.loaded_kind.is_some_and(|loaded| loaded != kind) || self.busy || self.downloading;
        if switching_away {
            self.runtime_epoch.begin();
            self.turn_epoch.begin();
            self.stopping_since = None;
            if let Some(cancel) = self.generation.take() {
                cancel.store(true, Ordering::SeqCst);
            }
            self.delta_batch.clear();
            let _ = self.session.finish_turn();
            self.pending_calls.clear();
            self.clear_tool_approval_ui(&ui);
            self.abort_remaining_tools(&ui, "engine switched");
            self.downloading = false;
            self.pending_auto_load = false;
            ui.set_download_active(false);
            if let Some(previous) = self.backend.take() {
                thread::spawn(move || previous.unload());
            }
            self.loaded_kind = None;
            ui.set_model_loaded(false);
            ui.set_model_indicator("No model".into());
            ui.set_resources("No runtime resources".into());
            self.set_busy(&ui, false);
        }
        if self.backend.is_none() {
            ui.set_empty_engine(idle_engine_label(kind).into());
        }
        self.set_engine_info(&ui);
        self.refresh_catalog(&ui);
        ui.set_status(
            format!(
                "{} selected. Pick a model and press Load.",
                kind.label()
            )
            .into(),
        );
        ui.set_status_error(false);
        self.save_now(&ui);
    }

    fn catalog_selected(&mut self, _index: i32) {
        let Some(ui) = self.ui.upgrade() else { return };
        let kind = BackendKind::from_index(ui.get_backend_index());
        if kind == BackendKind::Vllm {
            if let Some(CatalogEntry::Catalog(model)) = self.selected_entry(&ui) {
                ui.set_vllm_model_path(model.repo.into());
            }
        }
        if let Some(CatalogEntry::Catalog(model)) = self.selected_entry(&ui) {
            self.catalog_key = model.key();
        }
        self.sync_catalog_detail(&ui);
        self.save_now(&ui);
    }

    fn advanced_toggled(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        ui.set_advanced_open(!ui.get_advanced_open());
        self.save_now(&ui);
    }

    fn snapshot_settings(&self, ui: &MainWindow) -> Settings {
        let catalog_key = match self.selected_entry(ui) {
            Some(CatalogEntry::Catalog(model)) => model.key(),
            _ => String::new(),
        };
        Settings {
            version: crate::persist::SETTINGS_VERSION,
            edition_index: ui.get_edition_index(),
            backend_index: ui.get_backend_index(),
            catalog_key,
            use_manual_model: ui.get_use_manual_model(),
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

    fn apply_settings(&mut self, settings: Settings) {
        let Some(ui) = self.ui.upgrade() else { return };
        self.catalog_key = settings.catalog_key.clone();
        ui.set_edition_index(settings.edition_index);
        if settings.edition_index == 0 && ui.get_tab_index() == 1 {
            ui.set_tab_index(0);
        }
        ui.set_backend_index(settings.backend_index);
        ui.set_model_label(settings.model_label.clone().into());
        if !settings.model_label.is_empty() {
            self.auto_label = settings.model_label.clone();
        }
        ui.set_llama_executable(
            if settings.llama_executable.is_empty() {
                "llama-server".to_string()
            } else {
                settings.llama_executable.clone()
            }
            .into(),
        );
        ui.set_llama_model_path(settings.llama_model_path.clone().into());
        ui.set_llama_port(
            if settings.llama_port.is_empty() {
                "8080".to_string()
            } else {
                settings.llama_port.clone()
            }
            .into(),
        );
        ui.set_llama_context_size(
            if settings.llama_context_size.is_empty() {
                "4096".to_string()
            } else {
                settings.llama_context_size.clone()
            }
            .into(),
        );
        ui.set_llama_gpu_layers(
            if settings.llama_gpu_layers.is_empty() {
                "0".to_string()
            } else {
                settings.llama_gpu_layers.clone()
            }
            .into(),
        );
        ui.set_vllm_endpoint(settings.vllm_endpoint.clone().into());
        ui.set_vllm_model_path(settings.vllm_model_path.clone().into());
        ui.set_vllm_port(
            if settings.vllm_port.is_empty() {
                "8000".to_string()
            } else {
                settings.vllm_port.clone()
            }
            .into(),
        );
        ui.set_vllm_managed(settings.vllm_managed);
        ui.set_strata_endpoint(settings.strata_endpoint.clone().into());
        ui.set_strata_launch(settings.strata_launch.clone().into());
        ui.set_temperature_text(
            if settings.temperature_text.is_empty() {
                "0.7".to_string()
            } else {
                settings.temperature_text.clone()
            }
            .into(),
        );
        ui.set_max_tokens_text(
            if settings.max_tokens_text.is_empty() {
                "1024".to_string()
            } else {
                settings.max_tokens_text.clone()
            }
            .into(),
        );
        ui.set_mcp_url(settings.mcp_url.clone().into());
        ui.set_image_endpoint(settings.image_endpoint.clone().into());
        ui.set_image_model(settings.image_model.clone().into());
        ui.set_image_size(
            if settings.image_size.is_empty() {
                "1024x1024".to_string()
            } else {
                settings.image_size.clone()
            }
            .into(),
        );
        ui.set_image_output(settings.image_output.clone().into());
        ui.set_use_manual_model(settings.use_manual_model);

        self.set_engine_info(&ui);
        self.refresh_catalog(&ui);
    }

    fn save_now(&self, ui: &MainWindow) {
        let settings = self.snapshot_settings(ui);
        let _ = save_settings(&settings);
    }

    // ----- model load / download ----------------------------------------------

    fn llama_model_source(&self, ui: &MainWindow) -> Result<String, Option<&'static CatalogModel>> {
        if ui.get_use_manual_model() {
            return Ok(ui.get_llama_model_path().to_string());
        }
        match self.selected_entry(ui) {
            Some(CatalogEntry::Catalog(model)) => match catalog::installed_path(model) {
                Some(path) => Ok(path.to_string_lossy().to_string()),
                None => Err(Some(model)),
            },
            Some(CatalogEntry::Local { path, .. }) => Ok(path),
            _ => Ok(ui.get_llama_model_path().to_string()),
        }
    }

    fn maybe_auto_restore(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if BackendKind::from_index(ui.get_backend_index()) != BackendKind::LlamaCpp {
            return;
        }
        match self.llama_model_source(&ui) {
            Ok(path) if !path.trim().is_empty() && PathBuf::from(sanitize_path(&path)).is_file() => {
                ui.set_status("Restoring your last session...".into());
                self.load_model();
            }
            _ => {
                ui.set_status(
                    "Pick a model from the list and press Load \u{2014} it downloads and starts in one click."
                        .into(),
                );
            }
        }
    }

    fn load_model(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy || self.downloading {
            return;
        }
        let kind = BackendKind::from_index(ui.get_backend_index());

        if kind == BackendKind::LlamaCpp {
            match self.llama_model_source(&ui) {
                Ok(path) if !path.trim().is_empty() => {}
                Ok(_) => {
                    ui.set_status("Pick a model from the list first.".into());
                    ui.set_status_error(true);
                    return;
                }
                Err(Some(model)) => {
                    self.start_download(model, true);
                    return;
                }
                Err(None) => {}
            }
        }

        let choice = self.backend_choice(&ui);
        let raw_label = ui.get_model_label().to_string();
        let model_label = if raw_label.trim().is_empty() {
            "local-model".to_string()
        } else {
            raw_label.trim().to_string()
        };
        let local_path = match kind {
            BackendKind::LlamaCpp => Some(self.llama_model_source(&ui).unwrap_or_default()),
            BackendKind::Vllm => Some(ui.get_vllm_model_path().to_string()),
            BackendKind::Strata => None,
        };
        let model = ModelSpec::new(model_label.clone(), local_path);
        let previous = self.backend.take();
        self.loaded_kind = None;
        let epoch = self.runtime_epoch.begin();
        let sender = self.sender.clone();
        self.set_busy(&ui, true);
        ui.set_status("Loading engine...".into());
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
                    let _ = sender.send(UiEvent::Loaded {
                        epoch,
                        backend,
                        model_label,
                        status,
                    });
                }
                Err(error) => {
                    backend.unload();
                    let _ = sender.send(UiEvent::LoadFailed {
                        epoch,
                        message: error,
                    });
                }
            }
        });
    }

    fn start_download(&mut self, model: &'static CatalogModel, auto_load: bool) {
        if self.downloading {
            return;
        }
        let Some(ui) = self.ui.upgrade() else { return };
        if let Err(error) = catalog::ensure_models_dir() {
            ui.set_status(error.into());
            ui.set_status_error(true);
            return;
        }
        let epoch = self.runtime_epoch.begin();
        self.downloading = true;
        self.pending_auto_load = auto_load;
        self.set_busy(&ui, true);
        ui.set_download_active(true);
        ui.set_download_progress(0.0);
        ui.set_download_status(
            format!("Downloading {} \u{00B7} {}", model.display, model.size_label()).into(),
        );
        ui.set_status(format!("Downloading {} from Hugging Face...", model.display).into());
        ui.set_status_error(false);

        let sender = self.sender.clone();
        let url = model.download_url();
        let target = catalog::models_dir().join(model.file);
        let display = model.display.to_string();
        let expected = model.size_bytes;
        thread::spawn(move || {
            let progress_sender = sender.clone();
            let progress_display = display.clone();
            let mut progress = move |received: u64, total: u64| {
                let _ = progress_sender.send(UiEvent::DownloadProgress {
                    epoch,
                    received,
                    total,
                    display: progress_display.clone(),
                });
            };
            let outcome = catalog::download_with_progress(&url, &target, expected, &mut progress);
            match outcome {
                Ok(path) => {
                    let _ = sender.send(UiEvent::DownloadDone {
                        epoch,
                        path,
                        display,
                    });
                }
                Err(message) => {
                    let _ = sender.send(UiEvent::DownloadFailed {
                        epoch,
                        display,
                        message,
                    });
                }
            }
        });
    }

    fn unload_model(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy {
            return;
        }
        let previous = self.backend.take();
        self.loaded_kind = None;
        self.turn_epoch.begin();
        let epoch = self.runtime_epoch.begin();
        let sender = self.sender.clone();
        self.set_busy(&ui, true);
        ui.set_status("Unloading runtime...".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            if let Some(previous) = previous {
                previous.unload();
            }
            let _ = sender.send(UiEvent::Unloaded { epoch });
        });
    }

    fn backend_choice(&self, ui: &MainWindow) -> BackendChoice {
        match BackendKind::from_index(ui.get_backend_index()) {
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
        if self.busy {
            return;
        }
        let selected_kind = BackendKind::from_index(ui.get_backend_index());
        let Some(backend) = self.backend.clone() else {
            ui.set_status("Press Load first \u{2014} it downloads and starts the model for you.".into());
            ui.set_status_error(true);
            return;
        };
        if self.loaded_kind != Some(selected_kind) {
            ui.set_status("Engine changed \u{2014} press Load for the selected engine.".into());
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
            return;
        };
        ui.set_input_text("".into());
        self.pending_calls.clear();
        self.current_tool_round = 0;
        self.turn_epoch.begin();
        self.delta_batch.clear();
        if let Some(last_user) = self.session.turns().last().cloned() {
            self.turns_model.push(to_chat_turn(last_user));
            self.touch_transcript(&ui);
        }
        self.spawn_stream(&ui, backend, request);
    }

    fn spawn_stream(
        &mut self,
        ui: &MainWindow,
        backend: Arc<dyn LlmBackend>,
        request: ChatRequest,
    ) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.generation = Some(cancel.clone());
        self.stopping_since = None;
        let epoch = self.turn_epoch.current();
        let sender = self.sender.clone();
        self.set_busy(ui, true);
        ui.set_status("Generating...".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            backend.stream(&request, &cancel, &mut |event| match event {
                StreamEvent::Delta { text, reasoning } => {
                    let _ = sender.send(UiEvent::Delta {
                        epoch,
                        text,
                        reasoning,
                    });
                }
                StreamEvent::ToolCall(call) => {
                    let _ = sender.send(UiEvent::ToolCall { epoch, call });
                }
                StreamEvent::Error(error) => {
                    let _ = sender.send(UiEvent::Error {
                        epoch,
                        message: error,
                    });
                }
                StreamEvent::Done => {}
            });
            let cancelled = cancel.load(Ordering::SeqCst);
            let _ = sender.send(UiEvent::Done { epoch, cancelled });
        });
    }

    fn stop_generation(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if !self.pending_calls.is_empty() {
            self.pending_calls.clear();
            self.clear_tool_approval_ui(&ui);
            self.abort_remaining_tools(&ui, "stopped by user");
            self.set_busy(&ui, false);
            ui.set_status("Stopped.".into());
            ui.set_status_error(false);
            return;
        }
        if let Some(cancel) = &self.generation {
            cancel.store(true, Ordering::SeqCst);
            if self.stopping_since.is_none() {
                self.stopping_since = Some(Instant::now());
            }
            ui.set_status("Stopping...".into());
            ui.set_status_error(false);
        }
    }

    fn generate_image(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy {
            return;
        }
        let prompt = ui.get_image_prompt().trim().to_string();
        if prompt.is_empty() {
            ui.set_status("Image prompt is required.".into());
            ui.set_status_error(true);
            return;
        }
        let explicit_output = ui.get_image_output().trim().to_string();
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
        let epoch = self.image_epoch.begin();
        let sender = self.sender.clone();
        self.set_busy(&ui, true);
        ui.set_status("Generating image...".into());
        ui.set_status_error(false);
        thread::spawn(move || match backend.generate(&request) {
            Ok(result) => {
                let output = if explicit_output.is_empty() {
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or_default();
                    let ext = match result.media_type.as_str() {
                        "image/jpeg" => "jpg",
                        "image/gif" => "gif",
                        "image/webp" => "webp",
                        _ => "png",
                    };
                    app_data_dir()
                        .join("images")
                        .join(format!("offcon-image-{stamp}.{ext}"))
                } else {
                    PathBuf::from(explicit_output)
                };
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
                        let _ = sender.send(UiEvent::ImageWritten {
                            epoch,
                            path: output,
                            message: format!("Image written to {}", display_path.display()),
                        });
                    }
                    Err(error) => {
                        let _ = sender.send(UiEvent::ImageFailed { epoch, message: error });
                    }
                }
            }
            Err(error) => {
                let _ = sender.send(UiEvent::ImageFailed { epoch, message: error });
            }
        });
    }

    fn test_image_endpoint(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        if self.busy {
            return;
        }
        let endpoint = ui.get_image_endpoint().to_string();
        if endpoint.trim().is_empty() {
            ui.set_status("Enter an image endpoint first, then press Test.".into());
            ui.set_status_error(true);
            return;
        }
        let epoch = self.image_epoch.begin();
        let sender = self.sender.clone();
        self.set_busy(&ui, true);
        ui.set_status("Testing image endpoint...".into());
        ui.set_status_error(false);
        thread::spawn(move || {
            let base = endpoint.trim().trim_end_matches('/').to_string();
            let url = if base.ends_with("/v1") {
                format!("{base}/models")
            } else {
                format!("{base}/v1/models")
            };
            let outcome = (|| -> Result<usize, String> {
                let client = reqwest::blocking::Client::builder()
                    .connect_timeout(Duration::from_secs(8))
                    .timeout(Duration::from_secs(12))
                    .build()
                    .map_err(|error| format!("HTTP client error: {error}"))?;
                let response = client
                    .get(&url)
                    .send()
                    .map_err(|error| format!("unreachable: {error}"))?;
                let status = response.status();
                let text = response
                    .text()
                    .map_err(|error| format!("unreadable response: {error}"))?;
                if !status.is_success() {
                    return Err(format!("HTTP {status}"));
                }
                let count = serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("data")
                            .and_then(|data| data.as_array())
                            .map(|items| items.len())
                    })
                    .unwrap_or(0);
                Ok(count)
            })();
            let event = match outcome {
                Ok(count) if count > 0 => UiEvent::ImageTestDone {
                    epoch,
                    ok: true,
                    message: format!("Image endpoint reachable \u{00B7} {count} models listed"),
                },
                Ok(_) => UiEvent::ImageTestDone {
                    epoch,
                    ok: true,
                    message: "Image endpoint reachable".to_string(),
                },
                Err(message) => UiEvent::ImageTestDone {
                    epoch,
                    ok: false,
                    message: format!("Image endpoint failed: {message}"),
                },
            };
            let _ = sender.send(event);
        });
    }

    fn refresh_resources(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let Some(backend) = self.backend.as_ref() else {
            ui.set_resources("No runtime resources".into());
            return;
        };
        let snapshot = backend.resources();
        ui.set_resources(format_resources(&snapshot).into());
    }

    fn shutdown(&mut self) {
        if let Some(cancel) = self.generation.take() {
            cancel.store(true, Ordering::SeqCst);
        }
        self.mcp.disconnect();
        if let Some(backend) = self.backend.take() {
            backend.unload();
        }
        if let Some(ui) = self.ui.upgrade() {
            self.save_now(&ui);
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
    let ui = MainWindow::new()?;
    let controller = Rc::new(RefCell::new(DesktopController::new(ui.as_weak())));
    controller.borrow().install();
    controller.borrow_mut().apply_settings(load_settings());
    {
        let ui_ref = &ui;
        controller.borrow().save_now(ui_ref);
    }

    macro_rules! wire {
        ($method:ident, $action:ident) => {{
            let weak = Rc::downgrade(&controller);
            ui.$method(move || {
                if let Some(ctrl) = weak.upgrade() {
                    ctrl.borrow_mut().$action();
                }
            });
        }};
        ($method:ident, $action:ident, arg) => {{
            let weak = Rc::downgrade(&controller);
            ui.$method(move |arg| {
                if let Some(ctrl) = weak.upgrade() {
                    ctrl.borrow_mut().$action(arg);
                }
            });
        }};
    }

    wire!(on_load_requested, load_model);
    wire!(on_unload_requested, unload_model);
    wire!(on_new_chat_requested, new_chat);
    wire!(on_send_requested, send_message);
    wire!(on_stop_requested, stop_generation);
    wire!(on_generate_image_requested, generate_image);
    wire!(on_image_test_requested, test_image_endpoint);
    wire!(on_backend_changed, backend_changed);
    wire!(on_catalog_selected, catalog_selected, arg);
    wire!(on_advanced_toggled, advanced_toggled);
    wire!(on_mcp_connect_requested, connect_mcp);
    wire!(on_mcp_disconnect_requested, disconnect_mcp);
    wire!(on_approve_tool_requested, approve_pending_tool);
    wire!(on_decline_tool_requested, decline_pending_tool);

    let timer = slint::Timer::default();
    {
        let weak = Rc::downgrade(&controller);
        timer.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(45),
            move || {
                if let Some(ctrl) = weak.upgrade() {
                    ctrl.borrow_mut().tick();
                }
            },
        );
    }

    controller.borrow_mut().maybe_auto_restore();

    let result = ui.run();
    drop(timer);
    controller.borrow_mut().shutdown();
    result
}
