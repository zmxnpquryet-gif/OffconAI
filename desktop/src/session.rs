//! UI-independent conversation state machine.
//!
//! Keeps three things synchronized:
//! 1. the turn records the transcript renders,
//! 2. the OpenAI-compatible message history sent to the backend,
//! 3. the set of tool calls the assistant has committed but that have not produced a
//!    result yet.
//!
//! ## Why the "pending batch" exists
//!
//! OpenAI-compatible servers reject a request where an `assistant` message carries
//! `tool_calls` that are not each followed by exactly one `role: "tool"` message with
//! the matching `tool_call_id` (HTTP 400). The previous implementation committed the
//! assistant message as soon as the stream finished and then relied on the controller
//! to add results one by one, so declining, stopping, cancelling, hitting the tool-round
//! limit or an MCP failure left permanently invalid histories.
//!
//! This module now owns that invariant: a committed `tool_calls` message always has a
//! [`PendingBatch`] tracking which calls are still unresolved, and every exit path
//! either resolves each call exactly once ([`ConversationSession::resolve_tool`]) or
//! closes the whole batch deterministically
//! ([`ConversationSession::abort_pending_tools`]). [`ConversationSession::normalize_history`]
//! is the belt-and-braces repair pass run before every request and after restoring a
//! stored conversation.

use crate::contracts::{ChatMessage, ChatRequest, ToolCallRecord, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const KIND_USER: i32 = 0;
pub const KIND_ASSISTANT: i32 = 1;
pub const KIND_REASONING: i32 = 2;
pub const KIND_TOOL: i32 = 3;
pub const KIND_ERROR: i32 = 4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnEntry {
    pub kind: i32,
    pub role: String,
    pub body: String,
}

/// Why a committed tool call will never run.
///
/// The text is what the model sees as the `role: "tool"` content, so it can react to
/// the refusal instead of hallucinating a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolAbort {
    /// The user pressed Decline.
    Declined,
    /// The user stopped generation (Esc / Stop) while a batch was open.
    Cancelled,
    /// The app is closing or restoring a conversation; the call never ran.
    Interrupted,
    /// The call could not run: MCP is down, the round limit was hit, or a request failed.
    NotExecuted,
}

impl ToolAbort {
    pub fn text(self) -> &'static str {
        match self {
            Self::Declined => "The user declined to run this tool. Do not call it again unless asked.",
            Self::Cancelled => {
                "The user stopped the run before this tool executed. No output was produced."
            }
            Self::Interrupted => {
                "This tool call was interrupted before it executed (the run was closed or restored)."
            }
            Self::NotExecuted => {
                "This tool call was not executed because the tool runtime was unavailable or the tool-round limit was reached."
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Declined => "declined",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
            Self::NotExecuted => "not executed",
        }
    }
}

#[derive(Clone, Debug)]
struct PendingBatch {
    calls: Vec<ToolCallRecord>,
    resolved: Vec<bool>,
}

impl PendingBatch {
    fn new(calls: Vec<ToolCallRecord>) -> Self {
        let resolved = vec![false; calls.len()];
        Self { calls, resolved }
    }

    fn unresolved(&self) -> usize {
        self.resolved.iter().filter(|resolved| !**resolved).count()
    }
}

/// Structural check of the OpenAI message history: every committed tool call must have
/// exactly one result and every result must belong to a committed call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HistoryIntegrity {
    /// `tool_call_id`s that an assistant message promised but no tool message answered.
    pub orphan_tool_calls: Vec<String>,
    /// `tool_call_id`s answered without (or twice for) a matching assistant call.
    pub orphan_tool_results: Vec<String>,
}

impl HistoryIntegrity {
    pub fn is_clean(&self) -> bool {
        self.orphan_tool_calls.is_empty() && self.orphan_tool_results.is_empty()
    }
}

#[derive(Default)]
pub struct ConversationSession {
    turns: Vec<TurnEntry>,
    messages: Vec<ChatMessage>,
    active_reasoning_idx: Option<usize>,
    active_assistant_idx: Option<usize>,
    active_assistant_text: String,
    active_tool_calls: Vec<ToolCallRecord>,
    pending: Option<PendingBatch>,
    pub had_error: bool,
}

impl ConversationSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuilds a session from persisted parts. Runs the repair pass so a conversation
    /// saved mid-tool-call can never resurrect an invalid history.
    pub fn from_parts(turns: Vec<TurnEntry>, messages: Vec<ChatMessage>) -> Self {
        let mut session = Self {
            turns,
            messages,
            ..Self::default()
        };
        session.normalize_history();
        session
    }

    pub fn turns(&self) -> &[TurnEntry] {
        &self.turns
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.turns.clear();
        self.messages.clear();
        self.active_reasoning_idx = None;
        self.active_assistant_idx = None;
        self.active_assistant_text.clear();
        self.active_tool_calls.clear();
        self.pending = None;
        self.had_error = false;
    }

    /// Number of committed tool calls still waiting for a result.
    pub fn unresolved_tool_count(&self) -> usize {
        self.pending.as_ref().map(PendingBatch::unresolved).unwrap_or(0)
    }

    pub fn has_unresolved_tools(&self) -> bool {
        self.unresolved_tool_count() > 0
    }

    pub fn begin_user_turn(
        &mut self,
        prompt: &str,
        tools: Vec<ToolDefinition>,
        temperature: f64,
        max_tokens: u32,
    ) -> Option<ChatRequest> {
        let clean = prompt.trim();
        if clean.is_empty() {
            return None;
        }
        // A new prompt must never leave an older tool batch dangling.
        self.abort_pending_tools(ToolAbort::Interrupted);
        self.reset_active_buffers();

        self.turns.push(TurnEntry {
            kind: KIND_USER,
            role: "You".into(),
            body: clean.to_string(),
        });
        self.messages.push(ChatMessage::new("user", clean));

        Some(self.build_request(tools, temperature, max_tokens))
    }

    /// Builds the next request, repairing the history first so the wire payload is
    /// always accepted by the backend.
    pub fn build_request(
        &mut self,
        tools: Vec<ToolDefinition>,
        temperature: f64,
        max_tokens: u32,
    ) -> ChatRequest {
        self.normalize_history();
        let mut request = ChatRequest::text(self.messages.clone());
        request.tools = tools;
        request.temperature = temperature;
        request.max_tokens = max_tokens;
        request
    }

    pub fn prepare_continuation_round(&mut self) {
        self.abort_pending_tools(ToolAbort::Interrupted);
        self.reset_active_buffers();
    }

    fn reset_active_buffers(&mut self) {
        self.had_error = false;
        self.active_reasoning_idx = None;
        self.active_assistant_idx = None;
        self.active_assistant_text.clear();
        self.active_tool_calls.clear();
    }

    pub fn append_reasoning(&mut self, chunk: &str) -> Option<(usize, TurnEntry, bool)> {
        if chunk.is_empty() {
            return None;
        }
        match self.active_reasoning_idx {
            Some(idx) => {
                self.turns[idx].body.push_str(chunk);
                Some((idx, self.turns[idx].clone(), false))
            }
            None => {
                let idx = self.turns.len();
                let entry = TurnEntry {
                    kind: KIND_REASONING,
                    role: "Reasoning".into(),
                    body: chunk.to_string(),
                };
                self.turns.push(entry.clone());
                self.active_reasoning_idx = Some(idx);
                Some((idx, entry, true))
            }
        }
    }

    pub fn append_assistant(&mut self, chunk: &str) -> Option<(usize, TurnEntry, bool)> {
        if chunk.is_empty() {
            return None;
        }
        self.active_assistant_text.push_str(chunk);
        let visible = strip_inline_tool_blocks(&self.active_assistant_text);
        match self.active_assistant_idx {
            Some(idx) => {
                if self.turns[idx].body.as_str() != visible.as_ref() {
                    self.turns[idx].body.clear();
                    self.turns[idx].body.push_str(visible.as_ref());
                }
                Some((idx, self.turns[idx].clone(), false))
            }
            None => {
                if visible.is_empty() && self.active_assistant_text.contains("<tool_call>") {
                    return None;
                }
                let idx = self.turns.len();
                let entry = TurnEntry {
                    kind: KIND_ASSISTANT,
                    role: "OffconAI".into(),
                    body: visible.into_owned(),
                };
                self.turns.push(entry.clone());
                self.active_assistant_idx = Some(idx);
                Some((idx, entry, true))
            }
        }
    }

    pub fn push_tool_call(&mut self, call: ToolCallRecord) -> TurnEntry {
        let entry = TurnEntry {
            kind: KIND_TOOL,
            role: "Tool call".into(),
            body: call.display_text(),
        };
        self.active_tool_calls.push(call);
        self.turns.push(entry.clone());
        entry
    }

    /// Records the single result for the committed tool call at `index` in the open
    /// batch. Returns `None` when there is no such call or it was already resolved, so
    /// a duplicated worker event can never append a second result.
    pub fn resolve_tool(&mut self, index: usize, output: &str, ok: bool) -> Option<TurnEntry> {
        let call = {
            let pending = self.pending.as_ref()?;
            let call = pending.calls.get(index)?.clone();
            if pending.resolved.get(index).copied().unwrap_or(false) {
                return None;
            }
            call
        };
        if let Some(pending) = self.pending.as_mut() {
            if let Some(slot) = pending.resolved.get_mut(index) {
                *slot = true;
            }
        }
        let entry = self.push_tool_result(&call, output, ok);
        if self.pending.as_ref().map(PendingBatch::unresolved) == Some(0) {
            self.pending = None;
        }
        Some(entry)
    }

    /// Closes every unresolved call in the open batch with `reason`, in commit order.
    ///
    /// This is what makes Decline, Stop, Esc, cancel-on-exit, a failed request and the
    /// tool-round limit deterministic: whichever path runs, each committed `tool_call`
    /// ends up with exactly one `role: "tool"` message.
    pub fn abort_pending_tools(&mut self, reason: ToolAbort) -> Vec<TurnEntry> {
        let Some(pending) = self.pending.take() else {
            return Vec::new();
        };
        let mut entries = Vec::new();
        for (index, call) in pending.calls.iter().enumerate() {
            if pending.resolved.get(index).copied().unwrap_or(false) {
                continue;
            }
            entries.push(self.push_tool_result(call, reason.text(), false));
        }
        entries
    }

    fn push_tool_result(&mut self, call: &ToolCallRecord, output: &str, ok: bool) -> TurnEntry {
        let call_id = call
            .id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| "call_0".to_string());
        let status = if ok { "ok" } else { "error" };
        let truncated: String = output.chars().take(4000).collect();
        self.messages.push(ChatMessage::tool_result(
            call_id,
            call.name.clone(),
            truncated.clone(),
        ));
        let entry = TurnEntry {
            kind: KIND_TOOL,
            role: format!("Tool result ({status}) \u{00B7} {}", call.name),
            body: if truncated.trim().is_empty() {
                "(empty output)".into()
            } else {
                truncated
            },
        };
        self.turns.push(entry.clone());
        entry
    }

    pub fn push_error(&mut self, error: &str) -> TurnEntry {
        self.had_error = true;
        let entry = TurnEntry {
            kind: KIND_ERROR,
            role: "Error".into(),
            body: error.to_string(),
        };
        self.turns.push(entry.clone());
        entry
    }

    /// Commits the current assistant turn and returns `(tool_calls, parsed_from_inline_text)`.
    ///
    /// When tool calls are committed they are staged as a [`PendingBatch`]; the caller
    /// must then either resolve each one or abort the batch.
    pub fn finish_turn(&mut self) -> (Vec<ToolCallRecord>, bool) {
        // Never stack a second tool_calls message on an unresolved batch: the older
        // results have to be in the history before the newer assistant message.
        self.abort_pending_tools(ToolAbort::Interrupted);

        let mut calls = std::mem::take(&mut self.active_tool_calls);
        let mut from_inline = false;
        if calls.is_empty() && self.active_assistant_text.contains("<tool_call>") {
            calls = extract_inline_tool_calls(&self.active_assistant_text);
            from_inline = !calls.is_empty();
            for call in &calls {
                self.turns.push(TurnEntry {
                    kind: KIND_TOOL,
                    role: "Tool call".into(),
                    body: call.display_text(),
                });
            }
        }

        // Take the buffer instead of cloning it: the assistant text is consumed here, and
        // this keeps the borrow of `self` out of the way of the message push below.
        let raw_text = std::mem::take(&mut self.active_assistant_text);
        let clean_prose = strip_inline_tool_blocks(&raw_text);

        if !calls.is_empty() {
            assign_unique_ids(&mut calls);
            self.messages
                .push(ChatMessage::assistant_with_tool_calls(clean_prose.as_ref(), &calls));
            self.pending = Some(PendingBatch::new(calls.clone()));
        } else if !clean_prose.is_empty() {
            self.messages
                .push(ChatMessage::new("assistant", clean_prose.into_owned()));
        }

        self.active_reasoning_idx = None;
        self.active_assistant_idx = None;
        (calls, from_inline)
    }

    /// Structure of the current message history.
    pub fn integrity(&self) -> HistoryIntegrity {
        fn flush_open(open: &mut Vec<String>, result: &mut HistoryIntegrity) {
            result.orphan_tool_calls.append(open);
        }
        let mut result = HistoryIntegrity::default();
        let mut open: Vec<String> = Vec::new();
        for message in &self.messages {
            match message.role.as_str() {
                "assistant" => {
                    flush_open(&mut open, &mut result);
                    if let Some(calls) = &message.tool_calls {
                        for call in calls {
                            open.push(call.id.clone());
                        }
                    }
                }
                "tool" => {
                    let Some(id) = message.tool_call_id.clone() else {
                        result.orphan_tool_results.push("<missing-id>".to_string());
                        continue;
                    };
                    match open.iter().position(|open_id| *open_id == id) {
                        Some(position) => {
                            open.remove(position);
                        }
                        None => result.orphan_tool_results.push(id),
                    }
                }
                _ => flush_open(&mut open, &mut result),
            }
        }
        flush_open(&mut open, &mut result);
        result
    }

    /// Repairs the message history so the backend accepts it. Returns the number of
    /// repairs, which callers can surface as a diagnostic.
    ///
    /// - missing tool results are synthesized with [`ToolAbort::Interrupted`],
    /// - duplicate/unattached tool results are dropped,
    /// - `tool_calls: []` is normalized to no tool calls.
    pub fn normalize_history(&mut self) -> usize {
        let messages = std::mem::take(&mut self.messages);
        let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
        let mut repairs = 0usize;
        let mut iter = messages.into_iter().peekable();

        while let Some(message) = iter.next() {
            if message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
            {
                let calls = message.tool_calls.clone().unwrap_or_default();
                let mut following: Vec<ChatMessage> = Vec::new();
                while let Some(next) = iter.peek() {
                    if next.role == "tool" {
                        if let Some(result) = iter.next() {
                            following.push(result);
                        }
                    } else {
                        break;
                    }
                }

                let mut satisfied = vec![false; calls.len()];
                let mut kept: Vec<ChatMessage> = Vec::new();
                for result in following {
                    let matched = calls
                        .iter()
                        .position(|call| Some(&call.id) == result.tool_call_id.as_ref());
                    match matched {
                        Some(position) if !satisfied[position] => {
                            satisfied[position] = true;
                            kept.push(result);
                        }
                        _ => repairs += 1,
                    }
                }

                out.push(message);
                out.extend(kept);
                for (position, call) in calls.iter().enumerate() {
                    if !satisfied[position] {
                        repairs += 1;
                        out.push(ChatMessage::tool_result(
                            call.id.clone(),
                            call.function.name.clone(),
                            ToolAbort::Interrupted.text(),
                        ));
                    }
                }
                continue;
            }

            if message.role == "tool" {
                // A result with no preceding committed call cannot be sent.
                repairs += 1;
                continue;
            }

            if message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| calls.is_empty())
            {
                repairs += 1;
                let mut fixed = message;
                fixed.tool_calls = None;
                out.push(fixed);
                continue;
            }

            out.push(message);
        }

        self.messages = out;
        repairs
    }
}

/// Gives every committed call a unique non-empty id, preserving ids the server sent.
fn assign_unique_ids(calls: &mut [ToolCallRecord]) {
    let mut used: Vec<String> = Vec::with_capacity(calls.len());
    for (index, call) in calls.iter_mut().enumerate() {
        let base = call
            .id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("call_{index}"));
        let mut candidate = base.clone();
        let mut suffix = 2;
        while used.contains(&candidate) {
            candidate = format!("{base}_{suffix}");
            suffix += 1;
        }
        used.push(candidate.clone());
        call.id = Some(candidate);
    }
}

/// Removes complete `<tool_call>...</tool_call>` blocks, borrowing when there is
/// nothing to strip so a plain-prose stream costs no extra allocation.
fn strip_inline_tool_blocks(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains("<tool_call>") {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = text.to_string();
    while let Some(open) = out.find("<tool_call>") {
        if let Some(rel_close) = out[open + "<tool_call>".len()..].find("</tool_call>") {
            let end = open + "<tool_call>".len() + rel_close + "</tool_call>".len();
            out.replace_range(open..end, "");
        } else {
            out.truncate(open);
            break;
        }
    }
    std::borrow::Cow::Owned(out.trim().to_string())
}

/// Parses text-based `<tool_call>...</tool_call>` blocks (both JSON and `<function=...>`)
/// emitted by local models when a server does not convert them into `delta.tool_calls`.
pub fn extract_inline_tool_calls(text: &str) -> Vec<ToolCallRecord> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("<tool_call>") {
        let after_open = &rest[open + "<tool_call>".len()..];
        let Some(close) = after_open.find("</tool_call>") else {
            break;
        };
        let inner = after_open[..close].trim();
        if let Some(call) = parse_inline_call_body(inner, out.len()) {
            out.push(call);
        }
        rest = &after_open[close + "</tool_call>".len()..];
    }
    out
}

fn parse_inline_call_body(inner: &str, idx: usize) -> Option<ToolCallRecord> {
    if let Some(after_fn) = inner.strip_prefix("<function=") {
        let gt = after_fn.find('>')?;
        let name = after_fn[..gt].trim().to_string();
        if name.is_empty() {
            return None;
        }
        let mut args = serde_json::Map::new();
        let mut body = &after_fn[gt + 1..];
        while let Some(p_open) = body.find("<parameter=") {
            let after_p = &body[p_open + "<parameter=".len()..];
            let Some(p_gt) = after_p.find('>') else { break };
            let key = after_p[..p_gt].trim().to_string();
            let val_part = &after_p[p_gt + 1..];
            let Some(p_close) = val_part.find("</parameter>") else {
                break;
            };
            let val_str = val_part[..p_close].trim();
            args.insert(key, parse_xml_scalar(val_str));
            body = &val_part[p_close + "</parameter>".len()..];
        }
        return Some(ToolCallRecord {
            id: Some(format!("call_{idx}")),
            name,
            arguments: Value::Object(args).to_string(),
        });
    }

    let val: Value = serde_json::from_str(inner).ok()?;
    let target = val.get("function").unwrap_or(&val);
    let name = target.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = match target.get("arguments").or_else(|| target.get("parameters")) {
        Some(Value::Object(map)) => Value::Object(map.clone()).to_string(),
        Some(Value::String(s)) => s.clone(),
        _ => json!({}).to_string(),
    };
    Some(ToolCallRecord {
        id: Some(format!("call_{idx}")),
        name,
        arguments,
    })
}

fn parse_xml_scalar(raw: &str) -> Value {
    if raw.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if raw.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    let leading_zero = raw.len() > 1 && raw.starts_with('0') && !raw.starts_with("0.");
    if !leading_zero {
        if let Ok(i) = raw.parse::<i64>() {
            return json!(i);
        }
        if let Ok(f) = raw.parse::<f64>() {
            return json!(f);
        }
    }
    Value::String(raw.to_string())
}

#[cfg(test)]
pub(crate) fn tool_call_ids(message: &ChatMessage) -> Vec<String> {
    message
        .tool_calls
        .as_ref()
        .map(|calls| calls.iter().map(|call| call.id.clone()).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::OpenAiToolCall;

    fn call(name: &str, id: Option<&str>) -> ToolCallRecord {
        ToolCallRecord {
            id: id.map(str::to_string),
            name: name.into(),
            arguments: "{}".into(),
        }
    }

    fn tool_messages(session: &ConversationSession) -> Vec<&ChatMessage> {
        session
            .messages()
            .iter()
            .filter(|message| message.role == "tool")
            .collect()
    }

    fn assert_history_is_valid(session: &ConversationSession) {
        let integrity = session.integrity();
        assert!(
            integrity.is_clean(),
            "history must have no orphan tool call and no orphan result: {integrity:?}\n{:#?}",
            session.messages()
        );
        // Two independent checks: every call id has exactly one result, and every
        // result id is claimed by exactly one call.
        let mut promised: Vec<String> = Vec::new();
        let mut answered: Vec<String> = Vec::new();
        for message in session.messages() {
            if message.role == "assistant" {
                promised.extend(tool_call_ids(message));
            } else if message.role == "tool" {
                answered.push(message.tool_call_id.clone().unwrap_or_default());
            }
        }
        assert_eq!(promised.len(), answered.len(), "one result per call");
        let mut sorted_promised = promised.clone();
        let mut sorted_answered = answered.clone();
        sorted_promised.sort();
        sorted_answered.sort();
        assert_eq!(sorted_promised, sorted_answered, "ids line up exactly");
    }

    #[test]
    fn session_accumulates_reasoning_and_assistant_then_commits_history() {
        let mut session = ConversationSession::new();
        let req = session
            .begin_user_turn("  Explain KV quantization  ", Vec::new(), 0.4, 512)
            .expect("non-empty prompt");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content, "Explain KV quantization");
        assert_eq!(req.temperature, 0.4);
        assert_eq!(req.max_tokens, 512);

        let (_, _, inserted_r) = session.append_reasoning("step 1 ").unwrap();
        assert!(inserted_r);
        let (_, r2, inserted_r2) = session.append_reasoning("step 2").unwrap();
        assert!(!inserted_r2);
        assert_eq!(r2.body, "step 1 step 2");

        session.append_assistant("KV cache ");
        session.append_assistant("saves RAM.");
        let (calls, _) = session.finish_turn();
        assert!(calls.is_empty());

        assert_eq!(session.turns().len(), 3);
        assert_eq!(session.messages().len(), 2);
        assert_eq!(session.messages()[1].role, "assistant");
        assert_eq!(session.messages()[1].content, "KV cache saves RAM.");

        session.clear();
        assert!(session.turns().is_empty());
        assert!(session.messages().is_empty());
    }

    #[test]
    fn extracts_inline_json_and_qwen_xml_tool_calls() {
        let xml = "Checking.<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_inline_tool_calls(xml);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Seoul"}"#);
        assert_eq!(strip_inline_tool_blocks(xml), "Checking.");
    }

    #[test]
    fn declining_a_batch_resolves_every_committed_call_exactly_once() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("weather in three cities", Vec::new(), 0.7, 256);
        for name in ["get_weather", "get_weather", "get_forecast"] {
            session.push_tool_call(call(name, None));
        }
        let (calls, from_inline) = session.finish_turn();
        assert_eq!(calls.len(), 3);
        assert!(!from_inline);
        assert_eq!(session.unresolved_tool_count(), 3);

        // The user answers one prompt, then declines the rest.
        assert!(session.resolve_tool(0, "Seoul: 18C", true).is_some());
        // A duplicate worker event for the same slot is ignored.
        assert!(session.resolve_tool(0, "Seoul: 18C again", true).is_none());

        let closed = session.abort_pending_tools(ToolAbort::Declined);
        assert_eq!(closed.len(), 2, "the two remaining calls are closed");
        assert!(!session.has_unresolved_tools());
        assert_eq!(tool_messages(&session).len(), 3, "exactly one result per call");

        let declined = &tool_messages(&session)[1];
        assert_eq!(declined.content, ToolAbort::Declined.text());
        assert!(declined.content.contains("declined"));
        assert_history_is_valid(&session);

        // The next request is accepted by the backend because the history is valid.
        let request = session.begin_user_turn("thanks", Vec::new(), 0.7, 256).unwrap();
        assert_eq!(request.messages.len(), 6);
        assert_history_is_valid(&session);
    }

    #[test]
    fn stopping_mid_stream_closes_the_whole_pending_set() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("search everything", Vec::new(), 0.7, 256);
        session.push_tool_call(call("search", Some("call_a")));
        session.push_tool_call(call("fetch", Some("call_b")));
        let (calls, _) = session.finish_turn();
        assert_eq!(calls.len(), 2);

        // Esc / Stop before any approval.
        let closed = session.abort_pending_tools(ToolAbort::Cancelled);
        assert_eq!(closed.len(), 2);
        assert!(!session.has_unresolved_tools());
        let results = tool_messages(&session);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_call_id.as_deref(), Some("call_a"));
        assert_eq!(results[1].tool_call_id.as_deref(), Some("call_b"));
        for result in results {
            assert_eq!(result.content, ToolAbort::Cancelled.text());
        }
        assert_history_is_valid(&session);

        // Aborting again is a no-op, not a second result.
        assert!(session.abort_pending_tools(ToolAbort::Cancelled).is_empty());
        assert_eq!(tool_messages(&session).len(), 2);
    }

    #[test]
    fn cancelling_inline_tool_calls_leaves_no_orphans() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("what is the weather", Vec::new(), 0.7, 256);
        session.append_assistant(
            "Let me check.<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>",
        );
        let (calls, from_inline) = session.finish_turn();
        assert!(from_inline);
        assert_eq!(calls.len(), 1);
        // The prose keeps the sentence, the tool block is stripped from the transcript.
        assert_eq!(session.messages()[1].content, "Let me check.");
        assert_eq!(session.unresolved_tool_count(), 1);

        session.abort_pending_tools(ToolAbort::Cancelled);
        assert_history_is_valid(&session);
        assert_eq!(tool_messages(&session).len(), 1);
    }

    #[test]
    fn an_error_before_any_result_is_repaired_before_the_next_request() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("run the tool", Vec::new(), 0.7, 256);
        session.push_tool_call(call("dangerous_write", None));
        session.finish_turn();
        // The request failed: no result was ever recorded for the committed call.
        session.push_error("HTTP 500: upstream died");

        assert!(session.integrity().orphan_tool_calls.len() == 1);
        let request = session.begin_user_turn("try again", Vec::new(), 0.7, 256).unwrap();
        assert_history_is_valid(&session);
        // user, assistant(tool_calls), tool(repaired), user
        assert_eq!(request.messages.len(), 4);
        assert_eq!(request.messages[2].role, "tool");
        assert_eq!(request.messages[2].content, ToolAbort::Interrupted.text());
    }

    #[test]
    fn normalize_drops_orphan_and_duplicate_tool_results() {
        let mut session = ConversationSession::new();
        session.messages.push(ChatMessage::new("user", "hi"));
        session.messages.push(ChatMessage::new(
            "assistant",
            "calling",
        ));
        session.messages.last_mut().unwrap().tool_calls = Some(vec![
            OpenAiToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: crate::contracts::OpenAiFunctionCall {
                    name: "lookup".into(),
                    arguments: "{}".into(),
                },
            },
        ]);
        session
            .messages
            .push(ChatMessage::tool_result("call_1", "lookup", "first"));
        session
            .messages
            .push(ChatMessage::tool_result("call_1", "lookup", "duplicate"));
        session
            .messages
            .push(ChatMessage::tool_result("call_ghost", "lookup", "orphan"));

        let repairs = session.normalize_history();
        assert_eq!(repairs, 2, "duplicate + orphan results are dropped");
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[2].content, "first");
        assert_history_is_valid(&session);

        // A tool_calls list that is present but empty is not valid on the wire.
        let mut empty = ConversationSession::new();
        let mut message = ChatMessage::new("assistant", "no calls");
        message.tool_calls = Some(Vec::new());
        empty.messages.push(message);
        assert_eq!(empty.normalize_history(), 1);
        assert!(empty.messages[0].tool_calls.is_none());
        assert_history_is_valid(&empty);
    }

    #[test]
    fn restored_sessions_repair_dangling_tool_calls_and_synthesize_missing_ids() {
        // Simulates a session file saved while a tool call was still open.
        let mut messages = vec![ChatMessage::new("user", "check the weather")];
        messages.push(ChatMessage::assistant_with_tool_calls(
            "",
            &[call("get_weather", None), call("get_weather", None)],
        ));
        let session = ConversationSession::from_parts(Vec::new(), messages);
        assert!(!session.has_unresolved_tools());
        assert_eq!(tool_messages(&session).len(), 2);
        assert_history_is_valid(&session);

        let ids: Vec<String> = tool_call_ids(&session.messages()[1]);
        assert_eq!(ids, vec!["call_0".to_string(), "call_1".to_string()]);
        let results: Vec<String> = tool_messages(&session)
            .iter()
            .map(|message| message.tool_call_id.clone().unwrap_or_default())
            .collect();
        assert_eq!(results, ids);
    }

    #[test]
    fn duplicate_server_ids_are_made_unique_so_results_stay_attributable() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("go", Vec::new(), 0.7, 128);
        session.push_tool_call(call("a", Some("call_1")));
        session.push_tool_call(call("b", Some("call_1")));
        let (calls, _) = session.finish_turn();
        let ids: Vec<String> = calls.iter().filter_map(|call| call.id.clone()).collect();
        assert_eq!(ids, vec!["call_1".to_string(), "call_1_2".to_string()]);
        assert_eq!(tool_call_ids(&session.messages()[1]), ids);
        session.abort_pending_tools(ToolAbort::NotExecuted);
        assert_history_is_valid(&session);
    }

    #[test]
    fn a_new_prompt_and_a_new_round_never_leave_an_old_batch_open() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("first", Vec::new(), 0.7, 128);
        session.push_tool_call(call("a", None));
        session.finish_turn();
        assert!(session.has_unresolved_tools());

        // Whatever went wrong, starting the next request must not poison it.
        session.prepare_continuation_round();
        assert!(!session.has_unresolved_tools());
        assert_history_is_valid(&session);

        session.push_tool_call(call("b", None));
        session.finish_turn();
        session.begin_user_turn("second", Vec::new(), 0.7, 128);
        assert!(!session.has_unresolved_tools());
        assert_history_is_valid(&session);
    }
}
