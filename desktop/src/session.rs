//! UI-independent conversation state machine.
//! Keeps turn rendering records, OpenAI request history, and multi-round tool calls synchronized.

use crate::contracts::{ChatMessage, ChatRequest, OpenAiToolCall, ToolCallRecord, ToolDefinition};
use serde_json::{json, Value};
use std::collections::HashSet;

pub const KIND_USER: i32 = 0;
pub const KIND_ASSISTANT: i32 = 1;
pub const KIND_REASONING: i32 = 2;
pub const KIND_TOOL: i32 = 3;
pub const KIND_ERROR: i32 = 4;

const MAX_HISTORY_MESSAGES: usize = 40;
const MAX_HISTORY_ESTIMATED_TOKENS: usize = 16_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnEntry {
    pub kind: i32,
    pub role: String,
    pub body: String,
}

#[derive(Clone, Debug)]
struct PendingTool {
    id: String,
    name: String,
}

#[derive(Default)]
pub struct ConversationSession {
    turns: Vec<TurnEntry>,
    messages: Vec<ChatMessage>,
    active_reasoning_idx: Option<usize>,
    active_assistant_idx: Option<usize>,
    active_assistant_text: String,
    active_tool_calls: Vec<ToolCallRecord>,
    pending_batch: Vec<PendingTool>,
    next_call_seq: u64,
    pub had_error: bool,
}

impl ConversationSession {
    pub fn new() -> Self {
        Self::default()
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
        self.pending_batch.clear();
        self.had_error = false;
    }

    pub fn has_pending_tools(&self) -> bool {
        !self.pending_batch.is_empty()
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
        self.reset_active_buffers();

        self.turns.push(TurnEntry {
            kind: KIND_USER,
            role: "You".into(),
            body: clean.to_string(),
        });
        self.messages.push(ChatMessage::new("user", clean));

        Some(self.build_request(tools, temperature, max_tokens))
    }

    pub fn build_request(
        &self,
        tools: Vec<ToolDefinition>,
        temperature: f64,
        max_tokens: u32,
    ) -> ChatRequest {
        let prepared = trim_and_repair_history(
            &self.messages,
            MAX_HISTORY_MESSAGES,
            MAX_HISTORY_ESTIMATED_TOKENS,
        );
        let mut req = ChatRequest::text(prepared);
        req.tools = tools;
        req.temperature = temperature;
        req.max_tokens = max_tokens;
        req
    }

    pub fn prepare_continuation_round(&mut self) {
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
                self.turns[idx].body = visible;
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
                    body: visible,
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

    pub fn record_tool_result(&mut self, call: &ToolCallRecord, index: usize, output: &str, ok: bool) -> TurnEntry {
        let call_id = call
            .id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| format!("call_{index}"));
        self.pending_batch.retain(|item| item.id != call_id);
        self.push_tool_result_entry(call_id, &call.name, output, ok)
    }

    /// Resolves a single pending tool call by ID. Returns `None` if this ID is
    /// not pending (e.g. duplicate completion or already aborted).
    pub fn resolve_tool(
        &mut self,
        call: &ToolCallRecord,
        index: usize,
        output: &str,
        ok: bool,
    ) -> Option<TurnEntry> {
        let call_id = call
            .id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| format!("call_{index}"));
        let pos = self
            .pending_batch
            .iter()
            .position(|item| item.id == call_id)?;
        let pending = self.pending_batch.remove(pos);
        Some(self.push_tool_result_entry(pending.id, &pending.name, output, ok))
    }

    /// Closes every unresolved tool call in the current batch with a synthetic
    /// `role: "tool"` entry so the committed assistant `tool_calls` message never
    /// becomes orphaned in history.
    pub fn abort_pending_tools(&mut self, reason: &str) -> Vec<TurnEntry> {
        let remaining = std::mem::take(&mut self.pending_batch);
        let msg = format!("(not executed: {reason})");
        remaining
            .into_iter()
            .map(|pending| self.push_tool_result_entry(pending.id, &pending.name, &msg, false))
            .collect()
    }

    fn push_tool_result_entry(
        &mut self,
        call_id: String,
        name: &str,
        output: &str,
        ok: bool,
    ) -> TurnEntry {
        let status = if ok { "ok" } else { "error" };
        let truncated: String = output.chars().take(4000).collect();
        self.messages.push(ChatMessage::tool_result(
            call_id,
            name.to_string(),
            truncated.clone(),
        ));
        let entry = TurnEntry {
            kind: KIND_TOOL,
            role: format!("Tool result ({status}) \u{00B7} {name}"),
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
    /// Also assigns unique IDs to calls lacking one and tracks them in `pending_batch`
    /// so every exit path can guarantee complete `tool_calls` -> `tool` pairing.
    pub fn finish_turn(&mut self) -> (Vec<ToolCallRecord>, bool) {
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

        let clean_prose = strip_inline_tool_blocks(&self.active_assistant_text);
        self.active_assistant_text.clear();

        if !calls.is_empty() {
            let mut used_ids: HashSet<String> = self
                .messages
                .iter()
                .filter_map(|m| m.tool_calls.as_ref())
                .flatten()
                .map(|c| c.id.clone())
                .collect();
            for call in calls.iter_mut() {
                let needs_id = call
                    .id
                    .as_deref()
                    .map(|id| id.trim().is_empty() || used_ids.contains(id))
                    .unwrap_or(true);
                if needs_id {
                    loop {
                        self.next_call_seq += 1;
                        let candidate = format!("call_{}", self.next_call_seq);
                        if used_ids.insert(candidate.clone()) {
                            call.id = Some(candidate);
                            break;
                        }
                    }
                } else if let Some(id) = call.id.clone() {
                    used_ids.insert(id);
                }
            }
            self.pending_batch = calls
                .iter()
                .map(|call| PendingTool {
                    id: call.id.clone().unwrap_or_default(),
                    name: call.name.clone(),
                })
                .collect();
            self.messages
                .push(ChatMessage::assistant_with_tool_calls(clean_prose, &calls));
        } else if !clean_prose.is_empty() {
            self.messages
                .push(ChatMessage::new("assistant", clean_prose));
        }

        self.active_reasoning_idx = None;
        self.active_assistant_idx = None;
        (calls, from_inline)
    }
}

fn estimate_message_tokens(msg: &ChatMessage) -> usize {
    let mut chars = msg.content.len();
    if let Some(calls) = &msg.tool_calls {
        for call in calls {
            chars += call.function.name.len() + call.function.arguments.len() + 24;
        }
    }
    (chars / 4).max(4) + 4
}

/// Keeps the most recent messages within both count and approximate token budgets,
/// preserving leading system messages and repairing any cut edge so an OpenAI-strict
/// server never sees an orphan `role: "tool"` result or an assistant `tool_calls`
/// message whose results were partially/fully dropped.
pub fn trim_and_repair_history(
    messages: &[ChatMessage],
    max_messages: usize,
    max_tokens: usize,
) -> Vec<ChatMessage> {
    if messages.is_empty() {
        return Vec::new();
    }
    let system_end = messages
        .iter()
        .take_while(|m| m.role == "system")
        .count();
    let tail = &messages[system_end..];
    let system_tokens: usize = messages[..system_end]
        .iter()
        .map(estimate_message_tokens)
        .sum();
    let tail_budget = max_tokens.saturating_sub(system_tokens);
    let tail_limit = max_messages.saturating_sub(system_end).max(1);

    let mut kept_rev = Vec::new();
    let mut used_tokens = 0usize;
    for msg in tail.iter().rev() {
        let est = estimate_message_tokens(msg);
        if !kept_rev.is_empty()
            && (kept_rev.len() >= tail_limit || used_tokens + est > tail_budget)
        {
            break;
        }
        used_tokens += est;
        kept_rev.push(msg.clone());
    }
    kept_rev.reverse();

    // Seam repair at the cut edge: drop leading orphan `tool` messages whose
    // assistant `tool_calls` message fell before the window.
    while kept_rev.first().is_some_and(|m| m.role == "tool") {
        kept_rev.remove(0);
    }

    let mut combined: Vec<ChatMessage> = messages[..system_end].to_vec();
    combined.extend(kept_rev);
    normalize_history(&mut combined);
    combined
}

/// Ensures every assistant `tool_calls` entry in `messages` is paired with its
/// corresponding `role: "tool"` results, and drops any orphan `tool` messages.
pub fn normalize_history(messages: &mut Vec<ChatMessage>) {
    let mut satisfied_ids: HashSet<String> = messages
        .iter()
        .filter(|m| m.role == "tool")
        .filter_map(|m| m.tool_call_id.clone())
        .collect();

    // Remove `tool_calls` from any assistant message that lacks a matching result,
    // or prune individual unsatisfied calls.
    let mut valid_call_ids: HashSet<String> = HashSet::new();
    for msg in messages.iter_mut() {
        if msg.role == "assistant" {
            if let Some(calls) = msg.tool_calls.take() {
                let kept: Vec<OpenAiToolCall> = calls
                    .into_iter()
                    .filter(|c| satisfied_ids.contains(&c.id))
                    .collect();
                if !kept.is_empty() {
                    for c in &kept {
                        valid_call_ids.insert(c.id.clone());
                    }
                    msg.tool_calls = Some(kept);
                }
            }
        }
    }

    // Drop orphan tool results whose assistant `tool_calls` were dropped/trimmed,
    // and empty assistant placeholders whose dangling tool_calls were stripped.
    messages.retain(|m| {
        if m.role == "tool" {
            m.tool_call_id
                .as_deref()
                .is_some_and(|id| valid_call_ids.contains(id))
        } else if m.role == "assistant" && m.tool_calls.is_none() && m.content.trim().is_empty() {
            false
        } else {
            true
        }
    });
    let _ = &mut satisfied_ids;
}

fn strip_inline_tool_blocks(text: &str) -> String {
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
    out.trim().to_string()
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
            let Some(p_close) = val_part.find("</parameter>") else { break };
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
mod tests {
    use super::*;

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
    fn batch_integrity_resolves_once_and_aborts_remainder() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("run tools", Vec::new(), 0.7, 512);
        session.push_tool_call(ToolCallRecord {
            id: None,
            name: "alpha".into(),
            arguments: "{}".into(),
        });
        session.push_tool_call(ToolCallRecord {
            id: None,
            name: "beta".into(),
            arguments: "{}".into(),
        });
        let (calls, _) = session.finish_turn();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
        assert!(session.has_pending_tools());

        assert!(session.resolve_tool(&calls[0], 0, "ok-a", true).is_some());
        // Duplicate worker completion is ignored safely.
        assert!(session.resolve_tool(&calls[0], 0, "ok-a", true).is_none());

        let aborted = session.abort_pending_tools("declined");
        assert_eq!(aborted.len(), 1);
        assert!(!session.has_pending_tools());

        // Assistant message has 2 calls, followed by 2 tool results -> both survive normalization.
        let req = session.build_request(Vec::new(), 0.7, 512);
        let tool_msgs: Vec<_> = req.messages.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(tool_msgs.len(), 2);
        assert!(tool_msgs[1].content.contains("declined"));
    }

    #[test]
    fn trim_and_repair_drops_cut_orphans_and_preserves_system() {
        let mut msgs = vec![ChatMessage::new("system", "sys")];
        for i in 0..30 {
            msgs.push(ChatMessage::new("user", format!("u{i}")));
            let call = ToolCallRecord {
                id: Some(format!("id_{i}")),
                name: "t".into(),
                arguments: "{}".into(),
            };
            msgs.push(ChatMessage::assistant_with_tool_calls("", &[call]));
            msgs.push(ChatMessage::tool_result(format!("id_{i}"), "t", "res"));
        }
        // Cut inside the middle of a tool-call/result pair by keeping 10 messages.
        let trimmed = trim_and_repair_history(&msgs, 10, 16_000);
        assert_eq!(trimmed[0].role, "system");
        assert!(trimmed.len() <= 10);
        assert_ne!(trimmed[1].role, "tool");
        // Every assistant tool_call in `trimmed` has its matching tool result.
        let result_ids: HashSet<_> = trimmed
            .iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.tool_call_id.clone())
            .collect();
        for msg in &trimmed {
            if let Some(calls) = &msg.tool_calls {
                for call in calls {
                    assert!(result_ids.contains(&call.id));
                }
            }
        }
    }
}
