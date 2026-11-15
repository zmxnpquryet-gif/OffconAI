//! UI-independent conversation state machine.
//! Keeps turn rendering records, OpenAI request history, and multi-round tool calls synchronized.

use crate::contracts::{ChatMessage, ChatRequest, ToolCallRecord, ToolDefinition};
use serde_json::{json, Value};

pub const KIND_USER: i32 = 0;
pub const KIND_ASSISTANT: i32 = 1;
pub const KIND_REASONING: i32 = 2;
pub const KIND_TOOL: i32 = 3;
pub const KIND_ERROR: i32 = 4;
pub const MAX_TOOL_CALLS_PER_RESPONSE: usize = 4;
/// Cap on messages shipped per request. The UI transcript (`turns`) keeps full
/// history; only the API payload is windowed so long sessions cannot overflow
/// the model's context or slow every request with megabytes of history.
pub const MAX_HISTORY_MESSAGES: usize = 40;
/// Token budget cap on history messages. If messages within the message cap
/// exceed this token threshold, older messages are trimmed to prevent server 400s.
pub const MAX_HISTORY_TOKENS: usize = 16_384;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnEntry {
    pub kind: i32,
    pub role: String,
    pub body: String,
}

#[derive(Default)]
pub struct ConversationSession {
    turns: Vec<TurnEntry>,
    messages: Vec<ChatMessage>,
    system_prompt: String,
    active_reasoning_idx: Option<usize>,
    active_assistant_idx: Option<usize>,
    active_assistant_text: String,
    pending_inline_text: String,
    active_tool_calls: Vec<ToolCallRecord>,
    pub had_error: bool,
    active_in_think: bool,
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

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    pub fn clear(&mut self) {
        self.turns.clear();
        self.messages.clear();
        self.active_reasoning_idx = None;
        self.active_assistant_idx = None;
        self.active_assistant_text.clear();
        self.pending_inline_text.clear();
        self.active_tool_calls.clear();
        self.had_error = false;
        self.active_in_think = false;
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
        let mut messages = Vec::new();
        let sys = self.system_prompt.trim();
        if !sys.is_empty() {
            messages.push(ChatMessage::new("system", sys));
        }
        messages.extend(trim_history(self.messages.clone()));
        let mut req = ChatRequest::text(messages);
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
        self.pending_inline_text.clear();
        self.active_tool_calls.clear();
        self.active_in_think = false;
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

    /// Ingests streaming delta text/reasoning. If the model streams inline `<think>...</think>`,
    /// it routes the thinking part to reasoning turns and the answer part to assistant turns.
    pub fn ingest_delta(
        &mut self,
        text_opt: Option<&str>,
        reasoning_opt: Option<&str>,
    ) -> Vec<(bool /* is_reasoning */, usize, TurnEntry, bool /* inserted */)> {
        let mut updates = Vec::new();

        if let Some(r) = reasoning_opt {
            if let Some((idx, entry, ins)) = self.append_reasoning(r) {
                updates.push((true, idx, entry, ins));
            }
        }

        if let Some(text) = text_opt.filter(|text| !text.is_empty()) {
            self.pending_inline_text.push_str(text);
        }

        let mut remaining = std::mem::take(&mut self.pending_inline_text);
        while !remaining.is_empty() {
            let candidates = if self.active_in_think {
                CLOSE_THINK_MARKERS
            } else {
                OPEN_THINK_MARKERS
            };
            if let Some((marker_pos, matched_marker)) = find_earliest_marker(&remaining, candidates) {
                if let Some((is_reasoning, idx, entry, inserted)) =
                    self.append_stream_text(&remaining[..marker_pos])
                {
                    updates.push((is_reasoning, idx, entry, inserted));
                }
                self.active_in_think = !self.active_in_think;
                remaining = remaining[marker_pos + matched_marker.len()..].to_string();
            } else {
                let keep = candidates
                    .iter()
                    .map(|m| trailing_marker_prefix_len(&remaining, m))
                    .max()
                    .unwrap_or(0);
                let safe_end = remaining.len() - keep;
                if let Some((is_reasoning, idx, entry, inserted)) =
                    self.append_stream_text(&remaining[..safe_end])
                {
                    updates.push((is_reasoning, idx, entry, inserted));
                }
                self.pending_inline_text = remaining[safe_end..].to_string();
                break;
            }
        }

        updates
    }

    fn append_stream_text(&mut self, chunk: &str) -> Option<(bool, usize, TurnEntry, bool)> {
        if self.active_in_think {
            self.append_reasoning(chunk)
                .map(|(idx, entry, inserted)| (true, idx, entry, inserted))
        } else {
            self.append_assistant(chunk)
                .map(|(idx, entry, inserted)| (false, idx, entry, inserted))
        }
    }

    pub fn flush_pending_delta(&mut self) -> Option<(bool, usize, TurnEntry, bool)> {
        let pending = std::mem::take(&mut self.pending_inline_text);
        self.append_stream_text(&pending)
    }

    pub fn push_tool_call(&mut self, call: ToolCallRecord) -> Option<TurnEntry> {
        if self.active_tool_calls.len() >= MAX_TOOL_CALLS_PER_RESPONSE {
            return None;
        }
        let entry = TurnEntry {
            kind: KIND_TOOL,
            role: "Tool call".into(),
            body: call.display_text(),
        };
        self.active_tool_calls.push(call);
        self.turns.push(entry.clone());
        Some(entry)
    }

    pub fn record_tool_result(
        &mut self,
        call: &ToolCallRecord,
        index: usize,
        output: &str,
        ok: bool,
    ) -> TurnEntry {
        let call_id = call
            .id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| format!("call_{index}"));
        let status = if ok { "ok" } else { "error" };
        let mut output_chars = output.chars();
        let mut truncated: String = output_chars.by_ref().take(4000).collect();
        if output_chars.next().is_some() {
            truncated.push_str("\n… (truncated to 4000 chars)");
        }
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
    pub fn finish_turn(&mut self) -> (Vec<ToolCallRecord>, bool) {
        let _ = self.flush_pending_delta();
        self.active_in_think = false;
        let mut calls = std::mem::take(&mut self.active_tool_calls);
        let mut from_inline = false;
        if calls.is_empty() {
            calls = extract_inline_tool_calls(&self.active_assistant_text);
            if calls.is_empty() {
                if let Some(call) = extract_plain_json_tool_call(&self.active_assistant_text) {
                    calls = vec![call];
                }
            }
            calls.truncate(MAX_TOOL_CALLS_PER_RESPONSE);
            from_inline = !calls.is_empty();
            for call in &calls {
                self.turns.push(TurnEntry {
                    kind: KIND_TOOL,
                    role: "Tool call".into(),
                    body: call.display_text(),
                });
            }
        }

        let clean_prose = strip_inline_tool_blocks(&self.active_assistant_text)
            .trim()
            .to_string();
        self.active_assistant_text.clear();

        // If an assistant turn was added but ended up containing only tool calls, remove or clean it
        if let Some(idx) = self.active_assistant_idx {
            if clean_prose.is_empty() && !calls.is_empty() {
                if idx < self.turns.len() && self.turns[idx].kind == KIND_ASSISTANT {
                    self.turns.remove(idx);
                }
            } else if idx < self.turns.len() {
                self.turns[idx].body = clean_prose.clone();
            }
        }

        if !calls.is_empty() {
            for (idx, call) in calls.iter_mut().enumerate() {
                if call.id.as_deref().unwrap_or("").is_empty() {
                    call.id = Some(format!("call_{idx}"));
                }
            }
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

    pub fn export_markdown(&self) -> String {
        let mut doc = String::new();
        doc.push_str("# OffconAI Conversation Export\n\n");
        let sys = self.system_prompt.trim();
        if !sys.is_empty() {
            doc.push_str(&format!("> **System Prompt**: {sys}\n\n---\n\n"));
        }
        for turn in &self.turns {
            match turn.kind {
                KIND_USER => {
                    doc.push_str("### 👤 User\n\n");
                    doc.push_str(&turn.body);
                    doc.push_str("\n\n");
                }
                KIND_ASSISTANT => {
                    doc.push_str("### 🤖 OffconAI\n\n");
                    doc.push_str(&turn.body);
                    doc.push_str("\n\n");
                }
                KIND_REASONING => {
                    doc.push_str("<details><summary>💭 Reasoning</summary>\n\n");
                    doc.push_str(&turn.body);
                    doc.push_str("\n\n</details>\n\n");
                }
                KIND_TOOL => {
                    doc.push_str(&format!("> 🛠️ **{}**\n```json\n{}\n```\n\n", turn.role, turn.body));
                }
                KIND_ERROR => {
                    doc.push_str(&format!("> ⚠️ **Error**: {}\n\n", turn.body));
                }
                _ => {
                    doc.push_str(&format!("### {}\n\n{}\n\n", turn.role, turn.body));
                }
            }
        }
        doc
    }

    pub fn total_estimated_tokens(&self) -> usize {
        let sys_tokens = if self.system_prompt.trim().is_empty() {
            0
        } else {
            estimate_text_tokens(&self.system_prompt) + 4
        };
        sys_tokens + estimate_messages_tokens(&self.messages)
    }

    pub fn export_json(&self) -> String {
        let turns_json: Vec<Value> = self
            .turns
            .iter()
            .map(|t| {
                json!({
                    "kind": t.kind,
                    "role": t.role,
                    "body": t.body,
                })
            })
            .collect();
        let payload = json!({
            "version": "2.0.0",
            "system_prompt": self.system_prompt,
            "turns": turns_json,
            "messages": self.messages,
            "estimated_tokens": self.total_estimated_tokens(),
            "total_turns": self.turns.len(),
        });
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".into())
    }
}

pub fn estimate_text_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut tokens = 0;
    let mut in_word = false;
    for ch in text.chars() {
        if ch.is_ascii_whitespace() {
            in_word = false;
        } else if ch.is_ascii_alphanumeric() {
            if !in_word {
                tokens += 1;
                in_word = true;
            }
        } else {
            tokens += 1;
            in_word = false;
        }
    }
    tokens.max(text.len() / 4).max(1)
}

pub fn estimate_message_tokens(msg: &ChatMessage) -> usize {
    let mut total = 4;
    total += estimate_text_tokens(&msg.content);
    if let Some(calls) = &msg.tool_calls {
        for call in calls {
            total += 8;
            total += estimate_text_tokens(&call.function.name);
            total += estimate_text_tokens(&call.function.arguments);
        }
    }
    total
}

pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

const OPEN_THINK_MARKERS: &[&str] = &["<think>", "<thought>", "<thinking>"];
const CLOSE_THINK_MARKERS: &[&str] = &["</think>", "</thought>", "</thinking>"];

fn find_earliest_marker<'a>(text: &str, markers: &[&'a str]) -> Option<(usize, &'a str)> {
    let mut earliest: Option<(usize, &'a str)> = None;
    for &m in markers {
        if let Some(pos) = text.find(m) {
            match earliest {
                Some((min_pos, _)) if pos < min_pos => earliest = Some((pos, m)),
                None => earliest = Some((pos, m)),
                _ => {}
            }
        }
    }
    earliest
}

fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
    (1..marker.len().min(text.len() + 1))
        .rev()
        .find(|length| text.ends_with(&marker[..*length]))
        .unwrap_or(0)
}

/// Strips `<tool_call>...</tool_call>` blocks while PRESERVING regular surrounding whitespace and code indentation.
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
    out
}

/// Parses text-based `<tool_call>...</tool_call>` blocks (JSON, `<function=...>`, and markdown codeblocks)
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

    // Fallback: check for fenced markdown codeblocks like ```tool_call or ```json with name/arguments
    if out.is_empty() {
        let mut fence_rest = text;
        while let Some(open) = fence_rest.find("```") {
            let after_open = &fence_rest[open + 3..];
            let Some(close) = after_open.find("```") else { break };
            let block = after_open[..close].trim();
            let body = block
                .strip_prefix("tool_call")
                .or_else(|| block.strip_prefix("json"))
                .unwrap_or(block)
                .trim();
            if let Some(call) = parse_inline_call_body(body, out.len()) {
                out.push(call);
            }
            fence_rest = &after_open[close + 3..];
        }
    }

    out
}

/// Bare-JSON fallback: a whole assistant turn that is exactly one JSON tool call
/// (Hermes/OpenAI style) without any `<tool_call>` wrapper or code fence.
fn extract_plain_json_tool_call(text: &str) -> Option<ToolCallRecord> {
    let trimmed = text.trim();
    if trimmed.is_empty() || !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    if trimmed.contains("<tool_call>") || trimmed.contains("```") {
        return None;
    }
    let val: Value = serde_json::from_str(trimmed).ok()?;
    // Require an explicit tool shape so a plain JSON answer that merely
    // contains a "name" field (e.g. `{"name": "Ada"}`) is not mistaken
    // for a tool call.
    let target = val.get("function").unwrap_or(&val);
    if target.get("arguments").is_none() && target.get("parameters").is_none() {
        return None;
    }
    parse_json_tool_value(&val, 0)
}

fn parse_inline_call_body(inner: &str, idx: usize) -> Option<ToolCallRecord> {
    let trimmed = inner.trim();
    if let Some(after_fn) = trimmed.strip_prefix("<function=") {
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

    let val: Value = serde_json::from_str(trimmed).ok()?;
    if let Value::Array(arr) = &val {
        if let Some(first) = arr.first() {
            return parse_json_tool_value(first, idx);
        }
    }
    parse_json_tool_value(&val, idx)
}

/// Sliding window over stored history. Trims from the front and then repairs
/// the cut edge so the request never starts mid tool-sequence: leading
/// orphan `tool` results are dropped, and an assistant message whose tool
/// results were cut loses its dangling `tool_calls` (plain prose is kept).
fn trim_history(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    if messages.is_empty() {
        return messages;
    }
    let mut kept = if messages.len() > MAX_HISTORY_MESSAGES {
        messages[messages.len() - MAX_HISTORY_MESSAGES..].to_vec()
    } else {
        messages
    };
    while kept.len() > 2 && estimate_messages_tokens(&kept) > MAX_HISTORY_TOKENS {
        kept.remove(0);
    }
    loop {
        match kept.first() {
            // Orphaned tool result whose assistant call was trimmed away.
            Some(first) if first.role == "tool" => kept.remove(0),
            // Assistant message that carried only (now trimmed) tool calls.
            Some(first)
                if first.role == "assistant"
                    && first.tool_calls.is_some()
                    && first.content.trim().is_empty() =>
            {
                kept.remove(0)
            }
            _ => break,
        };
    }
    // An assistant message that kept its prose but lost its tool results
    // must not ship dangling `tool_calls` — strict servers reject those.
    if let Some(first) = kept.first_mut() {
        if first.role == "assistant" && first.tool_calls.is_some() {
            first.tool_calls = None;
        }
    }
    kept
}

fn parse_json_tool_value(val: &Value, idx: usize) -> Option<ToolCallRecord> {
    let target = val.get("function").unwrap_or(val);
    let name = target.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = match target.get("arguments").or_else(|| target.get("parameters")) {
        Some(Value::Object(map)) => Value::Object(map.clone()).to_string(),
        Some(Value::String(s)) => {
            if let Ok(Value::Object(inner_map)) = serde_json::from_str::<Value>(s) {
                Value::Object(inner_map).to_string()
            } else {
                s.clone()
            }
        }
        Some(other) => other.to_string(),
        None => json!({}).to_string(),
    };
    let id = target
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| Some(format!("call_{idx}")));
    Some(ToolCallRecord { id, name, arguments })
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
        session.set_system_prompt("You are a helpful assistant.");
        let req = session
            .begin_user_turn("  Explain KV quantization  ", Vec::new(), 0.4, 512)
            .expect("non-empty prompt");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[0].content, "You are a helpful assistant.");
        assert_eq!(req.messages[1].role, "user");
        assert_eq!(req.messages[1].content, "Explain KV quantization");
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

        let md = session.export_markdown();
        assert!(md.contains("System Prompt"));
        assert!(md.contains("Explain KV quantization"));
        assert!(md.contains("KV cache saves RAM."));

        session.clear();
        assert!(session.turns().is_empty());
        assert!(session.messages().is_empty());
    }

    #[test]
    fn history_window_never_starts_mid_tool_sequence() {
        // 50 plain exchanges -> windowed to MAX_HISTORY_MESSAGES.
        let mut session = ConversationSession::new();
        for i in 0..50 {
            session.messages.push(ChatMessage::new("user", format!("q{i}")));
            session.messages.push(ChatMessage::new("assistant", format!("a{i}")));
        }
        let req = session.build_request(Vec::new(), 0.7, 128);
        assert!(req.messages.len() <= MAX_HISTORY_MESSAGES + 1);
        assert_eq!(req.messages.last().unwrap().content, "a49");

        // Cut edge landing on an orphan tool result: it is dropped so the
        // request never starts mid tool-sequence.
        let mut seq = vec![
            ChatMessage::new("user", "first"),
            ChatMessage::tool_result("call_0", "lookup", "42"),
        ];
        for i in 0..39 {
            seq.push(ChatMessage::new("user", format!("q{i}")));
        }
        assert_eq!(seq.len(), MAX_HISTORY_MESSAGES + 1);
        let trimmed = trim_history(seq);
        // 40 kept, then the orphan tool result is repaired away.
        assert_eq!(trimmed.len(), MAX_HISTORY_MESSAGES - 1);
        assert_eq!(trimmed.first().unwrap().role, "user");
        assert_eq!(trimmed.first().unwrap().content, "q0");

        // An assistant message that kept prose but lost its tool results
        // ships as plain text without dangling `tool_calls`.
        let mut seq2 = vec![
            ChatMessage::new("user", "first"),
            ChatMessage::assistant_with_tool_calls("thinking out loud", &[ToolCallRecord {
                id: Some("call_0".into()),
                name: "lookup".into(),
                arguments: "{}".into(),
            }]),
        ];
        for i in 0..39 {
            seq2.push(ChatMessage::new("user", format!("q{i}")));
        }
        let trimmed2 = trim_history(seq2);
        assert_eq!(trimmed2.first().unwrap().role, "assistant");
        assert_eq!(trimmed2.first().unwrap().content, "thinking out loud");
        assert!(trimmed2.first().unwrap().tool_calls.is_none());
    }

    #[test]
    fn bare_json_fallback_requires_a_tool_shape() {
        assert!(extract_plain_json_tool_call(r#"{"name": "Ada"}"#).is_none());
        let call = extract_plain_json_tool_call(
            r#"{"name": "get_time", "arguments": {"tz": "utc"}}"#,
        )
        .expect("tool-shaped JSON");
        assert_eq!(call.name, "get_time");
    }

    #[test]
    fn extracts_inline_json_and_qwen_xml_tool_calls() {
        let xml = "Checking.<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_inline_tool_calls(xml);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Seoul"}"#);
        assert_eq!(strip_inline_tool_blocks(xml), "Checking.");

        let markdown = "I will check:\n```json\n{\"name\": \"calc\", \"arguments\": {\"expr\": \"2+2\"}}\n```";
        let md_calls = extract_inline_tool_calls(markdown);
        assert_eq!(md_calls.len(), 1);
        assert_eq!(md_calls[0].name, "calc");
        assert_eq!(md_calls[0].arguments, r#"{"expr":"2+2"}"#);
    }

    #[test]
    fn ingest_delta_splits_inline_think_tags_smoothly() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("Hi", Vec::new(), 0.7, 100);

        let u1 = session.ingest_delta(Some("<think>pondering"), None);
        assert_eq!(u1.len(), 1);
        assert!(u1[0].0); // is_reasoning == true
        assert_eq!(u1[0].2.body, "pondering");

        let u2 = session.ingest_delta(Some(" deep things</think>Hello!"), None);
        assert_eq!(u2.len(), 2);
        assert!(u2[0].0); // reasoning update
        assert_eq!(u2[0].2.body, "pondering deep things");
        assert!(!u2[1].0); // assistant update
        assert_eq!(u2[1].2.body, "Hello!");

        let (calls, _) = session.finish_turn();
        assert!(calls.is_empty());
        assert_eq!(session.turns().len(), 3); // User, Reasoning, Assistant
        assert_eq!(session.turns()[1].kind, KIND_REASONING);
        assert_eq!(session.turns()[2].kind, KIND_ASSISTANT);
        assert_eq!(session.turns()[2].body, "Hello!");
    }

    #[test]
    fn ingest_delta_handles_thought_and_thinking_tags() {
        let mut session = ConversationSession::new();
        session.begin_user_turn("Explain", Vec::new(), 0.7, 100);

        let u1 = session.ingest_delta(Some("<thought>analyzing input</thought>The answer is 42."), None);
        assert_eq!(u1.len(), 2);
        assert!(u1[0].0);
        assert_eq!(u1[0].2.body, "analyzing input");
        assert!(!u1[1].0);
        assert_eq!(u1[1].2.body, "The answer is 42.");
    }

    #[test]
    fn token_estimation_and_budget_trimming() {
        let text = "Hello world! How are you doing today? 12345";
        let tokens = estimate_text_tokens(text);
        assert!(tokens >= 8);

        let mut long_seq = Vec::new();
        for _i in 0..10 {
            long_seq.push(ChatMessage::new("user", "x ".repeat(2500)));
            long_seq.push(ChatMessage::new("assistant", "y ".repeat(2500)));
        }
        let trimmed = trim_history(long_seq);
        let total_tokens = estimate_messages_tokens(&trimmed);
        assert!(total_tokens <= MAX_HISTORY_TOKENS + 3000);
    }

    #[test]
    fn export_json_produces_valid_structured_data() {
        let mut session = ConversationSession::new();
        session.set_system_prompt("You are helpful.");
        session.begin_user_turn("Ping", Vec::new(), 0.7, 100);
        session.append_assistant("Pong");
        session.finish_turn();

        let json_str = session.export_json();
        let val: Value = serde_json::from_str(&json_str).expect("valid JSON export");
        assert_eq!(val["version"], "2.0.0");
        assert_eq!(val["system_prompt"], "You are helpful.");
        assert_eq!(val["total_turns"], 2);
        assert!(val["estimated_tokens"].as_u64().unwrap() > 0);
    }
}
