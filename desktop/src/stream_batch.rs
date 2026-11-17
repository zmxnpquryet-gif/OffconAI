//! Coalesces token-by-token streaming deltas into ~40 ms UI batches.
//!
//! The backend listener receives one `StreamEvent::Delta` per SSE chunk (often one per
//! token). Repainting the Slint transcript for every token forces a full text re-layout
//! of the active turn. This module accumulates chunks and hands them to the UI at most
//! once per interval, while guaranteeing that a terminating event (tool call, error,
//! done) can force an immediate flush so the transcript never loses or reorders text.
//!
//! The clock is injected (`advance(dt)`), which keeps the behaviour deterministic in
//! tests and independent of wall-clock jitter in production.

use std::time::Duration;

/// One coalesced update: the text and/or reasoning characters collected since the
/// previous flush, in arrival order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaBatch {
    pub text: Option<String>,
    pub reasoning: Option<String>,
}

impl DeltaBatch {
    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.reasoning.is_none()
    }

    /// Number of buffered characters, used by tests to assert nothing was dropped.
    pub fn len(&self) -> usize {
        self.text.as_deref().map(str::len).unwrap_or(0)
            + self.reasoning.as_deref().map(str::len).unwrap_or(0)
    }
}

/// Buffers streaming chunks and drains them on a fixed cadence.
#[derive(Debug)]
pub struct DeltaBatcher {
    interval: Duration,
    text: String,
    reasoning: String,
    since_flush: Duration,
    chunks: u64,
    flushes: u64,
    chars: u64,
}

impl Default for DeltaBatcher {
    fn default() -> Self {
        Self::new(Duration::from_millis(40))
    }
}

impl DeltaBatcher {
    /// Default batch window. 25 Hz is comfortably inside the 33–50 ms budget: fast
    /// enough to read as "live typing", slow enough that Slint lays the turn out about
    /// once per frame instead of once per token.
    pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(40);

    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            text: String::new(),
            reasoning: String::new(),
            since_flush: Duration::ZERO,
            chunks: 0,
            flushes: 0,
            chars: 0,
        }
    }

    /// Appends a streaming chunk. Empty chunks are ignored so an empty delta never
    /// schedules a repaint.
    pub fn push(&mut self, text: Option<&str>, reasoning: Option<&str>) {
        let mut pushed = false;
        if let Some(chunk) = text.filter(|chunk| !chunk.is_empty()) {
            self.text.push_str(chunk);
            self.chars += chunk.chars().count() as u64;
            pushed = true;
        }
        if let Some(chunk) = reasoning.filter(|chunk| !chunk.is_empty()) {
            self.reasoning.push_str(chunk);
            self.chars += chunk.chars().count() as u64;
            pushed = true;
        }
        if pushed {
            self.chunks += 1;
        }
    }

    /// Advances the batch clock by the elapsed UI tick.
    pub fn advance(&mut self, elapsed: Duration) {
        self.since_flush = self.since_flush.saturating_add(elapsed);
    }

    /// True when there is buffered text waiting to be shown.
    pub fn is_pending(&self) -> bool {
        !self.text.is_empty() || !self.reasoning.is_empty()
    }

    /// True when buffered text should be shown now.
    pub fn is_due(&self) -> bool {
        self.is_pending() && self.since_flush >= self.interval
    }

    /// Takes the buffered batch, if any, and restarts the batch clock.
    ///
    /// This is also the "forced flush" path used before tool calls, errors and done
    /// events: it never blocks, and it never splits a multi-byte character because
    /// chunks are only ever concatenated as whole `&str` slices.
    pub fn drain(&mut self) -> Option<DeltaBatch> {
        if !self.is_pending() {
            return None;
        }
        let batch = DeltaBatch {
            text: (!self.text.is_empty()).then(|| std::mem::take(&mut self.text)),
            reasoning: (!self.reasoning.is_empty()).then(|| std::mem::take(&mut self.reasoning)),
        };
        self.since_flush = Duration::ZERO;
        self.flushes += 1;
        Some(batch)
    }

    /// Drops buffered text without showing it (used when a run is abandoned).
    pub fn clear(&mut self) {
        self.text.clear();
        self.reasoning.clear();
        self.since_flush = Duration::ZERO;
    }

    /// `(pushed_chunks, emitted_batches, characters)` counters for diagnostics and tests.
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.chunks, self.flushes, self.chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn coalesces_many_token_chunks_into_one_batch() {
        let mut batcher = DeltaBatcher::new(ms(40));
        for chunk in ["The ", "KV ", "cache ", "saves ", "RAM."] {
            batcher.push(Some(chunk), None);
        }
        batcher.advance(ms(39));
        assert!(batcher.is_pending());
        assert!(!batcher.is_due(), "must not flush before the window elapses");

        batcher.advance(ms(1));
        assert!(batcher.is_due());
        let batch = batcher.drain().expect("batch is due");
        assert_eq!(batch.text.as_deref(), Some("The KV cache saves RAM."));
        assert_eq!(batch.reasoning, None);
        assert!(!batcher.is_pending());
        assert_eq!(batcher.stats().0, 5);
        assert_eq!(batcher.stats().1, 1);
    }

    #[test]
    fn keeps_reasoning_and_answer_streams_separate_and_ordered() {
        let mut batcher = DeltaBatcher::new(ms(40));
        batcher.push(Some("answer-1 "), Some("think-1 "));
        batcher.push(Some("answer-2"), Some("think-2"));
        let batch = batcher.drain().expect("forced drain");
        assert_eq!(batch.text.as_deref(), Some("answer-1 answer-2"));
        assert_eq!(batch.reasoning.as_deref(), Some("think-1 think-2"));

        // A second tool round must not inherit the previous round's buffers.
        batcher.push(None, Some("round-2-think"));
        let batch = batcher.drain().expect("forced drain");
        assert_eq!(batch.text, None);
        assert_eq!(batch.reasoning.as_deref(), Some("round-2-think"));
    }

    #[test]
    fn empty_chunks_never_schedule_a_repaint() {
        let mut batcher = DeltaBatcher::new(ms(40));
        batcher.push(Some(""), Some(""));
        batcher.push(None, None);
        assert!(!batcher.is_pending());
        assert!(batcher.drain().is_none());
        assert_eq!(batcher.stats().0, 0);
    }

    #[test]
    fn forced_flush_is_immediate_and_resets_the_window() {
        let mut batcher = DeltaBatcher::new(ms(40));
        batcher.push(Some("partial"), None);
        // A tool call / error / done arrives before the window elapsed.
        let batch = batcher.drain().expect("forced flush keeps text");
        assert_eq!(batch.text.as_deref(), Some("partial"));

        batcher.push(Some(" next"), None);
        batcher.advance(ms(10));
        assert!(!batcher.is_due(), "the window restarts after a forced flush");
        batcher.advance(ms(30));
        assert!(batcher.is_due());
    }

    #[test]
    fn batching_reduces_repaints_by_an_order_of_magnitude_without_losing_text() {
        // 1,000 one-character tokens arriving every 10 ms, flushed on a 40 ms cadence.
        let mut batcher = DeltaBatcher::new(ms(40));
        let mut shown = String::new();
        let mut flushed = 0u64;
        for _ in 0..1000 {
            batcher.push(Some("x"), None);
            batcher.advance(ms(10));
            if batcher.is_due() {
                let batch = batcher.drain().expect("due");
                shown.push_str(batch.text.as_deref().unwrap_or(""));
                flushed += 1;
            }
        }
        if let Some(batch) = batcher.drain() {
            shown.push_str(batch.text.as_deref().unwrap_or(""));
            flushed += 1;
        }

        assert_eq!(shown.len(), 1000, "no character was dropped or reordered");
        assert!(shown.bytes().all(|byte| byte == b'x'));
        // One repaint per ~40 ms of stream instead of one per token.
        assert_eq!(flushed, 250);
        assert!(flushed * 4 <= 1000);

        let batch = DeltaBatcher::new(ms(40));
        assert_eq!(batch.interval, DeltaBatcher::DEFAULT_INTERVAL);
    }

    #[test]
    fn clear_drops_buffered_text_without_emitting() {
        let mut batcher = DeltaBatcher::new(ms(40));
        batcher.push(Some("abandoned"), Some("reasoning"));
        batcher.clear();
        assert!(!batcher.is_pending());
        assert!(batcher.drain().is_none());
        assert_eq!(batcher.stats().1, 0);
    }

    #[test]
    fn batched_flushes_reconstruct_the_exact_transcript_across_tool_rounds() {
        // Mirrors the controller loop: deltas land in the batcher, the UI tick drains it
        // into the conversation session. The transcript must be identical to the token
        // stream, and the reasoning/answer split must survive a tool round.
        use crate::contracts::ToolCallRecord;
        use crate::session::ConversationSession;

        let mut batcher = DeltaBatcher::new(ms(40));
        let mut session = ConversationSession::new();
        session.begin_user_turn("describe the cache", Vec::new(), 0.7, 128);

        let reasoning: Vec<&str> = vec!["KV ", "cache ", "stores ", "keys."];
        let answer: Vec<&str> = vec!["The ", "KV ", "cache ", "saves ", "memory."];

        // Interleaved stream: reasoning first, then the answer, one token every 10 ms.
        for (index, chunk) in reasoning.iter().chain(answer.iter()).enumerate() {
            let is_reasoning = index < reasoning.len();
            batcher.push(
                (!is_reasoning).then_some(*chunk),
                is_reasoning.then_some(*chunk),
            );
            batcher.advance(ms(10));
            if batcher.is_due() {
                if let Some(batch) = batcher.drain() {
                    if let Some(text) = batch.reasoning {
                        session.append_reasoning(&text);
                    }
                    if let Some(text) = batch.text {
                        session.append_assistant(&text);
                    }
                }
            }
        }
        // Terminal event: forced flush before `finish_turn`.
        if let Some(batch) = batcher.drain() {
            if let Some(text) = batch.reasoning {
                session.append_reasoning(&text);
            }
            if let Some(text) = batch.text {
                session.append_assistant(&text);
            }
        }
        let (calls, _) = session.finish_turn();
        assert!(calls.is_empty());

        let bodies: Vec<&str> = session
            .turns()
            .iter()
            .map(|turn| turn.body.as_str())
            .collect();
        assert_eq!(bodies[1], "KV cache stores keys.");
        assert_eq!(bodies[2], "The KV cache saves memory.");
        assert_eq!(session.messages()[1].content, "The KV cache saves memory.");

        // Second tool round: the batcher starts empty, so round 1 text never leaks in.
        session.push_tool_call(ToolCallRecord {
            id: Some("call_1".into()),
            name: "lookup".into(),
            arguments: "{}".into(),
        });
        session.finish_turn();
        let _ = session.resolve_tool(0, "ok", true);
        session.prepare_continuation_round();
        batcher.push(Some("Round two."), None);
        if let Some(batch) = batcher.drain() {
            session.append_assistant(batch.text.as_deref().unwrap_or(""));
        }
        session.finish_turn();
        let bodies: Vec<&str> = session
            .turns()
            .iter()
            .map(|turn| turn.body.as_str())
            .collect();
        assert_eq!(bodies.last().copied(), Some("Round two."));
        assert!(session.integrity().is_clean());
    }
}
