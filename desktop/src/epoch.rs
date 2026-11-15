//! Monotonic operation generations ("epochs") used to discard stale async work.
//!
//! Every user-visible operation that can finish on a worker thread (load, unload,
//! MCP connect, chat stream, image render) starts by taking a fresh token from the
//! gate. When the worker posts its completion back to the UI thread, the controller
//! compares the token it was created with against the gate's current token. A token
//! from an older generation is dropped instead of being applied, so a slow
//! `Loaded` can never resurrect a backend the user already switched away from, and a
//! late `Delta`/`Done` from a stopped stream can never overwrite newer state.

/// A token identifying one generation of an asynchronous operation.
///
/// `Epoch::default()` (generation 0) is reserved for "no operation was started",
/// which keeps a freshly constructed controller from accepting events that were
/// never issued by it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(u64);

impl Epoch {
    /// The generation that exists before any operation starts.
    pub const NONE: Epoch = Epoch(0);

    pub fn value(self) -> u64 {
        self.0
    }

    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for Epoch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "epoch#{}", self.0)
    }
}

/// Hands out [`Epoch`] tokens and answers "is this completion still current?".
#[derive(Debug, Default)]
pub struct EpochGate {
    current: Epoch,
}

impl EpochGate {
    pub fn new() -> Self {
        Self {
            current: Epoch::NONE,
        }
    }

    /// Starts a new operation, invalidating every token handed out before it.
    pub fn begin(&mut self) -> Epoch {
        self.current = Epoch(self.current.0.wrapping_add(1).max(1));
        self.current
    }

    /// Invalidates the current generation without starting an operation.
    ///
    /// Used when the user changes something (backend switch, new conversation) that
    /// makes every in-flight completion meaningless.
    pub fn invalidate(&mut self) -> Epoch {
        self.begin()
    }

    /// The generation currently allowed to mutate state.
    pub fn current(&self) -> Epoch {
        self.current
    }

    /// True when `token` still belongs to the current generation.
    pub fn is_current(&self, token: Epoch) -> bool {
        !token.is_none() && token == self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_increase_and_only_the_newest_is_current() {
        let mut gate = EpochGate::new();
        assert!(gate.current().is_none());

        let load = gate.begin();
        assert!(load.value() >= 1);
        assert!(gate.is_current(load));

        let switch = gate.invalidate();
        assert!(switch > load);
        // A `Loaded` that finished after the user switched backends is stale.
        assert!(!gate.is_current(load));
        assert!(gate.is_current(switch));

        let connect = gate.begin();
        assert!(!gate.is_current(switch));
        assert!(gate.is_current(connect));
    }

    #[test]
    fn a_token_from_before_any_operation_is_never_current() {
        let gate = EpochGate::new();
        // Defensive: an event constructed without a token must not be applied even
        // before the first `begin()`.
        assert!(!gate.is_current(Epoch::NONE));
        assert!(!gate.is_current(Epoch(7)));
    }

    #[test]
    fn late_completion_of_an_older_load_cannot_beat_a_newer_one() {
        // Mirrors the real bug: two `Load / connect` clicks race; the first finishes last.
        let mut gate = EpochGate::new();
        let first_load = gate.begin();
        let second_load = gate.begin();

        // The second load finishes first and is applied.
        assert!(gate.is_current(second_load));
        // The first load finishing late must be dropped, not overwrite the second.
        assert!(!gate.is_current(first_load));
    }
}
