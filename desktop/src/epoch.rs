//! Monotonic epoch tokens that let the UI controller discard stale async
//! completions. A worker thread captures the epoch at dispatch time; when the
//! user switches engine, starts a new chat, or a new load supersedes an old
//! one, the epoch advances and every late event from the superseded run is
//! dropped instead of overwriting newer state.

#[derive(Default)]
pub struct EpochGate {
    epoch: u64,
}

impl EpochGate {
    /// Reserves a new generation and returns its token. `0` means "no run
    /// reserved", so `+= 1` on a wrapping boundary is guarded by `max(1)`.
    pub fn begin(&mut self) -> u64 {
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.epoch
    }

    /// True when `epoch` is the currently reserved generation.
    pub fn is_current(&self, epoch: u64) -> bool {
        epoch != 0 && epoch == self.epoch
    }

    pub fn current(&self) -> u64 {
        self.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_increase_and_zero_is_never_current() {
        let mut gate = EpochGate::default();
        assert!(!gate.is_current(0));
        let first = gate.begin();
        assert_ne!(first, 0);
        assert!(gate.is_current(first));
        let second = gate.begin();
        assert_ne!(first, second);
        assert!(!gate.is_current(first));
        assert!(gate.is_current(second));
    }

    #[test]
    fn epoch_never_wraps_to_zero() {
        let mut gate = EpochGate {
            epoch: u64::MAX,
        };
        let next = gate.begin();
        assert_ne!(next, 0, "wrapping must skip generation 0");
        assert!(gate.is_current(next));
    }
}
