use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum HandoffPhase {
    #[default]
    Idle,
    BridgeMode,
    AlignmentCheck,
    RampingUp,
    FullSpeed,
}

impl HandoffPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            HandoffPhase::Idle => "idle",
            HandoffPhase::BridgeMode => "bridge",
            HandoffPhase::AlignmentCheck => "checking",
            HandoffPhase::RampingUp => "ramping",
            HandoffPhase::FullSpeed => "synced",
        }
    }
}

#[allow(dead_code)]
pub struct HandoffProtocol {
    phase: HandoffPhase,
    bridge_tokens_remaining: usize,
    ramp_depth: usize,
    max_depth: usize,
    pending_tokens: Vec<u32>,
}

impl Default for HandoffProtocol {
    fn default() -> Self {
        Self {
            phase: HandoffPhase::Idle,
            bridge_tokens_remaining: 3,
            ramp_depth: 1,
            max_depth: 4,
            pending_tokens: Vec::new(),
        }
    }
}

impl HandoffProtocol {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn phase(&self) -> HandoffPhase {
        self.phase
    }

    pub fn trigger_handoff(&mut self, _new_model: &str) {
        self.phase = HandoffPhase::BridgeMode;
        self.bridge_tokens_remaining = 3;
        self.ramp_depth = 1;
    }

    pub fn tick(&mut self) {
        match self.phase {
            HandoffPhase::BridgeMode => {
                if self.bridge_tokens_remaining > 0 {
                    self.bridge_tokens_remaining -= 1;
                }
                if self.bridge_tokens_remaining == 0 {
                    self.phase = HandoffPhase::AlignmentCheck;
                }
            }
            HandoffPhase::AlignmentCheck => {
                self.phase = HandoffPhase::RampingUp;
            }
            HandoffPhase::RampingUp => {
                if self.ramp_depth < self.max_depth {
                    self.ramp_depth *= 2;
                }
                if self.ramp_depth >= self.max_depth {
                    self.phase = HandoffPhase::FullSpeed;
                }
            }
            _ => {}
        }
    }

    pub fn speculation_depth(&self) -> usize {
        match self.phase {
            HandoffPhase::RampingUp => self.ramp_depth,
            HandoffPhase::FullSpeed => self.max_depth,
            _ => 0,
        }
    }

    pub fn is_handoff_in_progress(&self) -> bool {
        self.phase != HandoffPhase::Idle && self.phase != HandoffPhase::FullSpeed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handoff_phases() {
        let mut protocol = HandoffProtocol::new();
        assert_eq!(protocol.phase(), HandoffPhase::Idle);

        protocol.trigger_handoff("new-model");
        assert_eq!(protocol.phase(), HandoffPhase::BridgeMode);

        for _ in 0..3 {
            protocol.tick();
        }
        assert_eq!(protocol.phase(), HandoffPhase::AlignmentCheck);

        protocol.tick();
        assert_eq!(protocol.phase(), HandoffPhase::RampingUp);
    }

    #[test]
    fn test_speculation_depth_ramp_up() {
        let mut protocol = HandoffProtocol::new();
        protocol.trigger_handoff("new-model");

        for _ in 0..4 {
            protocol.tick();
        }

        assert_eq!(protocol.speculation_depth(), 1);

        protocol.tick();
        assert!(protocol.speculation_depth() >= 1);
    }
}
