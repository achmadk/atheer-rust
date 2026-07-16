//! # SandboxedGpuBridge
//!
//! In-process bridge that manages the lifecycle of the GPU execution shard
//! (running in an Android IsolatedService) and batches KV pages across the
//! Binder IPC boundary.
//!
//! ## Lifecycle
//!
//! 1. `pre_warm()` — spawns the worker, passes model FD, waits for ready
//! 2. `batch()` — accumulates (token_id, position) pairs, sends as single IPC
//! 3. `shutdown()` — kills worker, releases resources
//!
//! ## Crash handling
//!
//! A `DeathRecipient` on the Binder connection detects worker death. After
//! `max_worker_crashes` within `worker_restart_window_secs`, the bridge
//! permanently falls back to CPU and fires `SandboxFallback`.

use crate::error::{AtheerCoreError, Result};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for the sandboxed GPU bridge.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Enable sandboxed execution (default: true on Android).
    pub sandbox_enabled: bool,
    /// Maximum worker crashes before permanent CPU fallback (default: 3).
    pub max_worker_crashes: u32,
    /// Sliding window in seconds for crash counting (default: 300).
    pub worker_restart_window_secs: u64,
    /// Number of KV pages to accumulate before sending a batch (default: 8).
    pub kv_page_batch_size: usize,
    /// Optional path to persist crash counts across engine sessions.
    /// A lightweight flat file is written at this path on each crash event.
    pub persistence_path: Option<PathBuf>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            sandbox_enabled: cfg!(target_os = "android"),
            max_worker_crashes: 3,
            worker_restart_window_secs: 300,
            kv_page_batch_size: 8,
            persistence_path: None,
        }
    }
}

// ─── Crash event ──────────────────────────────────────────────────────────

/// A recorded crash event with timestamp for sliding-window counting.
#[derive(Debug, Clone)]
struct CrashEvent {
    timestamp: Instant,
}

// ─── Bridge states ────────────────────────────────────────────────────────

/// Bridge operational state.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeState {
    /// Not yet initialized.
    Idle,
    /// Worker is being spawned / initialized.
    Starting,
    /// Worker is ready for inference.
    Ready,
    /// Worker process has crashed (awaiting restart or escalation).
    Crashed,
    /// Worker has permanently fallen back to CPU.
    Fallback,
}

// ─── SandboxedGpuBridge ───────────────────────────────────────────────────

/// Manages the lifecycle and IPC of the GPU execution shard worker.
pub struct SandboxedGpuBridge {
    config: SandboxConfig,
    state: BridgeState,
    crash_history: VecDeque<CrashEvent>,
    crash_count: u32,
    /// Accumulated (token_id, position) pairs ready for batch.
    pending_batch: Vec<(u32, usize)>,
}

impl SandboxedGpuBridge {
    /// Create a new bridge with the given config.
    ///
    /// If a persistence file exists from a previous session and the crash count
    /// meets or exceeds the threshold, the bridge starts in Fallback state.
    pub fn new(config: SandboxConfig) -> Self {
        // Load persisted crash count from a previous session
        let persisted_count = Self::load_persisted_crash_count(config.persistence_path.as_deref());

        let initial_state = if persisted_count >= config.max_worker_crashes {
            tracing::warn!(
                target: "atheer::sandbox::audit",
                event = "persisted_escalation",
                persisted_crash_count = persisted_count,
                max_worker_crashes = config.max_worker_crashes,
                reason = "previous_session_exceeded_threshold",
            );
            BridgeState::Fallback
        } else {
            BridgeState::Idle
        };

        Self {
            config,
            state: initial_state,
            crash_history: VecDeque::new(),
            crash_count: persisted_count,
            pending_batch: Vec::new(),
        }
    }

    /// Pre-warm the worker: spawn isolated process, pass model FD.
    ///
    /// In production this would bind to the Android IsolatedService via
    /// Binder. For now, we validate config and transition to Starting.
    pub fn pre_warm(&mut self) -> Result<()> {
        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "pre_warm_start",
            sandbox_enabled = self.config.sandbox_enabled,
        );

        if !self.config.sandbox_enabled {
            tracing::info!(
                target: "atheer::sandbox::audit",
                event = "pre_warm_fallback",
                reason = "sandbox_disabled",
            );
            self.state = BridgeState::Fallback;
            return Ok(());
        }

        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "pre_warm_starting",
        );
        self.state = BridgeState::Starting;

        // TODO: Real AIDL bind + init() call
        // For now transition directly to Ready
        self.state = BridgeState::Ready;

        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "pre_warm_ready",
        );
        Ok(())
    }

    /// Queue a (token_id, position) pair for batched inference.
    ///
    /// If the pending batch reaches `kv_page_batch_size`, the batch
    /// is automatically flushed (sent to worker).
    pub fn queue_token(&mut self, token_id: u32, position: usize) -> Result<Option<Vec<Vec<f32>>>> {
        if self.state != BridgeState::Ready {
            return Err(AtheerCoreError::GenerationFailed(
                "Sandbox not ready for inference".to_string(),
            ));
        }

        self.pending_batch.push((token_id, position));
        let batch_size = self.pending_batch.len();

        tracing::debug!(
            target: "atheer::sandbox::audit",
            event = "queue_token",
            token_id,
            position,
            batch_size,
            kv_page_batch_size = self.config.kv_page_batch_size,
        );

        if batch_size >= self.config.kv_page_batch_size {
            self.flush_batch()
        } else {
            Ok(None)
        }
    }

    /// Flush the pending batch to the worker.
    ///
    /// Returns logits for all queued tokens if successful.
    pub fn flush_batch(&mut self) -> Result<Option<Vec<Vec<f32>>>> {
        if self.pending_batch.is_empty() || self.state != BridgeState::Ready {
            return Ok(None);
        }

        let batch = std::mem::take(&mut self.pending_batch);
        let token_ids: Vec<u32> = batch.iter().map(|(t, _)| *t).collect();
        let positions: Vec<usize> = batch.iter().map(|(_, p)| *p).collect();
        let batch_len = token_ids.len();

        tracing::debug!(
            target: "atheer::sandbox::audit",
            event = "flush_batch",
            batch_size = batch_len,
        );

        // TODO: Real AIDL batch() call to worker
        // For now, simulate with CPU fallback behavior
        let logits = self.cpu_fallback_batch(&token_ids, &positions);

        tracing::debug!(
            target: "atheer::sandbox::audit",
            event = "flush_batch_result",
            logit_count = logits.len(),
            logit_dim = logits.first().map(|l| l.len()).unwrap_or(0),
        );
        Ok(Some(logits))
    }

    /// Graceful shutdown of the worker.
    pub fn shutdown(&mut self) {
        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "shutdown",
            previous_state = ?self.state,
            pending_batch_size = self.pending_batch.len(),
        );
        self.state = BridgeState::Idle;
        self.pending_batch.clear();
    }

    /// Record a worker crash. Returns true if escalation threshold is exceeded.
    pub fn record_crash(&mut self) -> bool {
        let now = Instant::now();
        let window = Duration::from_secs(self.config.worker_restart_window_secs);

        // Mark worker as crashed
        self.state = BridgeState::Crashed;

        tracing::warn!(
            target: "atheer::sandbox::audit",
            event = "crash",
            crash_count_before = self.crash_count,
            max_worker_crashes = self.config.max_worker_crashes,
            window_secs = self.config.worker_restart_window_secs,
        );

        // Prune crash history outside the window
        while let Some(front) = self.crash_history.front() {
            if now.duration_since(front.timestamp) > window {
                self.crash_history.pop_front();
            } else {
                break;
            }
        }

        self.crash_history.push_back(CrashEvent { timestamp: now });
        self.crash_count = self.crash_history.len() as u32;

        // Persist crash count to disk for cross-session compliance audit
        self.persist_crash_count();

        if self.crash_count >= self.config.max_worker_crashes {
            self.state = BridgeState::Fallback;
            tracing::error!(
                target: "atheer::sandbox::audit",
                event = "escalation",
                crash_count = self.crash_count,
                reason = "crash_threshold_exceeded",
            );
            return true; // threshold exceeded
        }

        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "crash_recovered",
            crash_count = self.crash_count,
            remaining_retries = self.config.max_worker_crashes - self.crash_count,
        );
        false
    }

    /// Whether the bridge has permanently fallen back to CPU.
    pub fn is_fallback(&self) -> bool {
        self.state == BridgeState::Fallback
    }

    /// Whether the bridge is ready for inference.
    pub fn is_ready(&self) -> bool {
        self.state == BridgeState::Ready
    }

    /// Current crash count within the sliding window.
    pub fn crash_count(&self) -> u32 {
        self.crash_count
    }

    /// Current bridge state.
    pub fn state(&self) -> BridgeState {
        self.state.clone()
    }

    /// Pending batch size.
    pub fn pending_batch_size(&self) -> usize {
        self.pending_batch.len()
    }

    /// Auto-restart after a crash (if threshold not exceeded).
    ///
    /// Returns true if the worker was successfully re-spawned.
    pub fn auto_restart(&mut self) -> bool {
        if self.is_fallback() {
            tracing::warn!(
                target: "atheer::sandbox::audit",
                event = "auto_restart_blocked",
                reason = "permanent_fallback",
            );
            return false;
        }
        if self.state != BridgeState::Crashed {
            return false;
        }

        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "auto_restart",
            crash_count = self.crash_count,
        );

        // Transition back to Starting for re-spawn
        self.state = BridgeState::Starting;

        // TODO: Real AIDL re-bind + init()
        // For now simulate success
        self.state = BridgeState::Ready;

        tracing::info!(
            target: "atheer::sandbox::audit",
            event = "auto_restart_success",
        );
        true
    }

    /// Persist the crash count to disk at `persistence_path`.
    /// Uses a lightweight text file containing the crash count as a decimal integer.
    pub fn persist_crash_count(&self) {
        let path = match self.config.persistence_path.as_ref() {
            Some(p) => p,
            None => return,
        };
        if let Err(e) = fs::write(path, format!("{}\n", self.crash_count)) {
            tracing::warn!(
                target: "atheer::sandbox::audit",
                event = "persist_failed",
                path = %path.display(),
                error = %e,
            );
        } else {
            tracing::debug!(
                target: "atheer::sandbox::audit",
                event = "persist_ok",
                path = %path.display(),
                crash_count = self.crash_count,
            );
        }
    }

    /// Load a previously persisted crash count from `path`.
    /// Returns 0 if the file does not exist or cannot be parsed.
    pub fn load_persisted_crash_count(path: Option<&std::path::Path>) -> u32 {
        let path = match path {
            Some(p) => p,
            None => return 0,
        };
        if !path.exists() {
            return 0;
        }
        match fs::read_to_string(path) {
            Ok(content) => content.trim().parse::<u32>().unwrap_or(0),
            Err(e) => {
                tracing::warn!(
                    target: "atheer::sandbox::audit",
                    event = "load_persist_failed",
                    path = %path.display(),
                    error = %e,
                );
                0
            }
        }
    }

    // ─── CPU fallback helpers ──────────────────────────────────────

    /// Simulate batch inference on CPU (one-hot at token index).
    fn cpu_fallback_batch(&self, token_ids: &[u32], _positions: &[usize]) -> Vec<Vec<f32>> {
        let vocab_size = 50257;
        token_ids
            .iter()
            .map(|&tid| {
                let mut logits = vec![0.0f32; vocab_size];
                if (tid as usize) < vocab_size {
                    logits[tid as usize] = 1.0;
                }
                logits
            })
            .collect()
    }
}

impl Drop for SandboxedGpuBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> SandboxConfig {
        SandboxConfig {
            sandbox_enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_bridge_default_config() {
        let config = SandboxConfig::default();
        assert_eq!(config.max_worker_crashes, 3);
        assert_eq!(config.worker_restart_window_secs, 300);
        assert_eq!(config.kv_page_batch_size, 8);
    }

    #[test]
    fn test_bridge_initial_state() {
        let bridge = SandboxedGpuBridge::new(enabled_config());
        assert_eq!(bridge.state(), BridgeState::Idle);
        assert!(!bridge.is_ready());
        assert!(!bridge.is_fallback());
    }

    #[test]
    fn test_pre_warm_transitions_to_ready() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        bridge.pre_warm().unwrap();
        assert!(bridge.is_ready());
    }

    #[test]
    fn test_pre_warm_sandbox_disabled_falls_back() {
        let config = SandboxConfig {
            sandbox_enabled: false,
            ..Default::default()
        };
        let mut bridge = SandboxedGpuBridge::new(config);
        bridge.pre_warm().unwrap();
        assert!(bridge.is_fallback());
    }

    #[test]
    fn test_queue_token_requires_ready() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        let result = bridge.queue_token(0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_queue_token_after_pre_warm() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        bridge.pre_warm().unwrap();
        let result = bridge.queue_token(0, 0);
        assert!(result.is_ok());
        assert_eq!(bridge.pending_batch_size(), 1);
    }

    #[test]
    fn test_batch_auto_flush_on_threshold() {
        let config = SandboxConfig {
            sandbox_enabled: true,
            kv_page_batch_size: 3,
            ..Default::default()
        };
        let mut bridge = SandboxedGpuBridge::new(config);
        bridge.pre_warm().unwrap();

        // Queue 2 tokens — no flush yet
        bridge.queue_token(0, 0).unwrap();
        bridge.queue_token(1, 1).unwrap();
        assert_eq!(bridge.pending_batch_size(), 2);

        // Queue 3rd token — triggers auto-flush
        let result = bridge.queue_token(2, 2).unwrap();
        assert!(result.is_some());
        assert_eq!(bridge.pending_batch_size(), 0);
    }

    #[test]
    fn test_flush_batch_returns_logits() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        bridge.pre_warm().unwrap();
        bridge.queue_token(10, 0).unwrap();
        bridge.queue_token(20, 1).unwrap();
        let logits = bridge.flush_batch().unwrap();
        assert!(logits.is_some());
        let logits = logits.unwrap();
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0][10], 1.0); // one-hot at token index
        assert_eq!(logits[1][20], 1.0);
    }

    #[test]
    fn test_crash_counting_and_escalation() {
        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 2,
            worker_restart_window_secs: 60, // 1 minute window
            ..Default::default()
        };
        let mut bridge = SandboxedGpuBridge::new(config);

        // First crash — below threshold
        let exceeded = bridge.record_crash();
        assert!(!exceeded);
        assert_eq!(bridge.crash_count(), 1);
        assert!(!bridge.is_fallback());

        // Second crash — hits threshold
        let exceeded = bridge.record_crash();
        assert!(exceeded);
        assert_eq!(bridge.crash_count(), 2);
        assert!(bridge.is_fallback());
    }

    #[test]
    fn test_auto_restart_after_crash() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        bridge.pre_warm().unwrap();
        bridge.record_crash();
        assert!(!bridge.is_ready());
        assert_eq!(bridge.state(), BridgeState::Crashed);
        assert!(bridge.auto_restart());
        assert!(bridge.is_ready());
    }

    #[test]
    fn test_auto_restart_blocked_on_escalation() {
        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 1,
            ..Default::default()
        };
        let mut bridge = SandboxedGpuBridge::new(config);
        bridge.pre_warm().unwrap();
        bridge.record_crash();
        assert!(bridge.is_fallback());
        assert!(!bridge.auto_restart());
    }

    #[test]
    fn test_shutdown_clears_state() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        bridge.pre_warm().unwrap();
        bridge.queue_token(0, 0).unwrap();
        bridge.shutdown();
        assert_eq!(bridge.state(), BridgeState::Idle);
        assert_eq!(bridge.pending_batch_size(), 0);
    }

    #[test]
    fn test_crash_window_prunes_old_events() {
        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 5,
            worker_restart_window_secs: 0, // zero window = immediate expiry
            ..Default::default()
        };
        let mut bridge = SandboxedGpuBridge::new(config);

        // With zero window, every crash should be immediately pruned
        bridge.record_crash();
        assert_eq!(bridge.crash_count(), 1);
        assert_eq!(bridge.state(), BridgeState::Crashed);

        bridge.record_crash();
        assert_eq!(bridge.crash_count(), 1); // previous was pruned
        assert_eq!(bridge.state(), BridgeState::Crashed);
    }

    #[test]
    fn test_drop_calls_shutdown() {
        let mut bridge = SandboxedGpuBridge::new(enabled_config());
        bridge.pre_warm().unwrap();
        bridge.queue_token(0, 0).unwrap();
        // Drop will call shutdown — no panic expected
        drop(bridge);
    }

    // ── T6 Compliance attestation tests ─────────────────────────

    /// Full compliance chain: pre-warm (probe) → batch inference → crash → auto-restart → crash escalation → Fallback
    #[test]
    fn test_compliance_full_lifecycle_chain() {
        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 2,
            worker_restart_window_secs: 300,
            kv_page_batch_size: 4,
            persistence_path: None,
        };
        let mut bridge = SandboxedGpuBridge::new(config);

        // Phase 1: Idle → verify initial state
        assert_eq!(bridge.state(), BridgeState::Idle);
        assert!(!bridge.is_ready());
        assert!(!bridge.is_fallback());
        assert_eq!(bridge.crash_count(), 0);

        // Phase 2: pre_warm (probe) → Ready
        bridge.pre_warm().unwrap();
        assert!(bridge.is_ready());
        assert_eq!(bridge.state(), BridgeState::Ready);

        // Phase 3: Batch inference — queue tokens up to flush threshold
        let r1 = bridge.queue_token(42, 0).unwrap();
        assert!(r1.is_none()); // not yet full
        let r2 = bridge.queue_token(7, 1).unwrap();
        assert!(r2.is_none());
        let r3 = bridge.queue_token(99, 2).unwrap();
        assert!(r3.is_none());
        // 4th token triggers auto-flush
        let r4 = bridge.queue_token(100, 3).unwrap();
        assert!(r4.is_some());
        let logits = r4.unwrap();
        assert_eq!(logits.len(), 4); // all 4 tokens returned
        assert_eq!(logits[0][42], 1.0); // one-hot verification
        assert_eq!(logits[2][99], 1.0);

        // Phase 4: Crash → Crashed state
        let exceeded = bridge.record_crash();
        assert!(!exceeded);
        assert_eq!(bridge.state(), BridgeState::Crashed);
        assert_eq!(bridge.crash_count(), 1);

        // Phase 5: Auto-restart → Ready again
        assert!(bridge.auto_restart());
        assert!(bridge.is_ready());
        assert_eq!(bridge.crash_count(), 1);

        // Phase 6: Second crash → escalation threshold exceeded → Fallback
        let exceeded = bridge.record_crash();
        assert!(exceeded);
        assert!(bridge.is_fallback());
        assert_eq!(bridge.crash_count(), 2);

        // Phase 7: Verify Fallback blocks further operations
        let err = bridge.queue_token(0, 0);
        assert!(err.is_err());

        // Phase 8: Shutdown from Fallback state
        bridge.shutdown();
        assert_eq!(bridge.state(), BridgeState::Idle);
    }

    #[test]
    fn test_compliance_persistence_roundtrip() {
        let dir = std::env::temp_dir();
        let persist_path = dir.join("atheer_test_crash_count.txt");

        // Clean up before test
        let _ = std::fs::remove_file(&persist_path);

        // Create bridge with persistence path, enabled sandbox, crash threshold=2
        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 2,
            worker_restart_window_secs: 60,
            kv_page_batch_size: 4,
            persistence_path: Some(persist_path.clone()),
        };
        let mut bridge = SandboxedGpuBridge::new(config);

        // Verify initial persistence file does not exist (crash_count=0)
        assert!(!persist_path.exists());

        // Pre-warm and crash once (below threshold)
        bridge.pre_warm().unwrap();
        bridge.record_crash();
        assert!(persist_path.exists());
        let content = std::fs::read_to_string(&persist_path).unwrap();
        assert_eq!(content.trim(), "1");

        // Crash again (hits threshold → persists 2)
        bridge.record_crash();
        let content = std::fs::read_to_string(&persist_path).unwrap();
        assert_eq!(content.trim(), "2");

        // Clean up
        let _ = std::fs::remove_file(&persist_path);
    }

    #[test]
    fn test_compliance_persisted_escalation_on_construction() {
        let dir = std::env::temp_dir();
        let persist_path = dir.join("atheer_test_escalation_start.txt");

        // Write a persisted crash count that meets threshold (3 >= 3)
        std::fs::write(&persist_path, "3\n").unwrap();

        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 3,
            worker_restart_window_secs: 300,
            kv_page_batch_size: 4,
            persistence_path: Some(persist_path.clone()),
        };
        let bridge = SandboxedGpuBridge::new(config);

        // Bridge should start in Fallback state due to persisted escalation
        assert!(bridge.is_fallback());
        assert_eq!(bridge.crash_count(), 3);

        // And auto_restart should be blocked
        let mut bridge = bridge;
        assert!(!bridge.auto_restart());

        // Clean up
        let _ = std::fs::remove_file(&persist_path);
    }

    #[test]
    fn test_compliance_persisted_below_threshold_starts_idle() {
        let dir = std::env::temp_dir();
        let persist_path = dir.join("atheer_test_below_threshold.txt");

        // Write a crash count below threshold (1 < 5)
        std::fs::write(&persist_path, "1\n").unwrap();

        let config = SandboxConfig {
            sandbox_enabled: true,
            max_worker_crashes: 5,
            worker_restart_window_secs: 300,
            kv_page_batch_size: 4,
            persistence_path: Some(persist_path.clone()),
        };
        let bridge = SandboxedGpuBridge::new(config);

        // Should start in Idle (not Fallback) because 1 < 5
        assert_eq!(bridge.state(), BridgeState::Idle);
        assert!(!bridge.is_fallback());
        assert_eq!(bridge.crash_count(), 1);

        // Clean up
        let _ = std::fs::remove_file(&persist_path);
    }
}
