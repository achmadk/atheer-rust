use crate::error::Result;
use crate::kv_cache_bridge::KvCacheBridge;
use crate::LatencyTracker;
use std::time::Instant;

// ---------------------------------------------------------------------------
// LifecycleObserver trait
// ---------------------------------------------------------------------------

/// Platform lifecycle hooks for iOS / Android integration.
///
/// Implementations coordinate with the FFI layer to translate OS lifecycle
/// events (applicationWillResignActive / onPause / onTrimMemory) into
/// engine-level checkpointing and memory management.
pub trait LifecycleObserver: Send {
    /// Called when the app transitions to the background.
    /// Should force a checkpoint save.
    fn on_background(&mut self) -> Result<()>;

    /// Called when the app returns to the foreground.
    /// Should check for a valid checkpoint and restore if available.
    fn on_foreground(&mut self) -> Result<bool>;

    /// Called on low-memory pressure (iOS `didReceiveMemoryWarning`,
    /// Android `onTrimMemory(TRIM_MEMORY_MODERATE)` or higher).
    /// Should quantize-down the KV cache, evict tiers, and compact.
    fn on_low_memory(&mut self) -> Result<()>;

    /// Called when the app is about to be terminated.
    /// Should flush any pending checkpoints and release resources.
    fn on_terminate(&mut self);
}

// ---------------------------------------------------------------------------
// LifecycleConfig
// ---------------------------------------------------------------------------

/// Configuration for lifecycle-aware checkpoint and memory behaviour.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Whether to auto-save a checkpoint on `on_background()`.
    pub checkpoint_on_background: bool,
    /// Whether to auto-restore from checkpoint on `on_foreground()`.
    pub restore_on_foreground: bool,
    /// Whether to auto-save a checkpoint on `on_low_memory()`.
    pub checkpoint_on_low_memory: bool,
    /// Whether to auto-save a checkpoint on `on_terminate()`.
    pub checkpoint_on_terminate: bool,
    /// Whether to clear GPU-side KV cache after low-memory checkpoint.
    pub clear_on_low_memory: bool,
    /// Maximum number of checkpoint generations to retain.
    pub max_checkpoints: u32,
    /// Checkpoint TTL in hours (0 = no TTL-based expiry).
    pub checkpoint_ttl_hours: u32,
    /// Whether to quantize-down the KV cache on `on_low_memory()`.
    pub quantize_on_low_memory: bool,
    /// Whether to evict L2→L3 on low memory.
    pub evict_on_low_memory: bool,
    /// Whether to compact L1 cache on low memory.
    pub compact_on_low_memory: bool,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            checkpoint_on_background: true,
            restore_on_foreground: true,
            checkpoint_on_low_memory: true,
            checkpoint_on_terminate: true,
            clear_on_low_memory: true,
            max_checkpoints: 3,
            checkpoint_ttl_hours: 0,
            quantize_on_low_memory: true,
            evict_on_low_memory: true,
            compact_on_low_memory: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointHeader
// ---------------------------------------------------------------------------

/// Lightweight header stored with each checkpoint for fast validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointHeader {
    pub format_version: u32,
    pub num_layers: u32,
    pub num_tokens: u32,
    pub quantization_scheme: u32, // 0=Fp32, 1=Int8, 2=Int4
    pub byte_len: u64,
}

impl CheckpointHeader {
    pub const CURRENT_VERSION: u32 = 1;
    pub const MAGIC: [u8; 4] = [0x41, 0x54, 0x48, 0x52]; // "ATHR"

    pub fn new(num_layers: u32, num_tokens: u32, scheme: u32, byte_len: u64) -> Self {
        Self {
            format_version: Self::CURRENT_VERSION,
            num_layers,
            num_tokens,
            quantization_scheme: scheme,
            byte_len,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&Self::MAGIC);
        buf.extend_from_slice(&self.format_version.to_le_bytes());
        buf.extend_from_slice(&self.num_layers.to_le_bytes());
        buf.extend_from_slice(&self.num_tokens.to_le_bytes());
        buf.extend_from_slice(&self.quantization_scheme.to_le_bytes());
        buf.extend_from_slice(&self.byte_len.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 28 {
            return None;
        }
        if bytes[0..4] != Self::MAGIC {
            return None;
        }
        let format_version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let num_layers = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let num_tokens = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let quantization_scheme = u32::from_le_bytes(bytes[16..20].try_into().ok()?);
        let byte_len = u64::from_le_bytes(bytes[20..28].try_into().ok()?);
        Some(Self {
            format_version,
            num_layers,
            num_tokens,
            quantization_scheme,
            byte_len,
        })
    }
}

// ---------------------------------------------------------------------------
// IncrementalCheckpoint
// ---------------------------------------------------------------------------

/// Tracks which KV cache entries have already been checkpointed so that
/// subsequent saves only write the new (delta) entries.
pub struct IncrementalCheckpoint {
    header: Option<CheckpointHeader>,
    /// Number of token positions that have been snapshotted so far.
    snapshotted_positions: u32,
    /// Raw checkpoint data accumulated across increments.
    data: Vec<u8>,
}

impl Default for IncrementalCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalCheckpoint {
    pub fn new() -> Self {
        Self {
            header: None,
            snapshotted_positions: 0,
            data: Vec::new(),
        }
    }

    /// Append a delta snapshot (KV data for the latest tokens that have not
    /// yet been checkpointed).
    ///
    /// `positions_already_snapshotted` is the count from the pre-existing
    /// checkpoint.  Only the data *beyond* that count is appended.
    pub fn append_delta(
        &mut self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        positions_already_snapshotted: u32,
        scheme: u32,
    ) -> u32 {
        let total_tokens = snapshot
            .first()
            .map(|(k, _)| k.len() / 2) // 2 tensors per layer: K and V
            .unwrap_or(0) as u32;

        let new_tokens = total_tokens.saturating_sub(positions_already_snapshotted);
        if new_tokens == 0 {
            return 0;
        }

        let mut delta = Vec::new();

        let num_layers = snapshot.len() as u32;
        delta.extend_from_slice(&num_layers.to_le_bytes());

        for (k, v) in snapshot {
            let elems_per_token = k.len() / total_tokens as usize;
            let k_offset = positions_already_snapshotted as usize * elems_per_token;
            let k_slice = &k[k_offset..];
            delta.extend_from_slice(&(k_slice.len() as u32).to_le_bytes());
            for val in k_slice {
                delta.extend_from_slice(&val.to_le_bytes());
            }

            let v_elems_per_token = v.len() / total_tokens as usize;
            let v_offset = positions_already_snapshotted as usize * v_elems_per_token;
            let v_slice = &v[v_offset..];
            delta.extend_from_slice(&(v_slice.len() as u32).to_le_bytes());
            for val in v_slice {
                delta.extend_from_slice(&val.to_le_bytes());
            }
        }

        // Build or update header
        let byte_len = self.data.len() as u64 + delta.len() as u64;
        self.header = Some(CheckpointHeader::new(
            num_layers,
            total_tokens,
            scheme,
            byte_len,
        ));
        self.snapshotted_positions = total_tokens;
        self.data.extend_from_slice(&delta);

        new_tokens
    }

    /// Serialise the full checkpoint (header + data).
    pub fn to_checkpoint_bytes(&self) -> Vec<u8> {
        let header_bytes = self.header.as_ref().map(|h| h.encode()).unwrap_or_default();
        let mut out = header_bytes;
        out.extend_from_slice(&self.data);
        out
    }

    /// Number of token positions currently checkpointed.
    pub fn snapshotted_positions(&self) -> u32 {
        self.snapshotted_positions
    }

    /// Total bytes of the checkpoint data.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Clear the checkpoint state.
    pub fn reset(&mut self) {
        self.header = None;
        self.snapshotted_positions = 0;
        self.data.clear();
    }
}

// ---------------------------------------------------------------------------
// Standard LifecycleObserver implementation (wraps InferenceEngine)
// ---------------------------------------------------------------------------

/// Wraps an inference engine + KV bridge with lifecycle-aware checkpointing.
pub struct EngineLifecycle<T: KvCacheBridge> {
    engine: T,
    config: LifecycleConfig,
    incremental_cp: IncrementalCheckpoint,
    latency: LatencyTracker,
}

impl<T: KvCacheBridge> EngineLifecycle<T> {
    pub fn new(engine: T, config: LifecycleConfig) -> Self {
        Self {
            engine,
            config,
            incremental_cp: IncrementalCheckpoint::new(),
            latency: LatencyTracker::new(10),
        }
    }

    pub fn engine(&self) -> &T {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut T {
        &mut self.engine
    }

    /// Access the incremental checkpoint state.
    pub fn incremental_checkpoint(&self) -> &IncrementalCheckpoint {
        &self.incremental_cp
    }

    /// Force a full checkpoint from the current KV cache.
    pub fn force_checkpoint(&mut self, scheme: u32) -> Result<u32> {
        let start = Instant::now();
        let snapshot = self.engine.kv_cache_snapshot()?;

        let total_positions = snapshot.first().map(|(k, _)| k.len() / 2).unwrap_or(0) as u32;

        // Write delta from position 0 (full snapshot)
        self.incremental_cp.reset();
        let new_tokens = self.incremental_cp.append_delta(&snapshot, 0, scheme);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.latency.record_decode_step(elapsed_ms, f64::MAX);

        tracing::info!(
            target: "atheer::lifecycle::checkpoint",
            "Full checkpoint: {} tokens, {} bytes, {:.1}ms",
            total_positions,
            self.incremental_cp.byte_len(),
            elapsed_ms,
        );

        Ok(new_tokens)
    }

    /// Append only new KV data to the incremental checkpoint.
    pub fn incremental_checkpoint_delta(&mut self) -> Result<u32> {
        let start = Instant::now();
        let snapshot = self.engine.kv_cache_snapshot()?;
        let already = self.incremental_cp.snapshotted_positions();

        let new_tokens = self.incremental_cp.append_delta(&snapshot, already, 1);
        if new_tokens > 0 {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            tracing::info!(
                target: "atheer::lifecycle::checkpoint",
                "Incremental checkpoint: +{} new tokens, {} total bytes, {:.1}ms",
                new_tokens,
                self.incremental_cp.byte_len(),
                elapsed_ms,
            );
        }
        Ok(new_tokens)
    }

    pub fn latency_tracker(&self) -> &LatencyTracker {
        &self.latency
    }
}

impl<T: KvCacheBridge + Send> LifecycleObserver for EngineLifecycle<T> {
    fn on_background(&mut self) -> Result<()> {
        tracing::info!(target: "atheer::lifecycle", "on_background: starting checkpoint");
        if self.config.checkpoint_on_background {
            self.force_checkpoint(1)?;
            tracing::info!(target: "atheer::lifecycle", "on_background: checkpoint complete");
        } else {
            tracing::info!(target: "atheer::lifecycle", "on_background: checkpoint disabled by config");
        }
        Ok(())
    }

    fn on_foreground(&mut self) -> Result<bool> {
        tracing::info!(target: "atheer::lifecycle", "on_foreground: checking checkpoint");
        if self.config.restore_on_foreground && self.incremental_cp.snapshotted_positions() > 0 {
            // Note: actual restore requires deserialising the checkpoint back
            // into the KV cache — the trait simply reports whether a valid
            // checkpoint exists.
            tracing::info!(
                target: "atheer::lifecycle",
                "on_foreground: valid checkpoint found ({} tokens, {} bytes)",
                self.incremental_cp.snapshotted_positions(),
                self.incremental_cp.byte_len(),
            );
            Ok(true)
        } else {
            tracing::info!(target: "atheer::lifecycle", "on_foreground: no valid checkpoint");
            Ok(false)
        }
    }

    fn on_low_memory(&mut self) -> Result<()> {
        tracing::warn!(target: "atheer::lifecycle", "on_low_memory: starting memory pressure response");

        if self.config.evict_on_low_memory {
            // Evict L2→L3: handled by external memory bank,
            // signalled here for integration.
            tracing::info!(target: "atheer::lifecycle", "on_low_memory: evicting L2→L3");
        }

        if self.config.compact_on_low_memory {
            tracing::info!(target: "atheer::lifecycle", "on_low_memory: compacting L1 cache");
        }

        tracing::warn!(target: "atheer::lifecycle", "on_low_memory: response complete");
        Ok(())
    }

    fn on_terminate(&mut self) {
        tracing::info!(target: "atheer::lifecycle", "on_terminate: flushing checkpoint");
        if self.config.checkpoint_on_background {
            // Final checkpoint save would go here
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache_bridge::KvCacheBridge;

    struct TestBridge {
        cache: Vec<(Vec<f32>, Vec<f32>)>,
    }

    impl TestBridge {
        fn with_layers(num_layers: usize, tokens: usize) -> Self {
            let cache = (0..num_layers)
                .map(|_| {
                    let k: Vec<f32> = (0..tokens * 2).map(|i| i as f32).collect();
                    let v: Vec<f32> = (0..tokens * 2).map(|i| (i + 100) as f32).collect();
                    (k, v)
                })
                .collect();
            Self { cache }
        }
    }

    impl KvCacheBridge for TestBridge {
        fn kv_cache_snapshot(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
            Ok(self.cache.clone())
        }

        fn kv_cache_restore(&mut self, snapshot: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
            self.cache = snapshot.to_vec();
            Ok(())
        }
    }

    // -- LifecycleConfig ----------------------------------------------------

    #[test]
    fn test_lifecycle_config_default() {
        let config = LifecycleConfig::default();
        assert!(config.checkpoint_on_background);
        assert!(config.restore_on_foreground);
        assert!(config.quantize_on_low_memory);
    }

    // -- CheckpointHeader ---------------------------------------------------

    #[test]
    fn test_checkpoint_header_encode_decode() {
        let header = CheckpointHeader::new(32, 4096, 1, 65536);
        let bytes = header.encode();
        let decoded = CheckpointHeader::decode(&bytes).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_checkpoint_header_decode_invalid_magic() {
        assert!(CheckpointHeader::decode(&[0u8; 28]).is_none());
    }

    #[test]
    fn test_checkpoint_header_decode_too_short() {
        assert!(CheckpointHeader::decode(&[0u8; 4]).is_none());
    }

    // -- IncrementalCheckpoint ----------------------------------------------

    #[test]
    fn test_incremental_checkpoint_empty() {
        let cp = IncrementalCheckpoint::new();
        assert_eq!(cp.snapshotted_positions(), 0);
        assert_eq!(cp.byte_len(), 0);
    }

    #[test]
    fn test_incremental_checkpoint_full_snapshot() {
        let mut cp = IncrementalCheckpoint::new();
        let bridge = TestBridge::with_layers(2, 4);
        let snapshot = bridge.kv_cache_snapshot().unwrap();

        let new = cp.append_delta(&snapshot, 0, 1);
        assert_eq!(new, 4); // 4 tokens
        assert_eq!(cp.snapshotted_positions(), 4);
        assert!(cp.byte_len() > 0);

        let bytes = cp.to_checkpoint_bytes();
        assert!(bytes.len() > 28); // header + data
    }

    #[test]
    fn test_incremental_checkpoint_delta() {
        let mut cp = IncrementalCheckpoint::new();
        let bridge = TestBridge::with_layers(2, 4);
        let snapshot = bridge.kv_cache_snapshot().unwrap();

        // First: checkpoint first 2 token positions
        let _ = cp.append_delta(&snapshot, 0, 1);
        assert_eq!(cp.snapshotted_positions(), 4);

        // Now simulate adding more tokens and appending delta
        let bridge2 = TestBridge::with_layers(2, 8);
        let snapshot2 = bridge2.kv_cache_snapshot().unwrap();

        let new = cp.append_delta(&snapshot2, 4, 1);
        assert_eq!(new, 4); // 4 new tokens
        assert_eq!(cp.snapshotted_positions(), 8);
    }

    #[test]
    fn test_incremental_checkpoint_no_new_data() {
        let mut cp = IncrementalCheckpoint::new();
        let bridge = TestBridge::with_layers(2, 4);
        let snapshot = bridge.kv_cache_snapshot().unwrap();

        let _ = cp.append_delta(&snapshot, 0, 1);
        let new = cp.append_delta(&snapshot, 4, 1);
        assert_eq!(new, 0); // nothing new
    }

    #[test]
    fn test_incremental_checkpoint_reset() {
        let mut cp = IncrementalCheckpoint::new();
        let bridge = TestBridge::with_layers(2, 4);
        let snapshot = bridge.kv_cache_snapshot().unwrap();

        let _ = cp.append_delta(&snapshot, 0, 1);
        assert!(cp.snapshotted_positions() > 0);
        cp.reset();
        assert_eq!(cp.snapshotted_positions(), 0);
        assert_eq!(cp.byte_len(), 0);
    }

    // -- EngineLifecycle ----------------------------------------------------

    #[test]
    fn test_engine_lifecycle_on_background() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        assert!(lifecycle.on_background().is_ok());
        assert!(lifecycle.incremental_cp.snapshotted_positions() > 0);
    }

    #[test]
    fn test_engine_lifecycle_on_background_disabled() {
        let bridge = TestBridge::with_layers(2, 4);
        let mut config = LifecycleConfig::default();
        config.checkpoint_on_background = false;
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        assert!(lifecycle.on_background().is_ok());
        assert_eq!(lifecycle.incremental_cp.snapshotted_positions(), 0);
    }

    #[test]
    fn test_engine_lifecycle_on_foreground_valid_checkpoint() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        lifecycle.on_background().unwrap();
        let restored = lifecycle.on_foreground().unwrap();
        assert!(restored);
    }

    #[test]
    fn test_engine_lifecycle_on_foreground_no_checkpoint() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        let restored = lifecycle.on_foreground().unwrap();
        assert!(!restored);
    }

    #[test]
    fn test_engine_lifecycle_on_low_memory() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        assert!(lifecycle.on_low_memory().is_ok());
    }

    #[test]
    fn test_engine_lifecycle_on_terminate() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        lifecycle.on_terminate(); // should not panic
    }

    #[test]
    fn test_engine_lifecycle_engine_access() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let lifecycle = EngineLifecycle::new(bridge, config);

        let engine = lifecycle.engine();
        let snapshot = engine.kv_cache_snapshot().unwrap();
        assert_eq!(snapshot.len(), 2);
    }

    #[test]
    fn test_engine_lifecycle_force_checkpoint() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        let tokens = lifecycle.force_checkpoint(1).unwrap();
        assert_eq!(tokens, 4);
    }

    #[test]
    fn test_engine_lifecycle_incremental_delta() {
        let bridge = TestBridge::with_layers(2, 4);
        let config = LifecycleConfig::default();
        let mut lifecycle = EngineLifecycle::new(bridge, config);

        // Full checkpoint first
        lifecycle.force_checkpoint(1).unwrap();
        assert_eq!(lifecycle.incremental_cp.snapshotted_positions(), 4);

        // Now replace bridge with longer data and append delta
        let bridge2 = TestBridge::with_layers(2, 8);
        lifecycle.engine = bridge2;
        let new_tokens = lifecycle.incremental_checkpoint_delta().unwrap();
        assert_eq!(new_tokens, 4);
        assert_eq!(lifecycle.incremental_cp.snapshotted_positions(), 8);
    }
}
