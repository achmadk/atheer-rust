use crate::{HandoffProtocol, L1ActiveCache, L2WarmCache, L3CompressedStorage, Result};
use parking_lot::RwLock;
use std::sync::Arc;

#[allow(dead_code)]
pub struct MemoryBank {
    l1: Arc<RwLock<Option<L1ActiveCache>>>,
    l2: Arc<RwLock<Option<L2WarmCache>>>,
    l3: Arc<RwLock<Option<L3CompressedStorage>>>,
    handoff: Arc<RwLock<HandoffProtocol>>,
    max_size_mb: usize,
}

impl MemoryBank {
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            l1: Arc::new(RwLock::new(None)),
            l2: Arc::new(RwLock::new(None)),
            l3: Arc::new(RwLock::new(None)),
            handoff: Arc::new(RwLock::new(HandoffProtocol::new())),
            max_size_mb,
        }
    }

    pub fn load_l1(&self, model_id: &str) -> Result<()> {
        let mut cache = self.l1.write();
        *cache = Some(L1ActiveCache::new(model_id.to_string()));
        Ok(())
    }

    pub fn load_l2(&self, model_id: &str) -> Result<()> {
        let mut cache = self.l2.write();
        *cache = Some(L2WarmCache::new(model_id.to_string()));
        Ok(())
    }

    pub fn l1_active(&self) -> Option<String> {
        self.l1.read().as_ref().map(|c| c.model_id.clone())
    }

    pub fn l2_warm(&self) -> Option<String> {
        self.l2.read().as_ref().map(|c| c.model_id.clone())
    }

    pub fn trigger_handoff(&self, new_model_id: &str) {
        let mut protocol = self.handoff.write();
        protocol.trigger_handoff(new_model_id);
    }

    pub fn handoff_phase(&self) -> crate::handoff::HandoffPhase {
        self.handoff.read().phase()
    }

    pub fn alignment_score(&self) -> f32 {
        self.l2
            .read()
            .as_ref()
            .map(|c| c.alignment_score())
            .unwrap_or(0.0)
    }

    pub fn is_ready_for_promotion(&self) -> bool {
        self.l2
            .read()
            .as_ref()
            .map(|c| c.is_ready())
            .unwrap_or(false)
    }

    // ── KvCacheBridge integration (Group 2) ────────────────────────────────

    /// Initialize the L1 active cache with a KV cache of the given dimensions.
    /// Must be called after `load_l1()`.
    pub fn initialize_kv_cache(
        &self,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        if let Some(ref mut l1) = *self.l1.write() {
            l1.set_kv_cache(num_layers, num_heads, head_dim, max_seq_len);
        }
    }

    /// Auto-create an L2 `KvCache` if not already present.
    fn ensure_l2_kv_cache(
        &self,
        num_layers: usize,
        n_kv_head: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        let mut l2 = self.l2.write();
        if let Some(ref mut l2_cache) = *l2 {
            if l2_cache.kv_cache().is_none() {
                l2_cache.set_kv_cache(num_layers, n_kv_head, head_dim, max_seq_len);
            }
        }
    }

    /// Load a bridge snapshot into the L2 warm cache.
    ///
    /// `snapshot` is the per-layer flat `(keys, values)` format produced by
    /// [`kv_cache_snapshot`] (the KvCacheBridge trait).  The method splits the
    /// flat data into per-position entries and stores them in L2's internal
    /// `KvCache` starting at `start_position`.
    pub fn update_from_snapshot(
        &self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        n_kv_head: usize,
        head_dim: usize,
        start_position: usize,
    ) {
        // Compute a reasonable max_seq_len from the snapshot (round up to
        // next power-of-two or 64, whichever is larger).
        let total_tokens: usize = snapshot
            .iter()
            .map(|(k, _)| k.len() / (n_kv_head * head_dim).max(1))
            .max()
            .unwrap_or(0);
        let max_seq_len = (total_tokens + start_position).max(64).next_power_of_two();
        self.ensure_l2_kv_cache(snapshot.len(), n_kv_head, head_dim, max_seq_len);

        if let Some(ref mut l2) = *self.l2.write() {
            l2.load_from_snapshot(snapshot, n_kv_head, head_dim, start_position);
            l2.update_alignment(1.0); // freshly loaded → fully aligned
        }
    }

    /// Promote L1 state into L2.
    ///
    /// Takes a bridge snapshot of L1 and stores it in L2, resetting the
    /// alignment score to 1.0.
    pub fn promote_to_l2(
        &self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        n_kv_head: usize,
        head_dim: usize,
    ) {
        let total_tokens: usize = snapshot
            .iter()
            .map(|(k, _)| k.len() / (n_kv_head * head_dim).max(1))
            .max()
            .unwrap_or(0);
        let max_seq_len = total_tokens.max(64).next_power_of_two();
        self.ensure_l2_kv_cache(snapshot.len(), n_kv_head, head_dim, max_seq_len);

        let mut l2 = self.l2.write();
        if let Some(ref mut l2_cache) = *l2 {
            l2_cache.load_from_snapshot(snapshot, n_kv_head, head_dim, 0);
            l2_cache.update_alignment(1.0);
        }
    }

    /// Freeze the L2 warm cache into L3 compressed storage.
    ///
    /// Serialises the L2 `KvCache` as a bridge snapshot, compresses it, and
    /// stores it via `L3CompressedStorage::snapshot`.  Returns the snapshot ID.
    pub fn freeze_to_l3(&self, model_id: &str, total_layers: usize) -> crate::Result<String> {
        let snapshot = {
            let l2 = self.l2.read();
            match l2.as_ref() {
                Some(cache) => cache.extract_snapshot(total_layers),
                None => return Err(crate::error::MemoryBankError::CacheEmpty),
            }
        };
        let json = serde_json::to_vec(&snapshot)?;
        let mut l3 = self.l3.write();
        match l3.as_mut() {
            Some(storage) => {
                let snap_id = storage.snapshot(model_id, &json)?;
                Ok(snap_id)
            }
            None => Err(crate::error::MemoryBankError::StorageNotInitialized),
        }
    }

    /// Thaw a previously frozen L3 snapshot back into L2.
    ///
    /// Restores the compressed snapshot, deserialises it, and loads it into
    /// L2's `KvCache`.  Also re-creates the L2 cache with the given dimensions
    /// if it already exists.
    pub fn thaw_from_l3(
        &self,
        snapshot_id: &str,
        n_kv_head: usize,
        head_dim: usize,
        num_layers: usize,
        max_seq_len: usize,
    ) -> crate::Result<()> {
        let compressed = {
            let l3 = self.l3.read();
            match l3.as_ref() {
                Some(storage) => storage.restore(snapshot_id)?,
                None => return Err(crate::error::MemoryBankError::StorageNotInitialized),
            }
        };
        let snapshot: Vec<(Vec<f32>, Vec<f32>)> = serde_json::from_slice(&compressed)?;
        let mut l2 = self.l2.write();
        let l2_cache = l2.get_or_insert_with(|| L2WarmCache::new("thawed".to_string()));
        l2_cache.set_kv_cache(num_layers, n_kv_head, head_dim, max_seq_len);
        l2_cache.load_from_snapshot(&snapshot, n_kv_head, head_dim, 0);
        l2_cache.update_alignment(1.0);
        Ok(())
    }

    /// Wire the handoff: on handoff trigger, capture the current L1 snapshot
    /// into L2 so the new model can use it.
    pub fn handoff_save_l1_to_l2(
        &self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        n_kv_head: usize,
        head_dim: usize,
    ) {
        self.promote_to_l2(snapshot, n_kv_head, head_dim);
    }

    /// Wire the handoff: after alignment check passes, restore L2 back
    /// into the inference engine (via a returned snapshot).
    pub fn handoff_restore_l2(&self, total_layers: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let l2 = self.l2.read();
        match l2.as_ref() {
            Some(cache) => cache.extract_snapshot(total_layers),
            None => vec![(vec![], vec![]); total_layers],
        }
    }

    // ── VRAM pressure monitoring (Group 4) ────────────────────────────────

    /// Estimated VRAM (bytes) consumed by the active L1 KV cache, based on
    /// the number of cached entries and the per-entry tensor size.
    pub fn l1_vram_bytes(&self) -> usize {
        self.l1
            .read()
            .as_ref()
            .and_then(|c| c.kv_cache())
            .map(|kv| kv.current_memory_bytes())
            .unwrap_or(0)
    }

    /// Estimated VRAM (bytes) consumed by the L2 warm cache.
    pub fn l2_vram_bytes(&self) -> usize {
        self.l2
            .read()
            .as_ref()
            .and_then(|c| c.kv_cache())
            .map(|kv| kv.current_memory_bytes())
            .unwrap_or(0)
    }

    /// Total memory bank VRAM usage (L1 + L2) in bytes.
    pub fn total_vram_bytes(&self) -> usize {
        self.l1_vram_bytes() + self.l2_vram_bytes()
    }

    /// Total L3 compressed storage size in bytes.
    pub fn l3_size_bytes(&self) -> usize {
        self.l3.read().as_ref().map(|s| s.size_bytes()).unwrap_or(0)
    }

    /// Total allocated memory across all tiers (L1 + L2 + L3) in bytes.
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_vram_bytes() + self.l3_size_bytes()
    }

    /// VRAM pressure ratio (0.0–1.0+) relative to `max_size_mb`.
    /// Values > 1.0 indicate the cache exceeds its budget.
    pub fn vram_pressure_ratio(&self) -> f32 {
        let max_bytes = self.max_size_mb as f64 * 1_048_576.0;
        if max_bytes <= 0.0 {
            return 1.0;
        }
        (self.total_vram_bytes() as f64 / max_bytes) as f32
    }

    /// Automatically demote L1 data to L2 when `vram_pressure_ratio` exceeds
    /// the given threshold.  Returns `true` if data was demoted.
    ///
    /// This consumes the L1 snapshot and writes it into L2, then clears L1.
    /// Call this periodically in the inference loop when VRAM pressure is high.
    pub fn demote_l1_to_l2_on_pressure(
        &self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        n_kv_head: usize,
        head_dim: usize,
        threshold: f32,
    ) -> bool {
        if self.vram_pressure_ratio() < threshold {
            return false;
        }
        // Only demote if there's actual data in the snapshot
        let has_data = snapshot.iter().any(|(k, v)| !k.is_empty() || !v.is_empty());
        if !has_data {
            return false;
        }
        self.promote_to_l2(snapshot, n_kv_head, head_dim);
        // Clear L1's internal kv_cache to free memory
        if let Some(ref mut l1) = *self.l1.write() {
            l1.clear_kv_cache();
        }
        true
    }

    /// Returns true if memory pressure exceeds threshold (80% of max).
    pub fn has_memory_pressure(&self) -> bool {
        self.vram_pressure_ratio() > 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bank_creation() {
        let bank = MemoryBank::new(512);
        assert!(bank.l1_active().is_none());
        assert!(bank.l2_warm().is_none());
    }

    #[test]
    fn test_load_l1() {
        let bank = MemoryBank::new(512);
        bank.load_l1("test-model").unwrap();
        assert_eq!(bank.l1_active(), Some("test-model".to_string()));
    }

    #[test]
    fn test_alignment_score() {
        let bank = MemoryBank::new(512);
        assert_eq!(bank.alignment_score(), 0.0);
    }

    // ── Bridge integration tests ───────────────────────────────────────────

    #[test]
    fn test_initialize_kv_cache_sets_up_l1() {
        let bank = MemoryBank::new(512);
        bank.load_l1("test").unwrap();
        bank.initialize_kv_cache(4, 8, 128, 512);

        let l1 = bank.l1.read();
        let l1 = l1.as_ref().unwrap();
        assert!(l1.kv_cache().is_some());
        assert_eq!(l1.kv_memory_bytes(), 4 * 512 * 8 * 128 * 2 * 4);
    }

    #[test]
    fn test_promote_to_l2_stores_snapshot() {
        let bank = MemoryBank::new(512);
        bank.load_l1("l1-model").unwrap();
        bank.initialize_kv_cache(2, 2, 4, 64);
        bank.load_l2("l2-model").unwrap();

        // Simulate a bridge snapshot: 2 layers, 3 positions each
        let n_kv_head = 2usize;
        let head_dim = 4usize;
        let kv_elem_stride = n_kv_head * head_dim;
        let mut snapshot = Vec::new();
        for layer in 0..2 {
            let offset = (layer * kv_elem_stride * 3) as f32;
            let k: Vec<f32> = (0..kv_elem_stride * 3).map(|i| offset + i as f32).collect();
            let v: Vec<f32> = (0..kv_elem_stride * 3)
                .map(|i| offset + 100.0 + i as f32)
                .collect();
            snapshot.push((k, v));
        }

        bank.promote_to_l2(&snapshot, n_kv_head, head_dim);

        assert!(bank.is_ready_for_promotion());
        assert_eq!(bank.alignment_score(), 1.0);

        // Verify we can extract a matching snapshot back
        let extracted = bank.handoff_restore_l2(2);
        assert_eq!(extracted.len(), 2);
        for (layer_idx, (k, v)) in extracted.iter().enumerate() {
            assert_eq!(k.len(), kv_elem_stride * 3);
            assert_eq!(v.len(), kv_elem_stride * 3);
            // Check data integrity — first element is layer-specific
            let expected_offset = (layer_idx * kv_elem_stride * 3) as f32;
            assert!((k[0] - expected_offset).abs() < f32::EPSILON);
            assert!((v[0] - (expected_offset + 100.0)).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_update_from_snapshot_with_offset() {
        let bank = MemoryBank::new(512);
        bank.load_l2("l2-model").unwrap();

        let n_kv_head = 1usize;
        let head_dim = 2usize;
        // 1 layer, 2 positions
        let snapshot = vec![(
            vec![10.0, 20.0, 30.0, 40.0], // keys: 2 positions × (1*2) = 4 elements
            vec![50.0, 60.0, 70.0, 80.0], // values
        )];

        // Start at position 5
        bank.update_from_snapshot(&snapshot, n_kv_head, head_dim, 5);

        let l2 = bank.l2.read();
        let l2 = l2.as_ref().unwrap();
        let kv = l2.kv_cache().unwrap();
        let layer0 = kv.get_layer(0).unwrap();
        assert_eq!(layer0.len(), 2);

        let first = layer0.iter().find(|e| e.position == 5).unwrap();
        assert_eq!(first.keys, vec![10.0, 20.0]);
        assert_eq!(first.values, vec![50.0, 60.0]);

        let second = layer0.iter().find(|e| e.position == 6).unwrap();
        assert_eq!(second.keys, vec![30.0, 40.0]);
    }

    #[test]
    fn test_handoff_save_restore_roundtrip() {
        let bank = MemoryBank::new(512);
        bank.load_l1("l1-model").unwrap();
        bank.initialize_kv_cache(1, 1, 4, 64);
        bank.load_l2("l2-model").unwrap();

        let snapshot = vec![(
            vec![1.0, 2.0, 3.0, 4.0], // keys: 1 pos × (1*4) = 4
            vec![5.0, 6.0, 7.0, 8.0], // values
        )];

        bank.handoff_save_l1_to_l2(&snapshot, 1, 4);

        let restored = bank.handoff_restore_l2(1);
        assert_eq!(restored[0].0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(restored[0].1, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_freeze_thaw_l3_roundtrip() {
        let bank = MemoryBank::new(512);
        bank.load_l2("l2-model").unwrap();

        // Set up L2 with snapshot data
        let n_kv_head = 1usize;
        let head_dim = 2usize;
        let snapshot = vec![(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])];
        bank.update_from_snapshot(&snapshot, n_kv_head, head_dim, 0);

        // Set up L3 storage
        let temp_dir = std::env::temp_dir().join("aether_test_l3_wiring");
        let l3 = L3CompressedStorage::new(temp_dir.clone()).unwrap();
        *bank.l3.write() = Some(l3);

        // Freeze
        let snap_id = bank.freeze_to_l3("test-model", 1).unwrap();
        assert!(!snap_id.is_empty());

        // Thaw into a fresh L2
        *bank.l2.write() = Some(L2WarmCache::new("thawed".to_string()));
        bank.thaw_from_l3(&snap_id, n_kv_head, head_dim, 1, 64)
            .unwrap();

        // Verify
        let restored = bank.handoff_restore_l2(1);
        assert_eq!(restored[0].0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(restored[0].1, vec![5.0, 6.0, 7.0, 8.0]);

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_empty_l2_returns_empty_snapshot() {
        let bank = MemoryBank::new(512);
        bank.load_l2("l2-model").unwrap();

        let extracted = bank.handoff_restore_l2(3);
        assert_eq!(extracted.len(), 3);
        for (k, v) in &extracted {
            assert!(k.is_empty());
            assert!(v.is_empty());
        }
    }

    #[test]
    fn test_freeze_fails_without_l3() {
        let bank = MemoryBank::new(512);
        bank.load_l2("l2-model").unwrap();
        bank.update_from_snapshot(&[(vec![1.0], vec![2.0])], 1, 1, 0);

        let result = bank.freeze_to_l3("test", 1);
        assert!(result.is_err());
    }

    // ── VRAM monitoring tests ─────────────────────────────────────────────

    #[test]
    fn test_vram_bytes_returns_zero_when_empty() {
        let bank = MemoryBank::new(512);
        assert_eq!(bank.l1_vram_bytes(), 0);
        assert_eq!(bank.l2_vram_bytes(), 0);
        assert_eq!(bank.total_vram_bytes(), 0);
    }

    #[test]
    fn test_vram_pressure_ratio() {
        let bank = MemoryBank::new(1); // 1 MB max → pressure will be high
        assert_eq!(bank.vram_pressure_ratio(), 0.0);

        bank.load_l1("test").unwrap();
        bank.initialize_kv_cache(1, 1, 64, 1024);
        // Insert a small amount of data
        {
            let mut l1 = bank.l1.write();
            let l1 = l1.as_mut().unwrap();
            if let Some(kv) = l1.kv_cache_mut() {
                kv.insert(0, 0, 1, vec![1.0; 64], vec![2.0; 64]);
            }
        }
        let vram = bank.l1_vram_bytes();
        assert!(vram > 0);
        let ratio = bank.vram_pressure_ratio();
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_demote_l1_to_l2_on_high_pressure() {
        // Use a minimal budget (1 byte) so any KV data triggers pressure.
        // The method clamps to 0, then vram_pressure_ratio returns 1.0.
        let bank = MemoryBank::new(0);
        bank.load_l1("l1").unwrap();
        bank.initialize_kv_cache(1, 1, 4, 32);
        bank.load_l2("l2").unwrap();

        let snapshot = vec![(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])];
        let demoted = bank.demote_l1_to_l2_on_pressure(&snapshot, 1, 4, 0.1);
        assert!(demoted);

        // L1 should be cleared
        let l1_vram = bank.l1_vram_bytes();
        assert_eq!(l1_vram, 0);

        // L2 should have the data
        let l2 = bank.l2.read();
        let l2 = l2.as_ref().unwrap();
        let kv = l2.kv_cache().unwrap();
        assert_eq!(kv.token_count(), 1);
    }

    #[test]
    fn test_demote_skipped_when_pressure_below_threshold() {
        let bank = MemoryBank::new(1024); // huge budget → no pressure
        bank.load_l1("l1").unwrap();
        bank.initialize_kv_cache(1, 1, 4, 32);
        bank.load_l2("l2").unwrap();

        let snapshot = vec![(vec![1.0; 4], vec![2.0; 4])];
        let demoted = bank.demote_l1_to_l2_on_pressure(&snapshot, 1, 4, 0.9);
        assert!(!demoted);
    }

    #[test]
    fn test_demote_skipped_with_empty_snapshot() {
        let bank = MemoryBank::new(1);
        bank.load_l1("l1").unwrap();
        bank.load_l2("l2").unwrap();

        let snapshot = vec![(vec![], vec![])];
        let demoted = bank.demote_l1_to_l2_on_pressure(&snapshot, 1, 4, 0.0);
        assert!(!demoted);
    }
}
