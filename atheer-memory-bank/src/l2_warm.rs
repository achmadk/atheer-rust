use std::sync::atomic::{AtomicU32, Ordering};

use crate::kv_cache::KvCache;

pub struct L2WarmCache {
    pub model_id: String,
    pub alignment_score: f32,
    pub sync_count: AtomicU32,
    kv_cache: Option<KvCache>,
}

impl L2WarmCache {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            alignment_score: 0.0,
            sync_count: AtomicU32::new(0),
            kv_cache: None,
        }
    }

    pub fn set_kv_cache(
        &mut self,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        self.kv_cache = Some(KvCache::new(num_layers, num_heads, head_dim, max_seq_len));
    }

    pub fn with_kv_cache(
        mut self,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        self.kv_cache = Some(KvCache::new(num_layers, num_heads, head_dim, max_seq_len));
        self
    }

    pub fn update_alignment(&mut self, score: f32) {
        self.alignment_score = score.clamp(0.0, 1.0);
    }

    pub fn alignment_score(&self) -> f32 {
        self.alignment_score
    }

    pub fn is_ready(&self) -> bool {
        self.alignment_score >= 0.4
    }

    pub fn increment_sync(&self) {
        self.sync_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn sync_count(&self) -> u32 {
        self.sync_count.load(Ordering::Relaxed)
    }

    pub fn kv_cache(&self) -> Option<&KvCache> {
        self.kv_cache.as_ref()
    }

    pub fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        self.kv_cache.as_mut()
    }

    pub fn clear_kv_cache(&mut self) {
        if let Some(kv) = &mut self.kv_cache {
            kv.clear();
        }
    }

    pub fn kv_memory_bytes(&self) -> usize {
        self.kv_cache
            .as_ref()
            .map(|kv| kv.max_memory_bytes())
            .unwrap_or(0)
    }

    // ── Bridge snapshot integration ─────────────────────────────────────────

    /// Load a bridge snapshot (per-layer flat key/value vectors) into this
    /// cache's internal `KvCache`.
    ///
    /// `snapshot` must have exactly one element per layer.  Each element is
    /// `(keys, values)` where both vecs have length `n_kv_head × seq_len × head_dim`.
    /// The function splits them into per-position entries and inserts them
    /// sequentially starting from `start_position`.
    pub fn load_from_snapshot(
        &mut self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        n_kv_head: usize,
        head_dim: usize,
        start_position: usize,
    ) {
        let kv = match self.kv_cache.as_mut() {
            Some(kv) => kv,
            None => return,
        };
        let kv_elem_stride = n_kv_head * head_dim;
        for (layer_idx, (k_data, v_data)) in snapshot.iter().enumerate() {
            if k_data.is_empty() || v_data.is_empty() {
                continue;
            }
            let seq_len = k_data.len() / kv_elem_stride;
            for pos in 0..seq_len {
                let offset = pos * kv_elem_stride;
                let k_chunk = k_data[offset..offset + kv_elem_stride].to_vec();
                let v_chunk = v_data[offset..offset + kv_elem_stride].to_vec();
                kv.insert(layer_idx, start_position + pos, 0, k_chunk, v_chunk);
            }
        }
    }

    /// Extract the internal `KvCache` as a bridge snapshot.
    ///
    /// Returns one `(keys, values)` per layer, with all positions
    /// concatenated into flat f32 vectors suitable for
    /// [`kv_cache_restore`](...).
    pub fn extract_snapshot(&self, total_layers: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let kv = match self.kv_cache.as_ref() {
            Some(kv) => kv,
            None => return vec![(vec![], vec![]); total_layers],
        };
        let mut result = Vec::with_capacity(total_layers);
        for layer in 0..total_layers {
            let entries = kv.get_layer(layer);
            let layer_snapshot = match entries {
                None => (vec![], vec![]),
                Some(entries) => {
                    let mut k_all = Vec::new();
                    let mut v_all = Vec::new();
                    // Collect and sort by position for deterministic output
                    let mut sorted: Vec<_> = entries.iter().collect();
                    sorted.sort_by_key(|e| e.position);
                    for entry in sorted {
                        k_all.extend_from_slice(&entry.keys);
                        v_all.extend_from_slice(&entry.values);
                    }
                    (k_all, v_all)
                }
            };
            result.push(layer_snapshot);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_kv_cache_integration() {
        let mut cache = L2WarmCache::new("test".to_string()).with_kv_cache(2, 4, 64, 128);

        assert!(cache.kv_cache().is_some());
        assert_eq!(cache.kv_memory_bytes(), 2 * 128 * 4 * 64 * 2 * 4);

        if let Some(kv) = cache.kv_cache_mut() {
            let keys = vec![1.0; 256];
            let values = vec![1.0; 256];
            kv.insert(0, 0, 42, keys, values);
            assert!(kv.get(0, 0).is_some());
        }
    }

    #[test]
    fn test_l2_clear_kv_cache() {
        let mut cache = L2WarmCache::new("test".to_string()).with_kv_cache(2, 4, 64, 128);

        if let Some(kv) = cache.kv_cache_mut() {
            kv.insert(0, 0, 42, vec![1.0; 256], vec![1.0; 256]);
        }

        cache.clear_kv_cache();

        assert!(cache.kv_cache_mut().unwrap().get(0, 0).is_none());
    }

    #[test]
    fn test_alignment_bounds() {
        let mut cache = L2WarmCache::new("test".to_string());
        cache.update_alignment(1.5);
        assert_eq!(cache.alignment_score(), 1.0);

        cache.update_alignment(-0.5);
        assert_eq!(cache.alignment_score(), 0.0);
    }

    #[test]
    fn test_ready_threshold() {
        let mut cache = L2WarmCache::new("test".to_string());
        cache.update_alignment(0.3);
        assert!(!cache.is_ready());

        cache.update_alignment(0.5);
        assert!(cache.is_ready());
    }
}
