use std::time::Instant;

use crate::kv_cache::KvCache;

pub struct L1ActiveCache {
    pub model_id: String,
    pub token_count: usize,
    pub last_update: Instant,
    kv_cache: Option<KvCache>,
}

impl L1ActiveCache {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            token_count: 0,
            last_update: Instant::now(),
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

    pub fn update(&mut self, token_ids: &[u32]) {
        self.token_count += token_ids.len();
        self.last_update = Instant::now();
    }

    pub fn is_stale(&self) -> bool {
        self.last_update.elapsed().as_secs() > 300
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_kv_cache_integration() {
        let mut cache = L1ActiveCache::new("test".to_string()).with_kv_cache(2, 4, 64, 128);

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
    fn test_l1_clear_kv_cache() {
        let mut cache = L1ActiveCache::new("test".to_string()).with_kv_cache(2, 4, 64, 128);

        if let Some(kv) = cache.kv_cache_mut() {
            kv.insert(0, 0, 42, vec![1.0; 256], vec![1.0; 256]);
        }

        cache.clear_kv_cache();

        assert!(cache.kv_cache_mut().unwrap().get(0, 0).is_none());
    }
}
