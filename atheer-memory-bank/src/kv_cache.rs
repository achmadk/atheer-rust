use std::collections::{HashMap, VecDeque};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    Priority,
}

pub struct KvCacheEntry {
    pub position: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub token_id: u32,
    pub access_count: usize,
    pub last_access: Instant,
    pub priority: u8,
}

pub struct KvCache {
    entries: HashMap<usize, VecDeque<KvCacheEntry>>,
    access_order: VecDeque<(usize, usize)>,
    eviction_policy: EvictionPolicy,
    max_seq_len: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            eviction_policy: EvictionPolicy::Lru,
            max_seq_len,
            num_layers,
            num_heads,
            head_dim,
        }
    }

    pub fn with_eviction_policy(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        policy: EvictionPolicy,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            eviction_policy: policy,
            max_seq_len,
            num_layers,
            num_heads,
            head_dim,
        }
    }

    pub fn set_eviction_policy(&mut self, policy: EvictionPolicy) {
        self.eviction_policy = policy;
    }

    pub fn max_memory_bytes(&self) -> usize {
        let per_entry = self.num_heads * self.head_dim * 2 * std::mem::size_of::<f32>();
        self.max_seq_len * self.num_layers * per_entry
    }

    pub fn current_memory_bytes(&self) -> usize {
        let entry_size = self.num_heads * self.head_dim * 2 * std::mem::size_of::<f32>();
        let total_entries: usize = self.entries.values().map(|v| v.len()).sum();
        total_entries * entry_size
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn memory_pressure(&self) -> f32 {
        let current = self.current_memory_bytes() as f32;
        let max = self.max_memory_bytes() as f32;
        if max == 0.0 {
            0.0
        } else {
            current / max
        }
    }

    pub fn insert(
        &mut self,
        layer: usize,
        position: usize,
        token_id: u32,
        keys: Vec<f32>,
        values: Vec<f32>,
    ) {
        self.evict_if_needed();

        let entry = KvCacheEntry {
            position,
            keys,
            values,
            token_id,
            access_count: 1,
            last_access: Instant::now(),
            priority: 0,
        };

        let entries = self.entries.entry(layer).or_default();
        entries.push_back(entry);
        self.access_order.push_back((layer, position));
    }

    pub fn get(&mut self, layer: usize, position: usize) -> Option<&KvCacheEntry> {
        if let Some(entries) = self.entries.get_mut(&layer) {
            if let Some(entry) = entries.iter_mut().find(|e| e.position == position) {
                entry.access_count += 1;
                entry.last_access = Instant::now();
                return Some(entry);
            }
        }
        None
    }

    pub fn get_layer(&self, layer: usize) -> Option<&VecDeque<KvCacheEntry>> {
        self.entries.get(&layer)
    }

    pub fn get_positions(&self, layer: usize, start: usize, end: usize) -> Vec<&KvCacheEntry> {
        self.entries
            .get(&layer)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| e.position >= start && e.position < end)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn evict_if_needed(&mut self) {
        while self.is_full() {
            self.evict_one();
        }
    }

    pub fn evict_one(&mut self) {
        let victim = self.select_victim();
        if let Some((layer, position)) = victim {
            self.remove_entry(layer, position);
        }
    }

    pub fn remove_entry(&mut self, layer: usize, position: usize) {
        if let Some(entries) = self.entries.get_mut(&layer) {
            entries.retain(|e| e.position != position);
        }
        self.access_order.retain(|(l, p)| !(*l == layer && *p == position));
    }

    fn select_victim(&self) -> Option<(usize, usize)> {
        match self.eviction_policy {
            EvictionPolicy::Lru => self.select_lru_victim(),
            EvictionPolicy::Lfu => self.select_lfu_victim(),
            EvictionPolicy::Priority => self.select_priority_victim(),
        }
    }

    fn select_lru_victim(&self) -> Option<(usize, usize)> {
        let mut oldest: Option<(usize, usize, Instant)> = None;

        for (layer, entries) in &self.entries {
            for entry in entries {
                match &oldest {
                    None => oldest = Some((*layer, entry.position, entry.last_access)),
                    Some((_, _, oldest_time)) => {
                        if entry.last_access < *oldest_time {
                            oldest = Some((*layer, entry.position, entry.last_access));
                        }
                    }
                }
            }
        }

        oldest.map(|(l, p, _)| (l, p))
    }

    fn select_lfu_victim(&self) -> Option<(usize, usize)> {
        let mut least_used: Option<(usize, usize, usize)> = None;

        for (layer, entries) in &self.entries {
            for entry in entries {
                match &least_used {
                    None => least_used = Some((*layer, entry.position, entry.access_count)),
                    Some((_, _, min_count)) => {
                        if entry.access_count < *min_count {
                            least_used = Some((*layer, entry.position, entry.access_count));
                        }
                    }
                }
            }
        }

        least_used.map(|(l, p, _)| (l, p))
    }

    fn select_priority_victim(&self) -> Option<(usize, usize)> {
        let mut lowest_priority: Option<(usize, usize, u8)> = None;

        for (layer, entries) in &self.entries {
            for entry in entries {
                match &lowest_priority {
                    None => lowest_priority = Some((*layer, entry.position, entry.priority)),
                    Some((_, _, min_prio)) => {
                        if entry.priority < *min_prio {
                            lowest_priority = Some((*layer, entry.position, entry.priority));
                        }
                    }
                }
            }
        }

        lowest_priority.map(|(l, p, _)| (l, p))
    }

    pub fn set_priority(&mut self, layer: usize, position: usize, priority: u8) {
        if let Some(entries) = self.entries.get_mut(&layer) {
            if let Some(entry) = entries.iter_mut().find(|e| e.position == position) {
                entry.priority = priority;
            }
        }
    }

    pub fn demote_low_priority(&mut self, threshold: u8) {
        let victims: Vec<_> = self
            .entries
            .iter()
            .flat_map(|(layer, entries)| {
                entries
                    .iter()
                    .filter(|e| e.priority < threshold)
                    .map(|e| (*layer, e.position))
                    .collect::<Vec<_>>()
            })
            .collect();

        for (layer, position) in victims {
            if let Some(entries) = self.entries.get_mut(&layer) {
                entries.retain(|e| e.position != position);
            }
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    pub fn truncate(&mut self, keep_positions: usize) {
        for entries in self.entries.values_mut() {
            entries.retain(|e| e.position >= keep_positions);
        }
        self.access_order.retain(|(_, pos)| *pos >= keep_positions);
    }

    pub fn token_count(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    pub fn is_full(&self) -> bool {
        self.token_count() >= self.max_seq_len * self.num_layers
    }

    pub fn is_near_full(&self, threshold: f32) -> bool {
        self.memory_pressure() >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_insert_get() {
        let mut cache = KvCache::new(2, 4, 64, 128);
        cache.insert(0, 0, 42, vec![1.0; 256], vec![2.0; 256]);

        let entry = cache.get(0, 0);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().token_id, 42);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = KvCache::with_eviction_policy(1, 1, 64, 3, EvictionPolicy::Lru);

        cache.insert(0, 0, 0, vec![1.0; 64], vec![1.0; 64]);
        cache.insert(0, 1, 1, vec![1.0; 64], vec![1.0; 64]);
        cache.insert(0, 2, 2, vec![1.0; 64], vec![1.0; 64]);

        assert_eq!(cache.token_count(), 3);

        cache.insert(0, 3, 3, vec![1.0; 64], vec![1.0; 64]);

        assert_eq!(cache.token_count(), 3);
        assert!(cache.get(0, 0).is_none());
        assert!(cache.get(0, 1).is_some());
    }

    #[test]
    fn test_lfu_eviction() {
        let mut cache = KvCache::with_eviction_policy(1, 1, 64, 3, EvictionPolicy::Lfu);

        cache.insert(0, 0, 0, vec![1.0; 64], vec![1.0; 64]);
        cache.insert(0, 1, 1, vec![1.0; 64], vec![1.0; 64]);
        cache.insert(0, 2, 2, vec![1.0; 64], vec![1.0; 64]);

        cache.get(0, 0);
        cache.get(0, 0);
        cache.get(0, 1);

        cache.insert(0, 3, 3, vec![1.0; 64], vec![1.0; 64]);

        assert!(cache.get(0, 2).is_none());
    }

    #[test]
    fn test_priority_eviction() {
        let mut cache = KvCache::with_eviction_policy(1, 1, 64, 3, EvictionPolicy::Priority);

        cache.insert(0, 0, 0, vec![1.0; 64], vec![1.0; 64]);
        cache.insert(0, 1, 1, vec![1.0; 64], vec![1.0; 64]);
        cache.insert(0, 2, 2, vec![1.0; 64], vec![1.0; 64]);

        cache.set_priority(0, 0, 1);
        cache.set_priority(0, 1, 2);
        cache.set_priority(0, 2, 3);

        cache.insert(0, 3, 3, vec![1.0; 64], vec![1.0; 64]);

        assert!(cache.get(0, 0).is_none());
    }

    #[test]
    fn test_memory_pressure() {
        let cache = KvCache::new(32, 8, 128, 2048);
        let max_bytes = cache.max_memory_bytes();
        assert!(max_bytes > 0);
        assert_eq!(cache.memory_pressure(), 0.0);
    }

    #[test]
    fn test_demote_low_priority() {
        let mut cache = KvCache::new(1, 1, 64, 10);

        for i in 0..5 {
            cache.insert(0, i, i as u32, vec![1.0; 64], vec![1.0; 64]);
            cache.set_priority(0, i, i as u8);
        }

        cache.demote_low_priority(3);

        assert!(cache.get(0, 0).is_none());
        assert!(cache.get(0, 1).is_none());
        assert!(cache.get(0, 2).is_none());
        assert!(cache.get(0, 3).is_some());
        assert!(cache.get(0, 4).is_some());
    }
}
