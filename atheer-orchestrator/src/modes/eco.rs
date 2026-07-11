use std::collections::{HashMap, VecDeque};

pub struct NGramCache {
    ngrams: HashMap<Vec<u32>, Vec<u32>>,
    max_order: usize,
    max_entries: usize,
    access_order: VecDeque<Vec<u32>>,
}

impl NGramCache {
    pub fn new(max_order: usize, max_entries: usize) -> Self {
        Self {
            ngrams: HashMap::new(),
            max_order,
            max_entries,
            access_order: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, prefix: &[u32], continuation: &[u32]) {
        if prefix.len() > self.max_order || continuation.is_empty() {
            return;
        }

        let prefix_key = prefix.to_vec();

        self.ngrams
            .entry(prefix_key.clone())
            .or_default()
            .extend(continuation.iter().take(10));

        if !self.access_order.contains(&prefix_key) {
            self.access_order.push_back(prefix_key);
            if self.access_order.len() > self.max_entries {
                if let Some(oldest) = self.access_order.pop_front() {
                    self.ngrams.remove(&oldest);
                }
            }
        }
    }

    pub fn get(&self, prefix: &[u32]) -> Option<&Vec<u32>> {
        self.ngrams.get(prefix)
    }

    pub fn lookup(&self, prefix: &[u32]) -> Option<u32> {
        self.ngrams.get(prefix)?.first().copied()
    }

    pub fn clear(&mut self) {
        self.ngrams.clear();
        self.access_order.clear();
    }

    pub fn size(&self) -> usize {
        self.ngrams.len()
    }
}

#[allow(dead_code)]
pub struct EcoMode {
    ngram_enabled: bool,
    ngram_cache: NGramCache,
    ngram_order: usize,
    power_saving_enabled: bool,
}

impl EcoMode {
    pub fn new() -> Self {
        Self {
            ngram_enabled: true,
            ngram_cache: NGramCache::new(3, 1000),
            ngram_order: 3,
            power_saving_enabled: true,
        }
    }

    pub fn ngram_enabled(&self) -> bool {
        self.ngram_enabled
    }

    pub fn set_ngram_enabled(&mut self, enabled: bool) {
        self.ngram_enabled = enabled;
    }

    pub fn ngram_cache(&self) -> &NGramCache {
        &self.ngram_cache
    }

    pub fn ngram_cache_mut(&mut self) -> &mut NGramCache {
        &mut self.ngram_cache
    }

    pub fn set_ngram_order(&mut self, order: usize) {
        self.ngram_order = order.clamp(2, 5);
    }

    pub fn ngram_order(&self) -> usize {
        self.ngram_order
    }

    pub fn train_on_sequence(&mut self, tokens: &[u32]) {
        if tokens.len() < self.ngram_order + 1 {
            return;
        }

        for window in tokens.windows(self.ngram_order + 1) {
            let prefix = &window[..self.ngram_order];
            let continuation = &window[self.ngram_order..];
            self.ngram_cache.insert(prefix, continuation);
        }
    }

    pub fn predict(&self, prefix: &[u32]) -> Option<u32> {
        if !self.ngram_enabled {
            return None;
        }

        let start = prefix.len().saturating_sub(self.ngram_order);
        let suffix = &prefix[start..];
        self.ngram_cache.lookup(suffix)
    }

    pub fn cache_stats(&self) -> (usize, usize) {
        (self.ngram_cache.size(), self.ngram_order)
    }
}

impl Default for EcoMode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eco_mode_defaults() {
        let mode = EcoMode::new();
        assert!(mode.ngram_enabled());
        assert_eq!(mode.ngram_order(), 3);
    }

    #[test]
    fn test_ngram_cache_insert() {
        let mut cache = NGramCache::new(3, 100);
        cache.insert(&[1, 2, 3], &[4, 5]);

        assert!(cache.get(&[1, 2, 3]).is_some());
        assert_eq!(cache.get(&[1, 2, 3]).unwrap(), &[4, 5]);
    }

    #[test]
    fn test_ngram_cache_lookup() {
        let mut cache = NGramCache::new(3, 100);
        cache.insert(&[1, 2, 3], &[4, 5, 6]);

        assert_eq!(cache.lookup(&[1, 2, 3]), Some(4));
    }

    #[test]
    fn test_ngram_cache_eviction() {
        let mut cache = NGramCache::new(3, 3);
        cache.insert(&[1, 2, 3], &[4]);
        cache.insert(&[2, 3, 4], &[5]);
        cache.insert(&[3, 4, 5], &[6]);
        cache.insert(&[4, 5, 6], &[7]);

        assert!(cache.get(&[1, 2, 3]).is_none());
        assert!(cache.get(&[4, 5, 6]).is_some());
    }

    #[test]
    fn test_eco_train_on_sequence() {
        let mut mode = EcoMode::new();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7];
        mode.train_on_sequence(&tokens);

        let (size, _) = mode.cache_stats();
        assert!(size > 0);
    }

    #[test]
    fn test_eco_predict() {
        let mut mode = EcoMode::new();
        mode.train_on_sequence(&[1, 2, 3, 4, 5]);

        let prediction = mode.predict(&[2, 3, 4]);
        assert_eq!(prediction, Some(5));
    }
}
