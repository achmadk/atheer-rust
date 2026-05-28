use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    pub available_mb: u64,
    pub total_mb: u64,
    pub low_memory_threshold_mb: u64,
}

impl MemoryStatus {
    pub fn new(available_mb: u64, total_mb: u64) -> Self {
        Self {
            available_mb,
            total_mb,
            low_memory_threshold_mb: 800,
        }
    }

    pub fn is_low(&self) -> bool {
        self.available_mb < self.low_memory_threshold_mb
    }

    pub fn usage_percent(&self) -> f32 {
        if self.total_mb == 0 {
            return 0.0;
        }
        let used = self.total_mb - self.available_mb;
        (used as f32 / self.total_mb as f32) * 100.0
    }
}

impl Default for MemoryStatus {
    fn default() -> Self {
        Self {
            available_mb: 0,
            total_mb: 0,
            low_memory_threshold_mb: 800,
        }
    }
}
