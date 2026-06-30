use crate::bench_runner::BenchResults;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub timestamp: i64,
    pub backend: String,
    pub device_name: String,
    pub batch_results: Vec<BatchResult>,
    pub sustained_sample_count: usize,
    pub sustained_duration_secs: u64,
    pub avg_tokens_per_second: f32,
    pub peak_tokens_per_second: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub batch_size: usize,
    pub tokens_generated: u32,
    pub duration_ms: u64,
    pub tokens_per_second: f32,
    pub thermal_state: String,
    pub available_ram_mb: u64,
    pub battery_level: u32,
}

impl BenchReport {
    pub fn from_results(results: &BenchResults) -> Self {
        let mut avg_tps = 0.0f32;
        let mut peak_tps = 0.0f32;
        let batch_results: Vec<BatchResult> = results
            .results
            .iter()
            .map(|r| {
                if r.tokens_per_second > peak_tps {
                    peak_tps = r.tokens_per_second;
                }
                avg_tps += r.tokens_per_second;
                BatchResult {
                    batch_size: r.batch_size,
                    tokens_generated: r.tokens_generated,
                    duration_ms: r.duration_ms,
                    tokens_per_second: r.tokens_per_second,
                    thermal_state: r.thermal_state.clone(),
                    available_ram_mb: r.available_ram_mb,
                    battery_level: r.battery_level,
                }
            })
            .collect();

        if !batch_results.is_empty() {
            avg_tps /= batch_results.len() as f32;
        }

        Self {
            timestamp: chrono::Utc::now().timestamp(),
            backend: "auto".to_string(),
            device_name: "unknown".to_string(),
            batch_results,
            sustained_sample_count: results.sustained.len(),
            sustained_duration_secs: results.args.duration_secs,
            avg_tokens_per_second: avg_tps,
            peak_tokens_per_second: peak_tps,
        }
    }

    pub fn print(&self) {
        println!("\n=== Benchmark Report ===");
        println!("Backend: {}", self.backend);
        println!("Device: {}", self.device_name);
        println!("Avg tok/s: {:.1}", self.avg_tokens_per_second);
        println!("Peak tok/s: {:.1}", self.peak_tokens_per_second);
        println!("Sustained samples: {}", self.sustained_sample_count);
        println!();

        for result in &self.batch_results {
            println!(
                "Batch {}: {} tok/s ({} tokens in {}ms, thermal: {})",
                result.batch_size,
                result.tokens_per_second,
                result.tokens_generated,
                result.duration_ms,
                result.thermal_state,
            );
        }
        println!("=======================\n");
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}
