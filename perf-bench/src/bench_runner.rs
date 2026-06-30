use crate::cli::BenchArgs;
use crate::power_monitor::PowerSample;
use atheer_core::InferenceEngine;
use atheer_hardware::HardwareMonitor;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub batch_size: usize,
    pub tokens_generated: u32,
    pub duration_ms: u64,
    pub tokens_per_second: f32,
    pub thermal_state: String,
    pub available_ram_mb: u64,
    pub battery_level: u32,
}

#[derive(Debug)]
pub struct BenchResults {
    pub results: Vec<BenchResult>,
    pub sustained: Vec<PowerSample>,
    pub args: BenchArgs,
}

pub struct BenchRunner {
    args: BenchArgs,
}

impl BenchRunner {
    pub fn new(args: BenchArgs) -> Self {
        Self { args }
    }

    pub fn run(
        &self,
        mut engine: InferenceEngine,
        monitor: &dyn HardwareMonitor,
    ) -> anyhow::Result<BenchResults> {
        let mut results = Vec::new();

        for &batch_size in &self.args.parsed_batch_sizes() {
            tracing::info!("Running batch size={} benchmark", batch_size);

            let prompt = self.args.prompt.repeat(batch_size.max(1));
            let start = Instant::now();

            let (_text, tokens, duration_ms) = engine
                .generate(&prompt, self.args.max_tokens, None)
                .map_err(|e| anyhow::anyhow!("Generation failed at batch={}: {}", batch_size, e))?;

            let _ = start.elapsed();

            let tokens_per_second = if duration_ms > 0 {
                (tokens as f32 / duration_ms as f32) * 1000.0
            } else {
                0.0
            };

            let health = monitor.health();

            results.push(BenchResult {
                batch_size,
                tokens_generated: tokens,
                duration_ms,
                tokens_per_second,
                thermal_state: format!("{:?}", health.thermal),
                available_ram_mb: health.available_ram_mb,
                battery_level: health.battery_level,
            });

            tracing::info!(
                "batch={}: {} tokens in {}ms ({:.1} tok/s)",
                batch_size,
                tokens,
                duration_ms,
                tokens_per_second
            );


        }

        // Sustained benchmark
        let sustained = self.run_sustained(&mut engine, monitor)?;

        Ok(BenchResults {
            results,
            sustained,
            args: self.args.clone(),
        })
    }

    fn run_sustained(
        &self,
        engine: &mut InferenceEngine,
        monitor: &dyn HardwareMonitor,
    ) -> anyhow::Result<Vec<PowerSample>> {
        tracing::info!(
            "Running sustained benchmark for {} seconds",
            self.args.duration_secs
        );

        let mut samples = Vec::new();
        let start = Instant::now();
        let prompt = &self.args.prompt;
        let sample_interval = Duration::from_secs(1);
        let mut last_sample = Instant::now();

        while start.elapsed() < Duration::from_secs(self.args.duration_secs) {
            let gen_start = Instant::now();
            let (_text, tokens, _duration_ms) = engine
                .generate(prompt, self.args.max_tokens, None)
                .map_err(|e| anyhow::anyhow!("Sustained generation failed: {}", e))?;
            let gen_elapsed = gen_start.elapsed();

            if last_sample.elapsed() >= sample_interval {
                let health = monitor.health();
                samples.push(PowerSample {
                    timestamp: chrono::Utc::now().timestamp(),
                    elapsed_secs: start.elapsed().as_secs_f64(),
                    tokens_generated: tokens,
                    generation_time_ms: gen_elapsed.as_millis() as u64,
                    thermal_state: format!("{:?}", health.thermal),
                    available_ram_mb: health.available_ram_mb,
                    battery_level: health.battery_level,
                    on_battery: health.on_battery,
                });
                last_sample = Instant::now();
            }
        }

        Ok(samples)
    }
}
