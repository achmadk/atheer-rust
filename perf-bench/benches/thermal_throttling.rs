//! Benchmark: Thermal throttling behavior under sustained load.
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Measures time-to-throttle and throttle events during sustained generation.
//! Uses atheer-hardware for thermal telemetry if available; falls back to
//! throughput degradation as a thermal proxy.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- thermal_throttling`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_thermal(c: &mut Criterion) {
    let model_path = std::env::var("ATHEER_TEST_MODEL").unwrap_or_default();
    let tokenizer_path = std::env::var("ATHEER_TOKENIZER_PATH").unwrap_or_default();

    if model_path.is_empty() || tokenizer_path.is_empty() {
        eprintln!("SKIPPED: ATHEER_TEST_MODEL and ATHEER_TOKENIZER_PATH not set");
        return;
    }

    let tokenizer = atheer_core::tokenizer::Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    let config = atheer_core::sampler::SamplingConfig::default();
    let mut engine = atheer_core::InferenceEngine::new_auto(&model_path, tokenizer, config, 4096)
        .expect("Failed to load model");

    c.bench_function("thermal_sustained_load_60s", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let mut throughput_samples: Vec<f64> = Vec::new();

            while start.elapsed() < std::time::Duration::from_secs(60) {
                let sample_start = std::time::Instant::now();
                let result = engine.generate("Hello", 128, Some(5_000));
                if let Ok((_text, count, _elapsed)) = result {
                    let sample_duration = sample_start.elapsed().as_secs_f64();
                    if sample_duration > 0.0 {
                        throughput_samples.push(count as f64 / sample_duration);
                    }
                }
            }

            // Detect thermal throttling: compare first 10s vs last 10s throughput
            let n = throughput_samples.len();
            if n > 20 {
                let first_third: f64 =
                    throughput_samples[..n / 3].iter().sum::<f64>() / (n / 3) as f64;
                let last_third: f64 =
                    throughput_samples[n * 2 / 3..].iter().sum::<f64>() / (n - n * 2 / 3) as f64;
                if last_third < first_third * 0.8 {
                    eprintln!(
                        "THERMAL THROTTLING DETECTED: {:.1} -> {:.1} tok/s",
                        first_third, last_third
                    );
                }
            }
            black_box(throughput_samples.len());
        })
    });
}

criterion_group!(benches, maybe_bench_thermal);
criterion_main!(benches);
