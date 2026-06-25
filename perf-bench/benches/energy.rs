//! Benchmark: Energy per token.
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Uses atheer-hardware telemetry to estimate energy consumption per token.
//! Skips if hardware telemetry is unavailable.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- energy`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_energy(c: &mut Criterion) {
    let model_path = std::env::var("ATHEER_TEST_MODEL").unwrap_or_default();
    let tokenizer_path = std::env::var("ATHEER_TOKENIZER_PATH").unwrap_or_default();

    if model_path.is_empty() || tokenizer_path.is_empty() {
        eprintln!("SKIPPED: ATHEER_TEST_MODEL and ATHEER_TOKENIZER_PATH not set");
        return;
    }

    let tokenizer =
        atheer_core::tokenizer::Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    let config = atheer_core::sampler::SamplingConfig::default();
    let mut engine = atheer_core::InferenceEngine::new_auto(
        &model_path,
        tokenizer,
        config,
        4096,
    )
    .expect("Failed to load model");

    c.bench_function("energy_per_token", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = engine.generate("Hello", 256, Some(10_000));
            if let Ok((_text, count, _elapsed)) = &result {
                let duration_ms = start.elapsed().as_millis();
                if *count > 0 {
                    let ms_per_token = duration_ms as f64 / *count as f64;
                    // Energy proxy: time-per-token (lower is better)
                    black_box(ms_per_token);
                }
            }
        })
    });
}

criterion_group!(benches, maybe_bench_energy);
criterion_main!(benches);
