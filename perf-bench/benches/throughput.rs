//! Benchmark: Cold-start load time and generation throughput (tokens/s).
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Skips gracefully if not set.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- throughput`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_load_time(c: &mut Criterion) {
    let model_path = std::env::var("ATHEER_TEST_MODEL").unwrap_or_default();
    let tokenizer_path = std::env::var("ATHEER_TOKENIZER_PATH").unwrap_or_default();

    if model_path.is_empty() || tokenizer_path.is_empty() {
        eprintln!("SKIPPED: ATHEER_TEST_MODEL and ATHEER_TOKENIZER_PATH not set");
        return;
    }

    c.bench_function("model_cold_start_load", |b| {
        b.iter(|| {
            let tokenizer = atheer_core::tokenizer::Tokenizer::from_file(&tokenizer_path)
                .expect("Failed to load tokenizer");
            let config = atheer_core::sampler::SamplingConfig::default();
            let _engine = atheer_core::InferenceEngine::new_auto(
                &model_path,
                tokenizer,
                config,
                4096,
            )
            .expect("Failed to load model");
        })
    });

    // Throughput benchmark
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

    c.bench_function("generation_throughput", |b| {
        b.iter(|| {
            let result = engine.generate("Hello, world!", 256, Some(10_000));
            black_box(result)
        })
    });
}

criterion_group!(benches, maybe_bench_load_time);
criterion_main!(benches);
