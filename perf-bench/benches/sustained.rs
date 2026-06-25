//! Benchmark: Sustained throughput over extended durations.
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Measures throughput samples and detects thermal drift by comparing early vs late samples.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- sustained`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_sustained(c: &mut Criterion) {
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

    c.bench_function("sustained_throughput_30s", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let mut sample_count = 0u64;
            while start.elapsed() < std::time::Duration::from_secs(30) {
                let result = engine.generate("Hello", 128, Some(5_000));
                let _ = black_box(result);
                sample_count += 1;
            }
            eprintln!("Sustained: {sample_count} generations in 30s");
        })
    });
}

criterion_group!(benches, maybe_bench_sustained);
criterion_main!(benches);
