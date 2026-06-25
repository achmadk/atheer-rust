//! Benchmark: Checkpoint save/restore latency.
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Skips gracefully if not set.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- checkpoint`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_checkpoint(c: &mut Criterion) {
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

    let context_lengths = [1024usize, 2048, 4096];
    let prompt = "Hello";

    for &ctx_len in &context_lengths {
        c.bench_function(&format!("checkpoint_save_{}k", ctx_len / 1024), |b| {
            b.iter(|| {
                let _ = engine.generate(prompt, ctx_len as u32, Some(30_000));
            })
        });
    }
}

criterion_group!(benches, maybe_bench_checkpoint);
criterion_main!(benches);
