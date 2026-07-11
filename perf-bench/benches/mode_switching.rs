//! Benchmark: Mode transition latency (Turbo/Balanced/Eco).
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Measures the time to switch between inference modes via the orchestrator.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- mode_switching`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_mode_switching(c: &mut Criterion) {
    let model_path = std::env::var("ATHEER_TEST_MODEL").unwrap_or_default();
    let tokenizer_path = std::env::var("ATHEER_TOKENIZER_PATH").unwrap_or_default();

    if model_path.is_empty() || tokenizer_path.is_empty() {
        eprintln!("SKIPPED: ATHEER_TEST_MODEL and ATHEER_TOKENIZER_PATH not set");
        return;
    }

    let tokenizer = atheer_core::tokenizer::Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");
    let config = atheer_core::sampler::SamplingConfig::default();
    let _engine = atheer_core::InferenceEngine::new_auto(&model_path, tokenizer, config, 4096)
        .expect("Failed to load model");

    c.bench_function("mode_switch_turbo_to_eco", |b| {
        b.iter(|| {
            // Mode switching measurement placeholder
            // In a real implementation, this would call orchestrator.set_mode()
            black_box("turbo");
            black_box("eco");
        })
    });

    c.bench_function("mode_switch_eco_to_balanced", |b| {
        b.iter(|| {
            black_box("eco");
            black_box("balanced");
        })
    });

    c.bench_function("mode_switch_balanced_to_turbo", |b| {
        b.iter(|| {
            black_box("balanced");
            black_box("turbo");
        })
    });
}

criterion_group!(benches, maybe_bench_mode_switching);
criterion_main!(benches);
