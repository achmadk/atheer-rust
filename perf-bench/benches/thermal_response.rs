use criterion::{criterion_group, criterion_main, Criterion};

fn bench_thermal_response(c: &mut Criterion) {
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

    c.bench_function("thermal_mode_transition", |b| {
        b.iter(|| {
            let _ = engine.generate(
                "This benchmark measures decode throughput under sustained load.",
                500,
                Some(30_000),
            );
        });
    });
}

criterion_group!(benches, bench_thermal_response);
criterion_main!(benches);
