use criterion::{criterion_group, criterion_main, Criterion};

fn bench_thermal_response(c: &mut Criterion) {
    let model_path = match std::env::var("ATHEER_TEST_MODEL") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("WARN: ATHEER_TEST_MODEL not set — skipping thermal response benchmark");
            return;
        }
    };

    let device = candle_core::Device::Cpu;
    let model = atheer_core::Model::from_gguf(&model_path, &device)
        .expect("Failed to load model");
    let tokenizer = atheer_core::Tokenizer::from_file("tokenizer.json")
        .expect("Failed to load tokenizer");

    let sampling_config = atheer_core::sampler::SamplingConfig {
        temperature: 0.0,
        ..Default::default()
    };

    let mut engine = atheer_core::InferenceEngine::new(
        model, tokenizer, sampling_config, 4096,
    )
    .expect("Failed to create inference engine");

    c.bench_function("thermal_mode_transition", |b| {
        b.iter(|| {
            let _ = engine.generate(
                "This benchmark measures decode throughput under sustained load.",
                500,
            );
        });
    });
}

criterion_group!(benches, bench_thermal_response);
criterion_main!(benches);
