//! Benchmark: Decode step latency (P50/P95/P99).
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Skips gracefully if not set.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- latency`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn maybe_bench_latency(c: &mut Criterion) {
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

    let prompt = "Hello, world!";
    let max_tokens = 512u32;
    let mut latencies: Vec<f64> = Vec::with_capacity(1000);

    for _ in 0..1000 {
        let start = std::time::Instant::now();
        match engine.generate(prompt, max_tokens, Some(10_000)) {
            Ok((_text, _count, _elapsed)) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
                latencies.push(elapsed);
            }
            Err(e) => {
                eprintln!("Generation error: {e}");
            }
        }
    }

    if latencies.is_empty() {
        return;
    }

    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = latencies[(latencies.len() as f64 * 0.50) as usize];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    let group = c.benchmark_group("decode_latency");
    eprintln!("P50: {p50:.2}ms, P95: {p95:.2}ms, P99: {p99:.2}ms, Samples: {}", latencies.len());
    drop(group);
}

criterion_group!(benches, maybe_bench_latency);
criterion_main!(benches);
