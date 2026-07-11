//! Benchmark: Peak RSS at various context lengths.
//!
//! Model-dependent. Requires `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH` env vars.
//! Reads RSS from /proc/self/status on Linux.
//!
//! Run: `ATHEER_TEST_MODEL=... ATHEER_TOKENIZER_PATH=... cargo bench -p perf-bench -- memory`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
fn read_rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            if let Some(kb_str) = line.split_whitespace().nth(1) {
                return kb_str.parse::<u64>().unwrap_or(0);
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn read_rss_kb() -> u64 {
    0
}

fn maybe_bench_memory(c: &mut Criterion) {
    let model_path = std::env::var("ATHEER_TEST_MODEL").unwrap_or_default();
    let tokenizer_path = std::env::var("ATHEER_TOKENIZER_PATH").unwrap_or_default();

    if model_path.is_empty() || tokenizer_path.is_empty() {
        eprintln!("SKIPPED: ATHEER_TEST_MODEL and ATHEER_TOKENIZER_PATH not set");
        return;
    }

    let config = atheer_core::sampler::SamplingConfig::default();

    for &ctx_len in &[2048usize, 4096, 8192] {
        let tokenizer = atheer_core::tokenizer::Tokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer");
        let mut engine =
            atheer_core::InferenceEngine::new_auto(&model_path, tokenizer, config.clone(), ctx_len)
                .expect("Failed to load model");

        let _ = engine.generate("Hello", 128, Some(10_000));
        let rss = read_rss_kb();

        c.bench_function(&format!("memory_peak_rss_{}k", ctx_len / 1024), |b| {
            b.iter(|| {
                let current = black_box(read_rss_kb());
                eprintln!("RSS at {}k ctx: {} KB", ctx_len / 1024, current);
            })
        });

        eprintln!("Peak RSS at {}k context: {} KB", ctx_len / 1024, rss);
    }
}

criterion_group!(benches, maybe_bench_memory);
criterion_main!(benches);
