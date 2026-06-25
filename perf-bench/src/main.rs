//! perf-bench: Performance-per-watt benchmarking binary for Atheer.
//!
//! Measures throughput, sustained performance, and energy characteristics
//! of the Atheer inference engine across configurable batch sizes and durations.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use atheer_core::sampler::SamplingConfig;
use atheer_core::tokenizer::Tokenizer;
use atheer_core::InferenceEngine;

/// Atheer performance benchmarking tool.
#[derive(Parser, Debug)]
#[command(name = "perf-bench", about = "Benchmark Atheer inference engine throughput and energy")]
struct Cli {
    /// Path to GGUF model file (or set ATHEER_TEST_MODEL env var)
    #[arg(long, env = "ATHEER_TEST_MODEL")]
    model_path: Option<PathBuf>,

    /// Path to tokenizer.json file (or set ATHEER_TOKENIZER_PATH env var)
    #[arg(long, env = "ATHEER_TOKENIZER_PATH")]
    tokenizer_path: Option<PathBuf>,

    /// Comma-separated batch sizes (e.g., "1,4,8")
    #[arg(long, default_value = "1")]
    batch_sizes: String,

    /// Benchmark duration in seconds
    #[arg(long, default_value_t = 30)]
    duration_secs: u64,

    /// Output JSON report path
    #[arg(long, default_value = "bench-report.json")]
    output: PathBuf,

    /// Max sequence length
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BenchReport {
    avg_tokens_per_second: f64,
    peak_tokens_per_second: f64,
    sustained_sample_count: u64,
    sustained_duration_secs: u64,
    backend: String,
    device_name: String,
    batch_results: Vec<BatchResult>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BatchResult {
    batch_size: u32,
    tokens_per_second: f64,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    // Validate required args
    let model_path = cli.model_path.unwrap_or_else(|| {
        eprintln!("ERROR: --model-path (or ATHEER_TEST_MODEL) is required");
        std::process::exit(1);
    });
    let tokenizer_path = cli.tokenizer_path.unwrap_or_else(|| {
        eprintln!("ERROR: --tokenizer-path (or ATHEER_TOKENIZER_PATH) is required");
        std::process::exit(1);
    });

    eprintln!("Loading model from: {}", model_path.display());
    eprintln!("Loading tokenizer from: {}", tokenizer_path.display());

    // Load tokenizer
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: Failed to load tokenizer: {e}");
            std::process::exit(1);
        }
    };

    // Load model via auto-backend
    let config = SamplingConfig::default();
    let mut engine = match InferenceEngine::new_auto(&model_path, tokenizer, config, cli.max_seq_len)
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let backend_name = "auto".to_string();
    let device_name = "auto".to_string();

    // Parse batch sizes
    let batch_sizes: Vec<u32> = cli
        .batch_sizes
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if batch_sizes.is_empty() {
        eprintln!("ERROR: No valid batch sizes specified");
        std::process::exit(1);
    }

    eprintln!("Batch sizes: {:?}", batch_sizes);
    eprintln!("Duration: {}s", cli.duration_secs);

    // Measure per-batch throughput
    let mut batch_results = Vec::new();
    for &batch_size in &batch_sizes {
        eprintln!("Benchmarking batch size {batch_size}...");
        let tokens_per_second = measure_throughput(&mut engine, cli.duration_secs);
        eprintln!("  Batch {batch_size}: {tokens_per_second:.1} tok/s");
        batch_results.push(BatchResult {
            batch_size,
            tokens_per_second,
        });
    }

    // Sustained sampling
    eprintln!("Running sustained benchmark for {}s...", cli.duration_secs);
    let samples = measure_sustained(&mut engine, cli.duration_secs);

    let avg_tps = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    };
    let peak_tps = samples.iter().copied().fold(0.0_f64, f64::max);

    let report = BenchReport {
        avg_tokens_per_second: avg_tps,
        peak_tokens_per_second: peak_tps,
        sustained_sample_count: samples.len() as u64,
        sustained_duration_secs: cli.duration_secs,
        backend: backend_name,
        device_name,
        batch_results,
    };

    let json = serde_json::to_string_pretty(&report).expect("Failed to serialize report");
    std::fs::write(&cli.output, &json).expect("Failed to write benchmark report");
    eprintln!("Report written to: {}", cli.output.display());
}

/// Measure average tokens/second for a given duration.
fn measure_throughput(engine: &mut InferenceEngine, duration_secs: u64) -> f64 {
    let prompt = "Hello";
    let max_tokens = 128u32;
    let start = Instant::now();
    let mut total_tokens: u64 = 0;

    while start.elapsed() < Duration::from_secs(duration_secs) {
        match engine.generate(prompt, max_tokens, Some(5000)) {
            Ok((_text, tokens_generated, _elapsed)) => {
                total_tokens += tokens_generated as u64;
            }
            Err(e) => {
                eprintln!("Generation error: {e}");
                break;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        total_tokens as f64 / elapsed
    } else {
        0.0
    }
}

/// Collect per-second throughput samples over the given duration.
fn measure_sustained(engine: &mut InferenceEngine, duration_secs: u64) -> Vec<f64> {
    let prompt = "Hello";
    let max_tokens = 128u32;
    let sample_interval = Duration::from_secs(1);
    let mut samples = Vec::new();

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(duration_secs) {
        let sample_start = Instant::now();
        let mut tokens_this_second: u64 = 0;

        while sample_start.elapsed() < sample_interval {
            match engine.generate(prompt, max_tokens, Some(5000)) {
                Ok((_text, tokens_generated, _elapsed)) => {
                    tokens_this_second += tokens_generated as u64;
                }
                Err(e) => {
                    eprintln!("Generation error during sustained: {e}");
                    break;
                }
            }
        }

        let tps = tokens_this_second as f64 / sample_interval.as_secs_f64();
        samples.push(tps);
    }

    samples
}
