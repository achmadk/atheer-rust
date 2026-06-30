use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "perf-bench", about = "Performance-per-watt benchmarking for atheer-rust")]
pub struct BenchArgs {
    /// Path to GGUF model file
    #[arg(short, long)]
    pub model_path: String,

    /// Path to tokenizer.json
    #[arg(short, long)]
    pub tokenizer_path: Option<String>,

    /// Prompt text for generation
    #[arg(short, long, default_value = "Hello, how are you today?")]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(short, long, default_value_t = 128)]
    pub max_tokens: u32,

    /// Maximum sequence length
    #[arg(long, default_value_t = 4096)]
    pub max_seq_len: u32,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7f64)]
    pub temperature: f64,

    /// Batch sizes to test (comma-separated, e.g. "1,4,8,16")
    #[arg(long, default_value = "1,4,8")]
    pub batch_sizes: String,

    /// Test duration in seconds for sustained benchmark
    #[arg(long, default_value_t = 30)]
    pub duration_secs: u64,

    /// Path to output JSON report
    #[arg(short, long, default_value = "bench-report.json")]
    pub output: String,
}

impl BenchArgs {
    pub fn parsed_batch_sizes(&self) -> Vec<usize> {
        self.batch_sizes
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }
}
