# Performance Benchmarks

This document records baseline performance numbers for the Atheer inference engine.
All numbers are measured on reference hardware; actual results vary by device.

## Running Benchmarks

### Criterion (model-free microbenchmarks)

```bash
# All microbenchmarks (run without model dependencies)
cargo bench -p atheer-benches

# perf-bench Criterion harnesses (model-free ones only by default)
cargo bench -p perf-bench

# Specific microbenchmark
cargo bench -p perf-bench -- kv_cache_quantize
```

### Criterion (model-dependent benchmarks)

```bash
# All model-dependent benchmarks (requires GGUF file + tokenizer)
ATHEER_TEST_MODEL=/path/to/model.gguf ATHEER_TOKENIZER_PATH=/path/to/tokenizer.json cargo bench -p perf-bench

# Specific model-dependent benchmark
ATHEER_TEST_MODEL=/path/to/model.gguf ATHEER_TOKENIZER_PATH=/path/to/tokenizer.json cargo bench -p perf-bench -- latency
```

### Binary (sustained throughput)

```bash
cargo run -p perf-bench -- --model-path model.gguf --tokenizer-path tokenizer.json --duration-secs 120 --batch-sizes 1,4,8
```

## Benchmark Suites

### Model-Free (run without GGUF file)

| Bench | File | Measures |
|-------|------|----------|
| `kv_cache_quantize` | `perf-bench/benches/kv_cache_quantize.rs` | Quantize/dequantize throughput (INT8, INT4) at 4K–128K element sizes |
| `kv_cache` | `tests/benches/benches/kv_cache.rs` | KV cache insert/get/truncate/memory operations |
| `ngram_cache` | `tests/benches/benches/ngram_cache.rs` | NGram cache insert/lookup/eviction/large-scale |
| `orchestrator` | `tests/benches/benches/orchestrator.rs` | Orchestrator creation/mode switch/mode query |

### Model-Dependent (require `ATHEER_TEST_MODEL` and `ATHEER_TOKENIZER_PATH`)

| Bench | File | Measures |
|-------|------|----------|
| `throughput` | `perf-bench/benches/throughput.rs` | Cold-start load time and generation tokens/s |
| `latency` | `perf-bench/benches/latency.rs` | P50/P95/P99 decode step latency |
| `checkpoint` | `perf-bench/benches/checkpoint.rs` | Checkpoint save/restore latency at 1K/2K/4K context |
| `memory` | `perf-bench/benches/memory.rs` | Peak RSS at 2K/4K/8K context |
| `sustained` | `perf-bench/benches/sustained.rs` | Long-run throughput and thermal drift |
| `energy` | `perf-bench/benches/energy.rs` | Energy per token |
| `mode_switching` | `perf-bench/benches/mode_switching.rs` | Mode transition latency (Turbo/Balanced/Eco) |
| `thermal_throttling` | `perf-bench/benches/thermal_throttling.rs` | Time-to-throttle under sustained load |

## KV Cache Quantization Throughput Baseline

*To be populated after `cargo bench -p perf-bench -- kv_cache_quantize` on reference hardware.*

| Operation | 4K elements | 16K elements | 32K elements | 128K elements |
|-----------|-------------|--------------|--------------|---------------|
| INT8 quantize | TBD | TBD | TBD | TBD |
| INT8 dequantize | TBD | TBD | TBD | TBD |
| INT8 roundtrip | TBD | TBD | TBD | TBD |
| INT4 quantize | TBD | TBD | TBD | TBD |
| INT4 dequantize | TBD | TBD | TBD | TBD |
| INT4 roundtrip | TBD | TBD | TBD | TBD |

## Decode Latency Baseline

*To be populated after `ATHEER_TEST_MODEL=... cargo bench -p perf-bench -- latency`.*

| Metric | Value |
|--------|-------|
| P50 decode | TBD |
| P95 decode | TBD |
| P99 decode | TBD |
| Samples | 1000 |

## Checkpoint Latency Baseline

*To be populated after `ATHEER_TEST_MODEL=... cargo bench -p perf-bench -- checkpoint`.*

| Context | Save (ms) | Restore (ms) |
|---------|-----------|--------------|
| 1K | TBD | TBD |
| 2K | TBD | TBD |
| 4K | TBD | TBD |

## Memory Baseline

*To be populated after `ATHEER_TEST_MODEL=... cargo bench -p perf-bench -- memory`.*

| Context | Peak RSS |
|---------|----------|
| 2K | TBD |
| 4K | TBD |
| 8K | TBD |

## Generating Baselines

1. Run on a quiet device (no competing workloads)
2. Use the same GGUF model file for all comparative runs
3. Record device specs (CPU, RAM, OS version) alongside each run
4. Run each benchmark 3 times and report the median
5. Update this file with new numbers after significant code changes
