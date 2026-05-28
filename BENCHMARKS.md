# Performance Benchmarks

This document records baseline performance numbers for the Atheer inference engine.
All numbers are measured on reference hardware; actual results vary by device.

## Running Benchmarks

### Criterion (microbenchmarks)

```bash
# All microbenchmarks (model-free ones run without dependencies)
cargo bench -p perf-bench

# Specific benchmark
cargo bench -p perf-bench -- kv_cache_quantize

# Model-dependent benchmarks (requires GGUF file)
ATHEER_TEST_MODEL=/path/to/model.gguf cargo bench -p perf-bench
```

### Binary (sustained throughput)

```bash
cargo run -p perf-bench -- --model-path model.gguf --duration-secs 120 --batch-sizes 1,4,8
```

## Benchmark Suites

| Bench | File | Model Required | Measures |
|-------|------|---------------|---------|
| `throughput` | `benches/throughput.rs` | Yes | Cold-start load time |
| `sustained` | `benches/sustained.rs` | Yes | Long-run throughput, thermal drift |
| `energy` | `benches/energy.rs` | Yes | Energy per token |
| `mode_switching` | `benches/mode_switching.rs` | Yes | Mode transition latency |
| `thermal_throttling` | `benches/thermal_throttling.rs` | Yes | Time-to-throttle under load |
| `kv_cache_quantize` | `benches/kv_cache_quantize.rs` | No | Quantize/dequantize throughput (INT8, INT4) at 4K–128K element sizes |
| `latency` | `benches/latency.rs` | Yes | P50/P95/P99 decode step latency |
| `checkpoint` | `benches/checkpoint.rs` | Yes | Checkpoint save/restore latency at 1K/2K/4K context |
| `memory` | `benches/memory.rs` | Yes | Peak RSS at 2K/4K/8K context |
| `thermal_response` | `benches/thermal_response.rs` | Yes | Sustained decode under thermal pressure |

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
