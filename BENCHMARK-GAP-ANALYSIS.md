# Benchmark Gap Analysis: Validating Atheer's USP Claims

> **Infrastructure Assessment** · July 2026
>
> An analysis of what we can measure today, what we cannot, and a phased plan
> to empirically validate the key differentiators described in the whitepaper.

---

## 1. Current Benchmarking Infrastructure

### 1.1 Overview

The `perf-bench` crate provides both a CLI binary and Criterion benchmark harnesses:

```
perf-bench/
├── src/
│   ├── main.rs            — CLI binary (model-dependent)
│   ├── bench_runner.rs    — Reusable benchmark runner with HardwareMonitor integration
│   ├── cli.rs             — CLI argument parsing
│   ├── power_monitor.rs   — PowerSample struct (tok/s, thermal, memory, battery)
│   └── report.rs          — BenchReport serialization (JSON output)
│
└── benches/
    ├── throughput.rs          — tok/s measurement
    ├── sustained.rs           — 30s sustained with per-second sampling
    ├── latency.rs             — P50/P95/P99 decode step latency
    ├── energy.rs              — Energy-per-token estimation
    ├── memory.rs              — Peak RSS at 2K/4K/8K context
    ├── thermal_response.rs    — (model-dep) decode under sustained load
    ├── thermal_throttling.rs  — (model-dep) thermal behavior over time
    ├── mode_switching.rs      — (model-dep) **PLACEHOLDER — does NOT call orchestrator**
    ├── kv_cache_quantize.rs   — Model-free INT8/INT4 quantize/dequantize
    └── checkpoint.rs          — Model checkpoint save/load
```

Additional model-free benches live in `atheer-benches/tests/benches/`:
- `kv_cache.rs` — insert/get/truncate/memory
- `ngram_cache.rs` — insert/lookup/eviction
- `orchestrator.rs` — creation/mode switch/mode query

### 1.2 Metrics Currently Captured

The `PowerSample` struct captures per-second telemetry during sustained runs:

```rust
pub struct PowerSample {
    pub timestamp: i64,
    pub elapsed_secs: f64,
    pub tokens_generated: u32,
    pub generation_time_ms: u64,
    pub thermal_state: String,      // From HardwareMonitor
    pub available_ram_mb: u64,      // From HardwareMonitor
    pub battery_level: u32,         // From HardwareMonitor
    pub on_battery: bool,           // From HardwareMonitor
}
```

The `BenchRunner` already integrates with `HardwareMonitor`:

```rust
pub fn run(
    &self,
    mut engine: InferenceEngine,
    monitor: &dyn HardwareMonitor,     // <-- already wired
) -> anyhow::Result<BenchResults>
```

---

## 2. USP Claims vs. Benchmark Coverage

| USP Claim | Current Coverage | Status | Phase |
|---|---|---|---|
| **Thermal prediction prevents throttling cliff** | No thermal simulator. Can't test predictive downgrade vs flat-out without heating a device. | ❌ | P2–P3 |
| **L1/L2/L3 cache saves memory** | No multi-turn session benchmark. Cache never exercised across promotion/eviction cycles. | ❌ | P1 |
| **NPU-first probe finds fastest backend** | `probe_all()` never benchmarked. No cross-backend comparison. | ❌ | P1 |
| **Mode switching adapts to conditions** | `mode_switching.rs` exists but uses `black_box("turbo")` — doesn't call orchestrator. | ⚠️ | P1 |
| **Grammar constraints add negligible overhead** | No benchmark comparing tok/s with and without grammar. | ❌ | P1 |
| **Eco mode reduces energy consumption** | `PowerSample` captures thermal, but no energy sensor (mA) on most devices. | ⚠️ | P3 |
| **Rust lower overhead than C++/Python** | No comparative benchmark vs llama.cpp/MLC LLM on identical hardware. | ❌ | P2–P3 |

---

## 3. Deep Analysis of Each Gap

### 3.1 Thermal Prediction (Hard)

**Why it's hard:** Real thermal behavior requires sustained inference (>5 min) on a
physical phone. Desktop CPUs have active cooling — they don't throttle like phone SoCs.
`ThermalModel` feeds on real temperature samples; you can't validate prediction
accuracy without either real data or a simulator.

**Solutions:**
- **Option A** (Phase 2): Record thermal traces from a real device (run 5-min inference,
  log temperature curve at 1 Hz) → build a `ThermalSimulator` that replays traces →
  validate `ThermalModel` prediction accuracy offline
- **Option B** (Phase 3): Thermal chamber with controlled ambient temperature →
  prove that adaptive mode switching maintains smoother throughput vs flat-out Turbo

### 3.2 L1/L2/L3 Cache Hierarchy (Medium)

**Why it's medium difficulty:** The `MemoryBank` API is well-defined and self-contained.
A multi-turn benchmark can run on any platform (even Linux desktop with CPU backend).

**What's needed:**
- A benchmark script simulating N conversation turns with cache promotion
- Run the same multi-turn session **with** and **without** L2/L3 enabled
- Measure: peak memory, total tokens processed before OOM, time-to-promotion

**Pseudocode:**
```rust
fn bench_cache_hierarchy() {
    let bank = MemoryBank::new(1024); // 1 GB max

    for turn in 0..20 {
        simulate_generation(500, &bank);
        let mem = get_current_rss();

        if turn % 5 == 0 {
            bank.trigger_handoff(&format!("model_{}", turn / 5));
        }

        let mem_after = get_current_rss();
        let memory_saved = mem - mem_after; // vs baseline without hierarchy
    }
}
```

### 3.3 NPU-first Probe (Easy)

**What's needed:**
- Benchmark `BackendManager::probe_all()` — measure time-to-first-available-backend
- Run same prompt through each available backend — compare tok/s

**Feasibility:** 4 hours of work. No real device needed for CPU/Metal (on macOS).

### 3.4 Mode Switching (Easy — Infrastructure Exists)

**Current state:** The bench file exists but the switching logic is `black_box("turbo")`
— it doesn't call `Orchestrator::set_mode()`.

**What's needed:**
- Wire `Orchestrator` into the benchmark
- Measure transition latency for Turbo→Eco, Eco→Balanced, Balanced→Turbo
- Verify each transition actually changes speculation depth

**Feasibility:** 4 hours of work.

### 3.5 Grammar Overhead (Easy)

**What's needed:**
- Run same prompt with and without `JsonGrammar` constraint
- Compare tok/s delta
- The `GrammarSampler` prunes invalid tokens at each step — quantify the cost

### 3.6 Energy Measurement (Hard)

**Why it's hard:**
- Desktop Linux: RAPL measures package power, not per-process
- Android `BatteryManager`: battery level (%) only, no instantaneous mA
- iOS: no public per-process energy API
- Real measurement requires external hardware (Monsoon power monitor)

**Workaround for Phase 1:** Use thermal curve slope as a proxy for power consumption
(steeper slope = more power). This is imprecise but directionally correct.

### 3.7 Competitive Comparison (Medium)

**What's needed:**
- Run identical model (GGUF) on Atheer and llama.cpp on identical hardware
- Measure tok/s, peak memory, sustained throughput over 5 min
- Requires competitor engine installed (trivial for llama.cpp, harder for MLC LLM)

---

## 4. Three-Phase Benchmark Plan

### Phase 1: Desktop-Validated (Doable Now)

| # | Benchmark | Effort | Impact | What It Proves |
|---|---|---|---|---|
| 1 | Cache hierarchy memory savings | 1–2 days | 🔥 High | Memory advantage of L1/L2/L3 |
| 2 | Backend probe latency | 4 hours | 📊 Medium | NPU-first auto-select speed |
| 3 | Grammar overhead | 4 hours | 📊 Medium | Cost of structured output |
| 4 | Fix mode switching bench | 4 hours | 🔧 Low | Unblocks the existing bench |
| 5 | Multi-turn throughput w/ eviction | 1–2 days | 🔥 High | Agent loop viability |

### Phase 2: Real-Device (Needs iOS/Android Hardware)

| # | Benchmark | Effort | Impact |
|---|---|---|---|
| 6 | Thermal curve recording (build simulator) | 1 week | 🔥🔥 Critical |
| 7 | Cross-backend: CoreML vs Metal vs CPU | 1 week | 🔥🔥 High |
| 8 | Sustained adaptive vs flat-out Turbo | 1 week | 🔥🔥🔥 Very High |
| 9 | Memory pressure under multi-turn agent | 3 days | 🔥 High |

### Phase 3: Lab-Grade (Needs Instrumentation)

| # | Benchmark | Effort | Impact |
|---|---|---|---|
| 10 | Thermal chamber controlled heating | 2 weeks | 🔥🔥🔥 Definitively validates thermal claim |
| 11 | Competitive: Atheer vs llama.cpp vs MLC | 2 weeks | 🔥🔥🔥 Direct comparison data |
| 12 | Energy measurement w/ power monitor | 3 weeks | 🔥🔥 Hard data for Eco mode |
| 13 | Battery drain: Eco vs Turbo over 30 min | 1 week | 🔥🔥 Consumer-relevant metric |

---

## 5. Recommended Sprint Plan

```
Week 1-2 (Phase 1):
├── Cache hierarchy benchmark        (P1, high impact)
├── Backend probe benchmark          (P1, medium impact)
├── Grammar overhead benchmark       (P1, medium impact)
├── Fix mode switching bench         (P1, quick win)
└── Multi-turn throughput bench      (P1, high impact)

Week 3-4 (Phase 2 start):
├── Record thermal traces on device
├── Cross-backend comparison
└── Sustained adaptive vs Turbo comparison

Future (Phase 3):
├── Competitive benchmark campaign
├── Thermal chamber validation
└── Energy measurement setup
```

---

## 6. Infrastructure Requirements

### Have
- Criterion.rs integration ✓
- CLI binary with JSON output ✓
- `HardwareMonitor` integration in `BenchRunner` ✓
- `PowerSample` with thermal/memory/battery fields ✓
- Model-free vs model-dependent separation ✓

### Need
- `ThermalSimulator` for offline `ThermalModel` validation
- `MemorySnapshot` utility (process RSS at cache promotion points)
- Orchestrator wiring into bench infra
- Cross-backend runner (cycle through available backends)
- Device-side benchmark scripts (Android ADB, iOS)
- Comparison visualization tool (Atheer vs competitors)

---

## 7. Conclusion

The benchmarking infrastructure has good bones — Criterion harnesses, hardware telemetry
integration, JSON output — but the highest-value benchmarks (those that would prove
the USP claims in the whitepaper) are either not wired or don't exist yet.

The good news: **Phase 1 benchmarks (cache hierarchy, backend probe, grammar overhead,
mode switching) are straightforward to implement and would produce real numbers within
a sprint.** These don't need a real device, a thermal chamber, or competitor engines
installed. They validate the architecture using the code itself.

Phase 2 and 3 require device access and are better suited for a dedicated benchmarking
campaign once the Phase 1 numbers establish a baseline.
