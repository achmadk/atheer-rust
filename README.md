# Atheer-Rust

Mobile inference engine for LLMs on iOS and Android.

## Features

- **Multi-backend GPU/NPU acceleration**: Powered by `candle` framework, supporting Metal (iOS GPU), Vulkan (Android GPU with custom GLSL compute shaders), NNAPI (Android NPU/DSP via NDK FFI), and CoreML/ANE (iOS NPU). Platform-specific probe order: NPU → GPU → CPU.
- **Agentic Workflows**: Built-in support for tool-calling definitions, extraction, and autonomous agent loops.
- **Structured Output**: Native grammar-constrained decoding (Pushdown Automaton) guaranteeing valid JSON output.
- **Memory hierarchy**: L1/L2/L3 KV caching with intelligent eviction policies.
- **Cache encryption**: L3 KV cache snapshots encrypted at rest via AES-256-GCM (LZ4 compress → encrypt with random 12-byte nonce, distinct AAD `"atheer-cache-v1"`). Encryption key zeroized on drop. Key resolved at engine init: config key → ephemeral session key → None (L3 disabled).
- **Dynamic mode switching**: Eco, Balanced, and Turbo modes based on live hardware telemetry (thermal, memory, battery) sampled at 1 Hz.
- **Platform hardware telemetry**: Android JNI bridge for thermal headroom, available memory, and battery level; iOS/macOS telemetry via `objc2` FFI (`IosMonitor` with 1 Hz sampling thread).
- **Performance-per-watt measurement**: Benchmarking infrastructure for throughput, energy, and thermal throttling curves (Criterion benches + perf-bench binary).
- **Production-ready**: Memory safe (Rust), crash reporting with `atheer-core`, graceful degradation to CPU when accelerators are unavailable.

## Architecture

```
┌─────────────────────────────────────────┐
│           iOS / Android App             │
└────────────────┬────────────────────────┘
                 │ uniffi FFI
┌────────────────▼────────────────────────┐
│           atheer-ffi                    │
│    (Swift/Kotlin bindings)              │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           atheer-core                   │
│   (Candle inference, model & token)     │
└────────┬───────────────────┬──-─────────┘
         │                   │
┌────────▼────────┐  ┌────────▼──────────┐
│ atheer-accel   │  │atheer-orchestrator │
│ (Metal/Vulkan/ │  │(Modes, Grammar,    │
│  NNAPI/CPU)    │  │ Agent Loop)        │
└────────────────┘  └────────┬───────────┘
         │                   │
┌────────▼───────────────────▼────────┐
│        atheer-memory-bank           │
│   (L1/L2/L3 KV cache, handoff,      │
│    EncryptedStore AES-256-GCM)      │
└─────────────────────────────────────┘
         ▲
         │ health snapshot (1 Hz)
┌────────▼────────────────────────────┐
│        atheer-hardware              │
│  (GenericMonitor, JNI bridge)       │
└─────────────────────────────────────┘
```

## Quick Start

```bash
# Build the workspace
cargo build --workspace

# Run all tests
cargo test --workspace
```

See platform-specific examples under [Backend Selection](#backend-selection) for iOS (Swift) and Android (Kotlin) configuration.

## Backend Selection

Atheer auto-detects the best available backend for the current device. To override, set `backendType` in `AtheerConfig`:

| Backend     | Platform          | Probe Priority | Notes                            |
|-------------|-------------------|----------------|----------------------------------|
| `cpu`       | Any               | always last    | Fallback, no acceleration        |
| `metal`     | iOS 15+           | iOS: 2nd       | GPU via Candle Metal backend (`Device::Metal`) |
| `vulkan`    | Android 26+       | Android: 2nd   | GPU via custom Vulkan compute shaders (SPIR-V) |
| `nnapi`     | Android 24+       | Android: 1st   | NPU/DSP via NNAPI NDK FFI bindings (requires NDK r29) |
| `coreml`    | iOS 15+           | iOS: 1st       | ANE via CoreML compatibility detection (real inference requires macOS build) |

When `backendType` is `None` (default), the engine probes backends in platform-specific priority order:
- **iOS**: CoreML (ANE) → Metal (GPU) → CPU
- **Android**: NNAPI (NPU) → Vulkan (GPU) → CPU

### iOS (Swift)

```swift
import AtheerFFI

var config = AtheerConfig()
config.modelPath = "/models/llama.gguf"
config.tokenizerPath = "/models/tokenizer.json"
config.adaptive = true
// Optional: explicitly select backend
// config.backendType = .metal  // Metal GPU
// config.backendType = .coreml // CoreML/ANE (requires macOS build)

let engine = AtheerEngine(config: config)
try engine.initialize()

var request = GenerationRequest(prompt: "Hello")
// To enable structured output:
// request.jsonSchema = "{ \"type\": \"object\" }"
// request.tools = [ToolDefinition(name: "get_weather", description: "...", parametersSchema: "...")]

let response = try engine.generateSync(request: request)
print(response.text)
```

On iOS, the Metal backend delegates tensor operations to `candle_core::Device::Metal`, providing real GPU acceleration for prefill and decode. The CoreML backend detects model compatibility with ANE constraints (architecture, quantization, size) and falls back to Metal for incompatible models.

**iOS hardware telemetry** (thermal, memory, battery) is planned but requires a macOS build environment for `objc2` FFI compilation — currently blocked on Linux CI.

### Android (Kotlin)

```kotlin
import com.aether.ffi.*

val config = AtheerConfig(
    modelPath = "/models/llama.gguf",
    tokenizerPath = "/models/tokenizer.json",
    adaptive = true,
    // backendType = AtheerBackendType.NNAPI  // optional override
)
val engine = AtheerEngine(config)
engine.initialize()

val request = GenerationRequest(
    prompt = "Hello",
    maxTokens = 512u,
    temperature = 0.7f,
    jsonSchema = null,
    tools = listOf()
)

val response = engine.generateSync(request)
println(response.text)
```

### NNAPI NDK Backend (Android NPU/DSP)

The NNAPI backend provides acceleration via Android's Neural Networks API using raw NDK FFI bindings declared in `atheer-accel/src/nnapi_ndk.rs` (~20 extern functions covering the full inference pipeline). The module also includes a full NNAPI graph compiler with 9 supported operation codes:

| Stage | NNAPI API | Implementation |
|-------|-----------|----------------|
| **Device discovery** | `ANeuralNetworks_getDeviceCount`, `getDevice`, `getDeviceName`/`getDeviceType` | `NnapiExecutor::probe()` — enumerates accelerators, returns `None` if no NNAPI runtime available |
| **Graph construction** | `ANeuralNetworksModel_create`, `addOperand`, `addOperation`, `setOperandValue` | `NnapiGraphBuilder` — operand/operation graph with validation, `NnapiOperation` enum with 9 variants |
| **Supported operations** | `ANEURALNETWORKS_ADD`, `MUL`, `FULLY_CONNECTED`, `SOFTMAX`, `LOGISTIC`, `RELU`, `TANH`, `CONCATENATION`, `RESHAPE` | `NnapiOperation::to_nnapi_code()` — maps each variant to its NDK constant with operand validation |
| **Compilation** | `ANeuralNetworksCompilation_create`, `setPreference` (`SUSTAINED_SPEED`), `finish` | `NnapiGraphBuilder::compile()` → `NnapiCompiledModel` |
| **Execution** | `ANeuralNetworksExecution_create`, `setInput`/`setOutput`, `compute` | `NnapiCompiledModel::execute()` with multi-input/output buffer support and automatic cleanup |

**Build requirements:**
- NDK r29+ at `$ANDROID_NDK_HOME`
- API target 26+ (feature level 3+ for device discovery)
- Cross-compilation: `cargo ndk -t arm64-v8a --platform 26 build -p atheer-accel`

### Vulkan Backend (Android GPU)

The Vulkan backend accelerates matrix-vector multiplication and attention using custom GLSL compute shaders compiled to SPIR-V at build time via `naga`:

| Shader | File | Purpose |
|--------|------|---------|
| GEMV | `atheer-accel/shaders/gemv.glsl` | Quantized int8 matrix-vector multiply (DP4A-style) for the decoder's feed-forward layers |
| Attention | `atheer-accel/shaders/attention.glsl` | Flash attention-style softmax + query-key matmul |

Shaders are compiled via a `build.rs` step and dispatched through the existing `VulkanContext` compute pipeline. The backend falls back to CPU when Vulkan is unavailable.

### Runtime Mode Switching

The engine automatically switches between Turbo, Balanced, and Eco modes using real-time thermal, memory, and battery telemetry:

```swift
// Override mode manually
try engine.setMode(.turbo)

// Read current mode (from orchestrator)
let currentMode = engine.modeChangeCount > 0 ? "eco/balanced/turbo" : "balanced"
```

**Roadmap** — hardware telemetry accessor methods (`getHardwareThermal`, `getAvailableRamMb`, `getBatteryLevel`) are planned for a future release to support custom monitoring in application code.

## Thermal Throttling

Atheer continuously samples hardware telemetry at 1 Hz and adjusts the inference mode:

| Thermal State | Action                                    |
|---------------|-------------------------------------------|
| Nominal (<40°C)| Turbo mode — max throughput               |
| Warm (40-45°C)| Balanced mode — reduced speculation        |
| Critical (>45°C)| Eco mode — NGram cache, minimal power     |

On devices with insufficient RAM (<800 MB) or low battery (<20% on battery), the orchestrator also downgrades to Eco mode regardless of thermal state. Mode transitions are logged via `tracing::info!`.

## Hardware Telemetry

Atheer continuously samples device hardware state at 1 Hz through a dedicated background thread managed by `GenericMonitor` in the `atheer-hardware` crate. The latest `HealthSnapshot` is exposed to the orchestrator for real-time mode selection.

### Android (JNI Bridge)

On Android, telemetry is read via JNI calls through `atheer-hardware/src/android.rs`:

| Metric | Java API | Rust function |
|--------|----------|---------------|
| **Thermal headroom** | `ThermalManager.getThermalHeadroom()` (API 30+) | `thermal_headroom()` — returns time-to-throttle in seconds; `None` means unknown |
| **Available memory** | `ActivityManager.MemoryInfo.availMem` / `totalMem` | `memory_mb()` — returns available and total MB |
| **Battery level** | `BatteryManager.getIntProperty(BATTERY_PROPERTY_CAPACITY)` | `battery_info()` — returns level (0–100) and charging status |
| **Charging status** | `BatteryManager.getIntProperty(BATTERY_PROPERTY_IS_CHARGING)` | |

The JNI bridge stores the `JavaVM` and `Context` in `OnceLock` globals. Each sampling call attaches the current thread via `attach_current_thread()` (auto-detaches on drop). Your application **must** call `init_jni()` early during startup (e.g., `Application.onCreate()`) with the JVM reference and application context.

### iOS / macOS (objc2 — requires macOS)

iOS hardware telemetry reads thermal, memory, and battery state via `objc2` FFI:

| Metric | API | Rust function |
|--------|-----|---------------|
| **Thermal state** | `NSProcessInfo.processInfo.thermalState` | `read_thermal_state()` → `ThermalState` (Nominal/Fair/Serious/Critical) |
| **Available memory** | `os_proc_available_memory()` C FFI | `read_memory()` → `(available_mb, total_mb)` |
| **Total memory** | `NSProcessInfo.processInfo.physicalMemory` | |
| **Battery level** | `UIDevice.batteryLevel` (0.0–1.0) | `read_battery()` → `(level 0–100, is_on_battery)` |
| **Battery state** | `UIDevice.batteryState` (charging/discharging/full) | |

The `IosMonitor` struct spawns a dedicated 1 Hz sampling thread and implements the `HardwareMonitor` trait. Module gated behind `#[cfg(any(target_os = "ios", target_os = "macos"))]` — requires macOS build environment with Xcode CLI tools.

```bash
# Build requirement: Xcode Command Line Tools
xcode-select --install
```

### Health Snapshot → Mode Selection

```
┌─────────────────────────────┐
│     GenericMonitor          │
│  (1 Hz sampling thread)     │
├─────────────────────────────┤
│ Thermal → ThermalState      │
│ Memory  → MemoryStatus      │
│ Battery → PowerState        │
│            + timestamp      │
└──────────┬──────────────────┘
           │ Arc<Mutex<HealthSnapshot>>
┌──────────▼──────────────────┐
│ Orchestrator::select_mode() │
│  (consumes latest snapshot) │
└─────────────────────────────┘
```

## Testing

The workspace contains **~400 tests** across all crates, verified via `cargo test --workspace`:

| Crate | Test Count | Scope |
|-------|-----------|-------|
| `atheer-accel` | 29 | Backend creation, forward pass, fallback, quantization, probe order |
| `atheer-core` | 99+ | Model loading, KV cache, block manager, accuracy, security, lifecycle, streaming, session management, multi-turn conversation |
| `atheer-ffi` | 8 | Config roundtrip, backend type conversion, engine lifecycle |
| `atheer-hardware` | 6 | Monitor creation, sampling thread, health status edge cases |
| `atheer-memory-bank` | 40 | L1/L2/L3 cache, EncryptedStore (AES-256-GCM), handoff protocol, alignment scoring, VRAM monitoring |
| Integration (orchestrator) | ~5 | NGram cache, Eco mode, mode switching with telemetry |
| `atheer-fuzz` | 3 | Fuzz-resistant KV cache, token, config parsing |

A further 4 tests remain `#[ignore]` (structurally blocked — they use `unsafe` construct patterns that cannot be safely tested without a real model).

To run the full atheer-core test suite with a real model (including 10 model-dependent tests that otherwise skip gracefully):

```bash
# One-time download (~350 MB)
scripts/download-test-model.sh

# Run all integration tests
ATHEER_TEST_MODEL=./models/LFM2-700M-Q4_0.gguf cargo test -p atheer-core
```

**CI**: The `.github/workflows/ci.yml` workflow runs `cargo check`, lint, and unit tests on every push/PR to `main`. Model-dependent integration tests run on every push to `main` (plus schedule and manual dispatch) with a cached GGUF model.

## Benchmarking

Atheer includes a `perf-bench` binary for measuring throughput, energy, and thermal behavior:

```bash
# Basic throughput benchmark
cargo run -p perf-bench -- --model-path model.gguf --batch-sizes 1,4,8

# Sustained 5-minute test with thermal monitoring
cargo run -p perf-bench -- --model-path model.gguf --duration-secs 300

# Compare two runs
python tools/bench-compare.py before.json after.json
```

The binary outputs a machine-readable JSON report (`bench-report.json` by default) with per-batch and sustained power samples. Criterion bench harnesses are also available under `perf-bench/benches/` for CI integration.

```bash
cargo bench -p perf-bench
```

## Crates

| Crate | Description | Tests |
|-------|-------------|-------|
| `atheer-core` | Core inference engine (powered by `candle` and `tokenizers`) | 17 |
| `atheer-ffi` | FFI bindings via uniffi (Swift/Kotlin) | 8 |
| `atheer-accel` | Hardware acceleration backends (Metal, Vulkan, NNAPI, CoreML, CPU) | 29 |
| `atheer-orchestrator` | Mode selection, grammar sampling, and agent execution loop | integration |
| `atheer-hardware` | Platform hardware telemetry (thermal, memory, battery) | 6 |
| `atheer-memory-bank` | KV cache hierarchy (L1/L2/L3 with handoff) + `EncryptedStore` (AES-256-GCM) | 40 |
| `perf-bench` | Performance-per-watt benchmarking binary and model-dependent Criterion harnesses | 1 binary + 9 benches |
| `atheer-benches` (tests/benches) | Model-free Criterion microbenchmarks (kv_cache, ngram_cache, orchestrator) | 3 benches |

## Requirements

- Rust 1.75+
- iOS 15+ (Metal, CoreML detection)
- Android API 26+ for Vulkan (API 28+ recommended for NNAPI device discovery)
- NDK r29+ for NNAPI cross-compilation (`$ANDROID_NDK_HOME`)
- `cargo ndk` for Android builds
- macOS: Xcode Command Line Tools for iOS/macOS telemetry compilation (`objc2` FFI)
  ```bash
  xcode-select --install
  ```

## Completed Work

### KV Cache Encryption (V2/V3) ✅

AES-256-GCM encryption for L2/L3 KV cache snapshots at rest, completed July 2026:

1. **`EncryptedStore` struct** — wraps `L3CompressedStorage` with encrypt-on-write / decrypt-on-read. Format: `[12 B random nonce || AES-256-GCM ciphertext + tag]`. LZ4 compresses before encrypting, never reverses order.
2. **`MemoryBank` integration** — `l3` field changed to `Option<EncryptedStore>`, `encryption_key` stored as `Option<Box<[u8; 32]>>` for zeroize-on-drop, `set_l3_storage()` method for deferred initialization.
3. **Key resolution** — at `AtheerEngine::new()` time: `config.cache_encryption_key` (if 32 bytes) → ephemeral session key (if `checkpoint_dir` set) → `None` (L3 unavailable).
4. **`AtheerConfig.cache_encryption_key`** — `Option<Vec<u8>>` field (default `None`) for apps to provide a persistent key.
5. **6 unit tests** — encrypt/decrypt roundtrip, wrong key fails, corrupted file fails, nonce uniqueness, large payload, empty payload. All 40 memory-bank tests pass.

**Architecture distinction**: KV cache tier-3 snapshots use `EncryptedStore`. Engine-level model checkpoint save/restore continues using plain `L3CompressedStorage` (checkpoint data is not KV cache context).

### CoreML/ANE Production Inference ✅

The `CoreMLBackend` now supports real ANE inference via `candle-coreml` integration (behind the `coreml` feature flag). Key deliverables:

1. **`ANECompatibility` struct** — per-layer-type heuristics with model size ceiling (200M params), quantization whitelist (`q4_k_m`, `q4_k_s`, `f16`, `f32`), per-layer flags (matmul, embedding, silu, rms_norm, conv2d, add supported; softmax, layer_norm, gelu fall back to GPU), and M3+ enhanced support (RoPE, softmax, gelu).
2. **`CoreMLBackend::with_model()`** — cfg-gated constructor that loads an `.mlpackage` into `candle_coreml::CoreMLModel`.
3. **ANE→Metal→CPU fallback chain** — ANE path via `candle_coreml::CoreMLModel::forward()`, fallback to `candle_core::Device::Metal`, then CPU one-hot.
4. **`catch_unwind` protection** — ANE forward panics are caught gracefully.
5. **Caching** — compatibility computed at load time and stored on the `CoreMLBackend` instance.
6. **16 unit tests** — all passing.

**Remaining**: Create the `atheer-npu/candle-coreml` GitHub fork (upstream dep bump from 0.9.1 to 0.10.2 and API adaptation), then uncomment the git dep in `atheer-accel/Cargo.toml` and verify the `coreml` feature compiles end-to-end.

### Metal Backend Stability

The Metal backend (`atheer-accel/src/metal.rs`) panics on systems without a Metal GPU (virtualized macOS, CI). Root cause: `candle-core`'s `metal_if_available()` uses `Vec::swap_remove` on an empty device list. Fix requires upstream patch to `candle-core` or a `catch_unwind` wrapper in the backend.

### Production Readiness

- **NNAPI real device testing** — graph builder and compiled model tests are verified on non-Android (stubs). Real Android device testing needed to validate `ANeuralNetworksModel_addOperation` and `execute()` produce correct outputs.
- **iOS telemetry on-device** — `IosMonitor` works on macOS. Testing on physical iOS devices is needed to validate `UIDevice` and `NSProcessInfo` selectors behave as expected.
- **Cross-compilation CI** — add `cargo ndk` and `xcodebuild` build verification to CI pipeline.

## Vendored Dependencies

Atheer vendors specific upstream crates that require patches for stability or platform compatibility. These are managed via `git subtree` and live in-tree as workspace members with `[patch.crates-io]` entries in the root `Cargo.toml`.

### `candle-core`

The vendored `candle-core` crate at `candle-core/` includes a stability fix for the Metal backend — on systems without a Metal GPU (virtualized macOS, CI), `MetalDevice::new()` returns `Err` instead of panicking on an empty device list.

**Upstream:** `https://github.com/huggingface/candle`  
**Fork:** `github.com/achmadk/candle` (branch `patched-v0.10.2`)  
**Remote:** `candle-core-upstream`

**Update procedure:**

```bash
# Pull latest from the fork
git subtree pull --prefix=candle-core --squash candle-core-upstream crate-candle-core

# Re-apply the Metal stability patch if it was not in the pulled revision
# The fix lives in candle-core/src/metal_backend/mod.rs ~line 1927
```

After pulling, run `cargo check -p candle-core` to verify the vendored crate compiles, then `cargo test --workspace --exclude candle-coreml` to verify the full workspace.

### `candle-coreml`

The vendored `candle-coreml` crate at `candle-coreml/` provides Apple CoreML/ANE integration for inference. It is vendored from a fork that maintains compatibility with the workspace's Cargo.toml conventions (no downstream deps like `clap`, `tokio`, `hf-hub`, `git2`, etc.).

**Upstream:** `https://github.com/mazhewitt/candle-cormel`  
**Fork:** `github.com/achmadk/candle-coreml` (branch `main`)  
**Remote:** `candle-coreml-upstream`

**Update procedure:**

```bash
# Pull latest from the fork
git subtree pull --prefix=candle-coreml --squash candle-coreml-upstream main

# Re-apply Cargo.toml customizations:
# - Simplified description and metadata
# - Strip unused deps: clap, tokio, hf-hub, git2, tracing-subscriber, criterion, which
# - Keep only: candle-core, candle-nn, candle-transformers, tokenizers, anyhow,
#   serde, serde_json, rand, dirs, half, tracing, once_cell, glob, chrono,
#   objc2/objc2-foundation/objc2-core-ml/block2 (macOS), tempfile (dev-dep)
```

After pulling, run `cargo check -p candle-coreml` to verify the vendored crate compiles, then `cargo test -p candle-coreml` to run its unit tests.

## License

MIT OR Apache-2.0
