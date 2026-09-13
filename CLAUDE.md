# Atheer-Rust

Mobile-first, NPU-first LLM inference engine. iOS + Android. Safe Rust.

Cross-platform local inference engine for large language models targeting Apple Neural Engine (ANE) and Android NNAPI, with Metal/Vulkan/CPU fallback. Includes predictive thermal management, hierarchical KV cache, encrypted checkpoints, grammar-constrained decoding, prompt-guardrail pipeline, and a built-in agent loop.

---

## Architecture

```
ios/  android/          ── Mobile SDK wrappers (Swift, Kotlin)
  │       │
  └── atheer-ffi ──────── UniFFI boundary (AtheerEngine, AtheerConfig, AtheerError)
        │
  ┌─────┴──────────────────────────────────┐
  │            atheer-core                  │
  │  InferenceEngine, Model, Tokenizer      │
  │  SecurityAudit, PiiRedactor             │
  │  GuardrailDetector (L1/L2/L3)          │
  │  ModelRegistry, ModelVerifier           │
  │  CertPinner, SafeContent                │
  ├────────────────────────────────────────┤
  │          atheer-orchestrator            │
  │  Turbo/Balanced/Eco mode selection     │
  │  Predictive thermal model              │
  │  Grammar/JSON/Tools sampler            │
  │  PerfCalibrator                        │
  │  Agent loop                            │
  ├────────────────────────────────────────┤
  │          atheer-accel                   │
  │  MetalBackend, CoreMLBackend           │
  │  VulkanBackend, CpuBackend             │
  │  BackendManager, PciValidator          │
  │  SandboxedGpuBridge                    │
  ├────────────────────────────────────────┤
  │      atheer-memory-bank                │
  │  L1/L2/L3 tiered KV cache              │
  │  EncryptedStore (AES-256-GCM)          │
  │  Checkpoint snapshot/restore           │
  ├────────────────────────────────────────┤
  │      atheer-hardware                   │
  │  iOS: thermal/charge/memory via objc2  │
  │  Android: thermal/battery via JNI      │
  └────────────────────────────────────────┘

  ┌─ vendored forks ───────────────────────┐
  │  candle-core (NNAPI backport)          │
  │  candle-transformers (kv_cache patch)  │
  │  candle-coreml (CoreML bridge)         │
  └────────────────────────────────────────┘
```

### Crate Purposes

| Crate | Key Types | Role |
|-------|-----------|------|
| `atheer-ffi` | `AtheerEngine`, `AtheerConfig`, `AtheerError` | UniFFI boundary. Exposes ~20 methods to Swift/Kotlin. Maps `AtheerCoreError` → `AtheerError`. |
| `atheer-core` | `InferenceEngine`, `Model`, `Tokenizer`, `SamplingConfig`, `SecurityAudit`, `PiiRedactor`, `GuardrailDetector`, `ModelVerifier`, `CertPinner` | Core inference. Guardrail pipeline (L1/L2/L3). Model load/verify. Certificate pinning. |
| `atheer-orchestrator` | `Orchestrator`, `ModeSelector`, `ThermalModel`, `GrammarSampler`, `JsonGrammar`, `PerfCalibrator`, `AgentLoop` | Mode switching (Turbo/Balanced/Eco). Prediction-based thermal. Structured output. Agent loop. |
| `atheer-accel` | `MetalBackend`, `CoreMLBackend`, `VulkanBackend`, `CpuBackend`, `BackendManager`, `SandboxedGpuBridge` | Hardware acceleration backends. Backend auto-selection. Sandboxed GPU execution. |
| `atheer-memory-bank` | `MemoryBank`, `L1CompressedStorage`, `L2CompressedStorage`, `L3CompressedStorage`, `EncryptedStore`, `PolicySelector` | Tiered KV cache. Compression. Encrypted checkpoints. |
| `atheer-hardware` | `HardwareMonitor`, `ThermalState`, `BatteryState` | Platform telemetry (iOS via `objc2`, Android via JNI). |

---

## Build & Test

### Workspace (all Atheer crates)

```bash
# Full workspace build (—features coreml on macOS for CoreML/ANE)
cargo build --workspace

# Run all workspace tests (817 across the workspace)
cargo test --workspace

# Run specific crate tests
cargo test -p atheer-core
cargo test -p atheer-orchestrator
cargo test -p atheer-memory-bank
cargo test -p atheer-ffi

# Build with CoreML support (macOS only)
cargo build --features coreml
```

### Vendored Forks

```bash
# candle-core (vendored, NNAPI backport)
cargo check -p candle-core              # Must pass on all platforms
cargo check -p candle-core --target aarch64-linux-android  # NNAPI build target

# candle-coreml (vendored, CoreML bridge)
cargo check -p candle-coreml            # macOS only for full typecheck

# candle-transformers (vendored, kv_cache_snapshot/kv_cache_restore patched)
cargo check -p candle-transformers
```

### Specific Test Invocations

```bash
# Guardrail tests (42 unit + 59 curated test suite)
cargo test -p atheer-core guardrail

# Sandbox tests
cargo test -p atheer-core sandbox

# Memory bank + checkpoint tests (40 tests)
cargo test -p atheer-memory-bank

# Privacy mode tests
cargo test -p atheer-core privacy

# Certificate pinning tests
cargo test -p atheer-core cert_pinner

# Model verification tests (hash + signature)
cargo test -p atheer-core model_verifier

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --check
```

### Supported Platforms

| Target | Primary Backend | Test Status |
|--------|----------------|-------------|
| `aarch64-apple-ios` | ANE → Metal → CPU | ✅ CI |
| `aarch64-linux-android` | NNAPI → Vulkan → CPU | ✅ CI |
| `aarch64-apple-darwin` | ANE → Metal → CPU (dev) | ✅ Dev |
| `x86_64-apple-darwin` | Metal → CPU | ✅ Dev |
| Other (Windows, Linux x86) | CPU only | Compiles, untested |

---

## Code Conventions

### Error Handling

```rust
// atheer-core uses typed AtheerCoreError with thiserror
#[derive(Error, Debug)]
pub enum AtheerCoreError {
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),
    #[error("Device mismatch: expected {expected}, got {actual}")]
    DeviceMismatch { expected: String, actual: String },
    #[error("GGUF header magic mismatch: expected GGUF, got {actual:?}")]
    InvalidMagic { actual: [u8; 4] },
    // ... 15+ variants
}

pub type Result<T> = std::result::Result<T, AtheerCoreError>;

// FFI layer maps to AtheerError (UniFFI-compatible, String-only fields)
// see atheer-ffi/src/error.rs: map_core_error()

// candle-core uses its own Error type with .bt() backtrace and .context()
// see candle-core/src/error.rs
```

Error handling rules:
- `AtheerCoreError` for all domain errors in `atheer-core`/`atheer-orchestrator`/`atheer-memory-bank`
- `AtheerError` at FFI boundary (string fields only — UniFFI constraint)
- `AtheerCoreError::Timeout { elapsed_ms, tokens_generated }` for generation timeout
- Use `map_err(|e| AtheerError::ModelLoadFailed { msg: format!("{e}") })` at FFI boundary
- candle-core errors use `.bt()` for backtrace capture, `.context()` for wrapping

### Conditional Compilation

```rust
// Platform gating — ALWAYS use exact cfg() not feature flags for platform
#[cfg(target_os = "ios")]
#[cfg(target_os = "android")]

// Backend capability gating
#[cfg(feature = "coreml")]            // CoreML support (macOS/iOS)
#[cfg(all(feature = "nnapi", target_os = "android"))]  // NNAPI + Android

// Privacy-gated logging
// trace_if_ok! macro: emits tracing only when should_log() is true
trace_if_ok!(self.should_log(), info, target: "atheer::engine", "message {}", val);
// should_log() returns true for Normal/Audited, false for Ephemeral
```

### Naming

- `Atheer*` prefix for public types exposed through FFI (AtheerEngine, AtheerConfig, AtheerError, AtheerCoreError)
- `camelCase` for Swift/Kotlin generated methods (initialize, generateSync, pollStreamToken)
- `snake_case` for internal Rust
- Backend types: `*Backend` trait + implementations (`MetalBackend`, `CoreMLBackend`, `VulkanBackend`)

### Testing Patterns

```rust
// Unit tests inline in source modules
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxx() {
        // ...
    }
}

// Integration tests in tests/ directory (atheer-core, atheer-orchestrator, etc.)
// Test data in tests/test_data/ (e.g., s4_guardrails_test_suite.json)
// Proptest for fuzz-like property testing
```

---

## Key Design Decisions

1. **NPU-first, not GPU-first**: The engine probes for Neural Engine/NNAPI first, falls back to Metal/Vulkan/CPU. This is the inverse of most inference engines.

2. **Inference engine pattern for CoreML**: CoreML is treated as an inference backend (accepts CPU/Metal tensors, runs forward pass), not as a Candle device type. This avoids the complexity of making CoreML a first-class `Device` variant.

3. **Tiered KV cache (L1/L2/L3)**: L1 = hot (RAM, uncompressed), L2 = warm (RAM, compressed), L3 = cold (disk, compressed, optionally encrypted). Checkpoints persist across app restarts.

4. **Sandboxed GPU bridge**: Forward passes can be routed through an isolated subprocess with a state machine (Unprovisioned → Standby → Ready → Fallback). Crash in sandbox → falls back to in-process, never crashes the app.

5. **Guardrail pipeline**: Three-layer defense (L1: fast heuristics <100μs, L2: token analysis <5ms, L3: output guard <100μs). Configurable levels (None/Basic/Balanced/Strict). Hot-reloadable pattern files.

6. **Predictive thermal management**: `ThermalModel` predicts temperature 15s ahead using workload + recent readings. `ModeSelector` uses hysteresis to prevent mode oscillation.

7. **Privacy-first**: `PrivacyMode` enum (Normal/Ephemeral/Audited) controls logging and persistence at runtime. Ephemeral mode suppresses all info/warn/debug logs (errors still fire), disables L3 cache persistence, and skips crash log file writes.

---

## Current State

### Completed (July 2026)

- **S1** — Model file encryption (AES-256-GCM for GGUF/.mlpackage, CLI tool)
- **S2+S3** — Ed25519 signature verification, SHA-256 hash verification
- **S4** — Prompt injection guardrails (L1/L2/L3 detection pipeline, 42 tests)
- **S5** — Model file encryption decryption pipeline
- **S6** — Pre-allocation header gate (GGUF validation before Vec allocation, 18 tests)
- **S7** — TLS certificate pinning (rustls PinningVerifier, dual Amazon+HF pins)
- **S8** — Sandboxed GPU execution (state machine, pre-warm, crash escalation)
- **V1** — Configurable privacy mode (Normal/Ephemeral/Audited)
- **V2** — KV cache checkpoint encryption (EncryptedStore + AES-256-GCM)
- **V3** — Checkpoint lifecycle (FFI methods for background/foreground/terminate)
- **R1** — Draft speculation engine (load_draft, generate_speculative, acceptance loop)
- **P2** — Continuous runtime calibration (PerfCalibrator adjusts depth/temperature/cache)
- **P3** — KV cache checkpoint persistence (LZ4 snapshots, L3 orphan sweep)
- **P5** — ANE compilation pre-heat (background thread, OnceLock swap)

### Known Issues

- 3 integration tests ignored (require real GGUF model download)
- ~10 compiler warnings (unused imports, dead code across platform-gated paths)
- `candle-core` vendored + heavily patched — upstream sync is manual
- `candle-coreml` crate has ~2263-line `qwen.rs` — refactoring in progress (see its CLAUDE.md)

---

## Security Model

```
┌─ Model Load ──────────────────────────────────────────┐
│ 1. GGUF header validator (S6) — no pre-allocation     │
│ 2. SHA-256 hash verification (S2+S3) — streaming hash │
│ 3. Ed25519 signature verification (S2+S3)              │
│ 4. Model decryption (S1+S5) — AES-256-GCM             │
└───────────────────────────────────────────────────────┘
┌─ Inference ───────────────────────────────────────────┐
│ 5. Guardrail detector (S4) — L1/L2/L3 pipeline        │
│ 6. PII redactor — regex-based PII masking             │
│ 7. Sandboxed GPU (S8) — isolated subprocess           │
└───────────────────────────────────────────────────────┘
┌─ Data at Rest ────────────────────────────────────────┐
│ 8. Encrypted KV cache (V2) — AES-256-GCM per snapshot │
│ 9. Ephemeral mode (V1) — no persistence               │
│ 10. Certificate pinning (S7) — rustls PinningVerifier │
└───────────────────────────────────────────────────────┘
```

---

## FFI Boundary Constraints

- UniFFI v0.27 — `uniffi::Error` derives for FFI error types
- All FFI error fields must be `String` (no structured enum variants across FFI)
- `map_core_error()` in `atheer-ffi/src/error.rs` converts typed errors to flat strings
- FFI methods are synchronous (blocking) — generation streams via `pollStreamToken`/`streamDone`
- AtheerConfig struct exposed to Swift/Kotlin with optional fields for model path, encryption keys, guardrail settings
- Re-generate Swift/Kotlin bindings from the .udl file after adding/changing FFI methods

---

## Cross-Cutting Concerns

### Privacy Mode Behavior

| Aspect | Normal | Ephemeral | Audited |
|--------|--------|-----------|---------|
| Info/warn/debug logs | ✅ | ❌ (errors only) | ✅ |
| Crash file writes | ✅ | ❌ (counter only) | ✅ |
| L3 checkpoint persist | ✅ | ❌ (encryption_key = None) | ✅ |
| Crash report scrubbing | ✅ | N/A | ❌ (full detail) |

### Backend Selection Priority

1. CoreML/ANE (`--features coreml`, macOS/iOS only)
2. NNAPI (`--features nnapi`, Android only)
3. Metal (Apple GPU)
4. Vulkan (Android GPU, desktop)
5. CPU (universal fallback)

---

## CLI Tools

```bash
# Encrypt a GGUF model file
cargo run -p atheer-encrypt -- --method aes-256-gcm --input model.gguf --output model.enc

# Benchmarks
cargo bench -p bench-crate

# Fuzz (cargo-fuzz)
cd fuzz && cargo fuzz run fuzz_gguf_header
```

---

## Important Files

| File | Purpose |
|------|---------|
| `atheer-ffi/src/engine.rs` | Main FFI engine — initialize, generate_sync, generate_stream |
| `atheer-ffi/src/error.rs` | FFI error mapping (`map_core_error()`) |
| `atheer-core/src/error.rs` | `AtheerCoreError` enum (15+ variants) |
| `atheer-core/src/privacy.rs` | `PrivacyMode` enum, `should_log()` helper |
| `atheer-core/src/guardrails/` | L1/L2/L3 guardrail detector modules |
| `atheer-core/src/cert_pinner.rs` | TLS certificate pinning (rustls PinningVerifier) |
| `atheer-core/src/safe_content.rs` | GGUF pre-allocation header gate |
| `atheer-core/src/sandbox/` | Sandboxed GPU bridge module |
| `atheer-memory-bank/src/` | L1/L2/L3 tiered cache + EncryptedStore |
| `atheer-orchestrator/src/calibrator.rs` | PerfCalibrator — dynamic parameter adjustment |
| `atheer-accel/src/` | Backend implementations (Metal/CoreML/Vulkan/CPU) |
| `atheer-hardware/src/` | Platform telemetry (iOS objc2, Android JNI) |
| `Cargo.toml` | Workspace members, patched crates, dependencies |

## Out of Scope

`candle-core/` and `candle-coreml/` have their own CLAUDE.md files. Do not modify them without consulting those files.
