# Atheer-Rust Progress Report

> Generated: 2026-10-22
> Scope: 16 workspace packages, ~31.7K first-party Rust lines, 820 tests
> Status: Functional. Remaining work is concentrated in bindings generation, real-device validation, and benchmark baselines.

---

## 1. Current Snapshot

| Check | Result |
|-------|--------|
| `cargo build --workspace` | Passes |
| `cargo test --workspace` | 820 passed, 0 failed, 8 ignored |
| `cargo clippy --workspace -- -D warnings` | Clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | Clean |
| `cargo fmt --all -- --check` | Clean |
| Rust MSRV | 1.87 |
| `wgpu` / `naga` | 30.0.1 |
| UniFFI | 0.32 (proc-macro mode) |
| Android NDK (CI) | 30.0.16248370 |

`cargo build --workspace`, both clippy modes, and the full test suite are green. CI still runs only the default-target lint job; adding `--all-targets` is a low-cost follow-up.

---

## 2. Crate Status

| Package | First-party Rust | Tests | Status |
|---------|------------------|-------|--------|
| `atheer-core` | 13,108 | 285 (3 ignored) | Inference, KV cache, security, guardrails, privacy, sandbox bridge |
| `atheer-orchestrator` | 3,413 | 84 | Mode selection, grammar, agent loop |
| `atheer-memory-bank` | 2,034 | 40 | L1/L2/L3 cache, encrypted storage, handoff |
| `atheer-hardware` | 1,293 | 8 | Android JNI and Apple `objc2` telemetry |
| `atheer-accel` | 5,976 | 58 | Metal, Vulkan, NNAPI, CoreML, CPU, auto-selection |
| `atheer-ffi` | 3,639 | 45 | UniFFI Swift/Kotlin boundary |
| `atheer-gpu-shard` | 267 | 0 | Android IsolatedService JNI shard |
| `atheer-encrypt` | 141 | 0 | GGUF encryption CLI |
| `atheer-bindgen` | 76 | 0 | Dead UDL-based generator |
| `atheer-integration-tests` | 635 | 38 | Cross-crate integration/property tests |
| `atheer-fuzz` | 94 | 4 | Fuzz smoke targets |
| `perf-bench` | 1,040 | 2 + 10 benches | Perf-per-watt binary + Criterion harnesses |
| `atheer-benches` | - | 3 benches | Model-free microbenchmarks |
| `candle-core` | 56,234 | 170 | Vendored upstream + Atheer Vulkan/NNAPI/Metal paths |
| `candle-coreml` | 20,506 | 16 | Vendored CoreML/ANE integration |
| `candle-transformers` | 83,352 | 23 | Vendored upstream + KV-cache patches |

First-party line count excludes the three vendored Candle crates.

---

## 3. Test Status

```text
Crate                  Passed   Failed   Ignored
-------------------------------------------------
atheer-accel               58        0         0
atheer-core               285        0         3
atheer-ffi                 45        0         0
atheer-fuzz                 4        0         0
atheer-hardware             8        0         0
integration tests          38        0         0
atheer-memory-bank         40        0         0
atheer-orchestrator        84        0         0
perf-bench                  2        0         0
candle-core               170        0         0
candle-coreml              16        0         0
candle-transformers        23        0         0
doc tests                  47        0         5
-------------------------------------------------
Total                     820        0         8
```

Ignored tests:

- 3 `atheer-core` tests require a real GGUF model or an unsupported KV-cache snapshot path.
- 5 doc tests are ignored by their examples.

`cargo test --workspace --lib` is the CI unit-test command. The full `cargo test --workspace` run above includes integration and doc tests.

---

## 4. Recently Completed Code Fixes

### 4.1 Backend auto-selection

`BackendManager::new()` registers CPU first so the manager always has a usable default. `best_available()` previously returned the first available backend, which meant CPU always won. It now skips CPU during the accelerator pass and only falls back to CPU when no accelerator is available.

Regression coverage: `test_best_available_prefers_accelerator_over_cpu`.

### 4.2 CoreML feature forwarding

`atheer-ffi` now exposes a `coreml` feature that forwards to `atheer-accel/coreml`. A plain FFI build can now compile the real CoreML/ANE path with:

```bash
cargo check -p atheer-ffi --features coreml
```

The `ane_is_available()` cfg fallback was also fixed so enabling `coreml` on a non-Apple host compiles.

### 4.3 Android CI NDK

The Android workflow installed NDK `30.0.16248370` but pointed `ANDROID_NDK_HOME` at `27.3.13750724`. All four `ANDROID_NDK_HOME` entries now match the installed NDK.

### 4.4 Earlier dependency work

The workspace was adapted to the current dependency set, including `rand` 0.9.5 / `rand_distr` 0.5.1 for `candle-core`, `zip` 9.0.5 API changes, `reqwest` 0.13, and `wgpu` / `naga` 30.0.1.

### 4.5 All-targets warning cleanup

`cargo clippy --workspace --all-targets -- -D warnings` is clean. The cleanup replaced the deprecated `criterion::black_box` import with `std::hint::black_box`, moved test-only config construction to struct update syntax, and resolved the remaining needless-borrow, identity-op, range-contains, and test-style lints.

---

## 5. Remaining Tasks

### P1 — Blocking or high-value gaps

| Area | Issue |
|------|-------|
| Bindings | `atheer-bindgen` still expects a `.udl` file; the project uses UniFFI proc-macro mode. `generate-bindings.sh` remains a no-op. |
| Benchmarks | `BENCHMARKS.md` baselines are still TBD; no reference-hardware numbers are checked in. |
| NNAPI | Real Android device validation is still required for graph execution and output correctness. |
| CoreML / Metal / iOS telemetry | CoreML has unit tests; Metal and iOS telemetry compile on Apple targets. Physical iOS device validation remains. |
| iOS CI | No `xcodebuild` / `aarch64-apple-ios` build job. |
| Sandbox | The bridge lifecycle and Android shard JNI exist; full isolated-process spawn integration is incomplete. |
| Vulkan tests | `candle-core/tests/vulkan_tests.rs` is Android-only and returns early when no Vulkan device is available, so Linux CI does not exercise GPU parity. |

### P2 — Hygiene and polish

| Area | Issue |
|------|-------|
| Bindings | Pre-generated Swift bindings may be stale; Kotlin bindings are handwritten. |
| Fuzz | Four smoke targets only; no corpus, dictionary, or CI fuzz run. |
| Git hygiene | Generated bindings, model files, and bench outputs are not all covered by `.gitignore`. |
| Docs | Archived OpenSpec changes and `anchored_summary.md` are historical snapshots; current status is in this file and `README.md`. |

---

## 6. CI Status

| Job | Runner | Command / Coverage | Status |
|-----|--------|--------------------|--------|
| `cargo check` | Ubuntu, macOS | `cargo check --workspace` | Local run green; CI job configured |
| Lint & format | Ubuntu, macOS | `cargo fmt --all --check`, `cargo clippy --workspace -- -D warnings` | Local run green; CI job configured (default targets only) |
| Unit tests | Ubuntu, macOS | `cargo test --workspace --lib` | Local full suite green; CI runs `--lib` |
| Model tests | Ubuntu | `cargo test -p atheer-core` with cached GGUF model | CI job configured; requires cached model |
| Build perf-bench | Ubuntu | Release binary + Criterion `--no-run` | CI job configured |
| Android | Ubuntu | NDK 30.0.16248370, `cargo ndk` shard build, Vulkan/NNAPI candle tests, Gradle | CI job configured; NDK path fixed |

Both default and all-targets clippy runs pass with `-D warnings` locally. The CI lint job currently checks only default targets; adding `--all-targets` is a low-cost follow-up.

---

## 7. Toolchain and Dependency Snapshot

| Dependency | Version / Note |
|------------|----------------|
| Rust | 1.87 MSRV |
| `wgpu` / `naga` | 30.0.1 |
| `uniffi` | 0.32 |
| `reqwest` | 0.13 |
| `zip` | 9.0.5 |
| `rand` (candle-core) | 0.9.5 |
| `rand_distr` (candle-core) | 0.5.1 |
| `rand` (atheer-memory-bank, candle-coreml) | 0.10 |
| `candle-coreml` | 0.3.1, in-tree path dependency |
| Android NDK | 30.0.16248370 |

`Cargo.lock` intentionally contains both `rand` 0.9 and 0.10.

---

## 8. Quick Commands

```bash
# Build and test
cargo build --workspace
cargo test --workspace
cargo test --workspace --lib

# Default CI lint
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings

# All-targets lint
cargo clippy --workspace --all-targets -- -D warnings

# CoreML / FFI
cargo check -p atheer-ffi --features coreml
cargo test -p atheer-accel --features coreml -- coreml   # macOS

# Android
cargo ndk -t arm64-v8a --platform 26 build -p atheer-ffi
cargo ndk -t x86_64 build -p candle-core --features nnapi --tests

# Benchmarks
cargo bench -p perf-bench
cargo bench -p atheer-benches

# Fuzz
cd fuzz && cargo fuzz run fuzz_gguf_header
```

---

## 9. OpenSpec and Documentation State

- Active changes: 0
- Archived changes: 2 (`update-candle-upstream-0-11-0`, `update-wgpu-30`)
- Specs: 0

The archived changes are historical records and should not be rewritten. `README.md`, `CLAUDE.md`, `BENCHMARKS.md`, and this report describe the current tree.

---

## 10. Assessment

The core engine, memory hierarchy, security layers, FFI surface, and test suite are in good shape. The main production risks are now integration and validation gaps rather than missing subsystems:

1. Regenerate bindings through the supported UniFFI path.
2. Validate NNAPI, CoreML, Metal, and iOS telemetry on real devices.
3. Add iOS build verification to CI.
4. Populate benchmark baselines.
5. Add `--all-targets` to the CI lint job to keep the warning set at zero.
