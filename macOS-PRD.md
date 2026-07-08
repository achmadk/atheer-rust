# macOS-Blocked Tasks — Product Requirements Document

**Status:** ✅ Complete — 21 + 13 OpenSpec tasks implemented across two changes
**Changes:** 
  - `openspec/changes/archive/2026-07-09-macos-blocked-tasks/` — initial implementation (21 tasks)
  - `openspec/changes/archive/2026-07-09-fix-remaining-gaps/` — ANE wiring + macOS battery fix (13 tasks)
**Blocked by:** Nothing. All macOS-requiring work has been delivered.

---

## Why

The atheer-rust engine had three implementation gaps that required macOS build tooling:

1. **iOS hardware telemetry** — `objc2` FFI for thermal, memory, battery monitoring
2. **CoreML/ANE real inference** — CoreML integration for on-device NPU acceleration
3. **Full NNAPI model compiler** — extending the FULLY_CONNECTED-only bridge to general NNAPI operation support

All development infrastructure previously ran on Linux CI, which lacked `objc2` compilation support, macOS SDK headers, and Xcode toolchain. With a macOS build environment available, all three gaps were closed.

---

## What Changed

### Core (`macos-blocked-tasks`)

- **iOS Hardware Telemetry (objc2 FFI)**: New `atheer-hardware/src/ios.rs` module with `objc2` bindings for `ProcessInfo.thermalState`, `os_proc_available_memory()`, `NSProcessInfo.physicalMemory`, `UIDevice.batteryLevel`/`batteryState`. `IosMonitor` implementing `HardwareMonitor`. Conditionally compiled behind `#[cfg(any(target_os = "ios", target_os = "macos"))]`.
- **CoreML/ANE integration**: `CoreMLBackend` with ANE compatibility detection, Metal GPU compute path as primary accelerator, and CPU fallback. Real ANE inference via vendored `candle-coreml` fork (bumped to `candle-core` 0.10.2).
- **NNAPI full model compiler**: Extended from FULLY_CONNECTED-only to support 9 operation types (ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE, FULLY_CONNECTED) via `NnapiGraphBuilder`.
- **No code changes on Linux**: All new modules are `#[cfg]`-gated to their target platform. Linux CI remains unaffected.

### Fixes (`fix-remaining-gaps`)

- **ANE wired as default forward pass**: Added `coreml_model_path: Option<String>` to `AtheerConfig`. When set, `BackendManager::with_coreml_model()` replaces `CoreMLBackend::new()` with `CoreMLBackend::with_model()` on Apple platforms, loading the `.mlpackage` for real ANE inference. `BackendManager::device()` returns CPU when ANE model is loaded (ANE handles device placement internally) and Metal when no model is set.
- **macOS battery panic fixed**: `read_battery()` now uses `AnyClass::get()` without `.expect()`, returning safe defaults `(0, true)` on macOS where `UIDevice` does not exist. The IosMonitor sampling thread stays alive on macOS.
- **Cfg gate corrections**: `#[cfg(target_os = "ios")]` gates on `BackendManager` expanded to `#[cfg(any(target_os = "ios", target_os = "macos"))]` so macOS development builds include CoreML and Metal backends.

---

## Impact

| File | Change |
|------|--------|
| `atheer-hardware/src/ios.rs` | New module — `IosMonitor` with `objc2` FFI (409 lines, 9 unit tests) — macOS-only build |
| `atheer-hardware/Cargo.toml` | Added `objc2` + `objc2-foundation` under `[target.'cfg(any(target_os = "ios", target_os = "macos"))'.dependencies]` |
| `atheer-accel/src/coreml.rs` | New module — `CoreMLBackend` with ANE detection, Metal acceleration, fallback chain |
| `atheer-accel/src/nnapi_ndk.rs` | Extended with additional operation codes and operand types (Linux+buildable) |
| `atheer-accel/src/nnapi.rs` | Added `NnapiGraphBuilder` op-to-NNAPI mapper dispatch with 9 operations |
| `atheer-accel/Cargo.toml` | Added `candle-coreml` path dep under `[target.'cfg(any(target_os = "ios", target_os = "macos"))'.dependencies]` |
| `Cargo.toml` (workspace) | Added `candle-coreml` as workspace member |
| `candle-coreml/` | Vendored fork (git subtree) — updated to `candle-core` 0.10.2 |
| `atheer-ffi/src/config.rs` | Added `coreml_model_path: Option<String>` field to `AtheerConfig` |
| `atheer-ffi/src/engine.rs` | Wired `coreml_model_path` through `AtheerEngine::new()` → `BackendManager::with_coreml_model()`; added `parse_param_count()` free function for model ID → parameter estimation |
| `atheer-accel/src/manager.rs` | Added `with_coreml_model()` builder, `coreml_model_path` field, ANE-aware `device()` (CPU when model loaded), `#[cfg]` expanded to include `macos` |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Phase ordering: telemetry → CoreML → NNAPI** | Telemetry has a clear API (`HardwareMonitor`) and no new crate deps beyond `objc2`; CoreML required vendoring a fork; NNAPI was lowest priority (FULLY_CONNECTED stub worked) |
| **`objc2` v0.6 + `objc2-foundation` for iOS FFI** | Maintains compatibility with existing `objc2` ecosystem; provides ergonomic `extern_category` and `rc` types; avoid `msg_send!` raw messaging where possible |
| **`candle-coreml` vendored via git subtree** | Forked to `github.com/achmadk/candle-coreml`, dep bumped to `candle-core` 0.10.2, vendored in-tree. Avoids external repo resolution issues |
| **Metal GPU as primary CoreML accelerator** | Upstream `candle-coreml` 0.9.x API drift from our vendored `candle-core` 0.10.2 made real ANE forward pass deferred; Metal GPU provides production acceleration in the interim |
| **NNAPI compiler as separate `NnapiGraphBuilder`** | Separation of concerns — `Executor` handles device management, `GraphBuilder` maps Candle ops to NNAPI `addOperation`; allows swapping backends without touching graph logic |
| **All new modules behind `#[cfg]` + `#[cfg_attr(not(...), allow(dead_code))`** | Keeps Linux CI green; platform-specific code is opt-in at compile time |
| **Naming: `IosMonitor` (not `iOSMonitor`)** | `IosMonitor` avoids Rust lint warnings about non-ASCII identifiers; consistent with crate naming conventions |
| **`coreml_model_path` as separate config field** | ANE requires `.mlpackage` format, distinct from `.gguf` model path. Explicit field makes the dependency clear at API level; avoids confusing semantics if a single `model_path` served both |
| **`device()` returns CPU when ANE model loaded** | `candle_coreml::CoreMLModel::forward()` handles device placement internally; CPU tensors are the correct input. Metal returned only when no ANE model is loaded |
| **`BackendManager::with_coreml_model()` as builder method** | Keeps dependency direction clean (accel ← core ← ffi); `BackendManager` never imports FFI-layer types |

---

## Actual Outcomes vs. Original Risks

| Risk | Outcome |
|------|---------|
| `objc2` v0.6 API drift from latest v0.7 | Pinned to v0.6; iOS telemetry compiles and tests pass on macOS |
| `candle-core` 0.10.2 vs `candle-coreml` ^0.9 semver gap | **Resolved**: Forked `candle-coreml` to `achmadk/candle-coreml`, bumped to `candle-core` 0.10.2, vendored via git subtree at `candle-coreml/` |
| NNAPI supports ~100+ operation codes; full coverage unrealistic | Delivered 9 operations: ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE, FULLY_CONNECTED — covers the highest-frequency ops |
| No on-device test harness for NNAPI/iOS | Documented `cargo ndk` / `xcodebuild` test commands; no CI integration until device runner is available |

---

## Specifications

### iOS Telemetry (`ios-telemetry`)

#### Requirement: iOS thermal state reading
The system SHALL read the current iOS thermal state via `ProcessInfo.thermalState` using `objc2` FFI and map it to the crate's `ThermalState` enum.

- **WHEN** `IosMonitor::sample()` is called
- **THEN** query `NSProcessInfo.processInfo.thermalState` and map:
  - `NSProcessInfoThermalStateCritical` → `ThermalState::Critical`
  - `NSProcessInfoThermalStateSerious` → `ThermalState::Throttled`
  - Other values → `ThermalState::Nominal`

#### Requirement: iOS available memory reading
The system SHALL read available memory via `os_proc_available_memory()` and total memory via `NSProcessInfo.physicalMemory`.

- **WHEN** `IosMonitor::sample()` is called
- **THEN** return available and total memory in megabytes

#### Requirement: iOS battery state reading
The system SHALL read battery level and charging state via `UIDevice.batteryLevel` and `UIDevice.batteryState`.

- **WHEN** `IosMonitor::sample()` is called
- **THEN** enable battery monitoring, read `batteryLevel` (0.0–1.0 → 0–100) and `batteryState` (charging if `Charging` or `Full`)
- **WHEN** battery monitoring was already disabled beforehand
- **THEN** restore the previous state after sampling

#### Requirement: IosMonitor implements HardwareMonitor
`IosMonitor` SHALL implement `HardwareMonitor`, sampling all three metrics in a single 1 Hz thread, returning `HealthSnapshot` with timestamp.

- **WHEN** `IosMonitor::new()` is called
- **THEN** spawn a 1 Hz sampling thread storing results in `Arc<Mutex<HealthSnapshot>>`
- **WHEN** `health()` is called but no sample collected yet
- **THEN** return `ThermalState::Nominal`, zeroed `MemoryStatus`, `PowerState { on_battery: true, level: 0 }`

#### Requirement: iOS telemetry tests
At least 6 unit tests covering creation, thermal/memory/battery conversion, fallback behavior, and snapshot freshness.

- **9 unit tests delivered** (exceeds the minimum)
- **WHEN** compiled on non-macOS, non-iOS targets
- **THEN** tests excluded via `#[cfg]`

### CoreML/ANE Inference (`coreml-ane-inference`)

#### Requirement: ANE hardware detection
The system SHALL detect Apple Neural Engine availability via sysctl and model compatibility.

- **WHEN** `AneCapability::detect()` is called
- **THEN** return `Available` / `Unavailable` / `NotSupported` based on `hw.optional.arm64` and CPU brand string
- **WHEN** a model is within supported constraints (≤200M params, whitelisted quantization)
- **THEN** `CoreMLBackend::is_compatible()` returns `true`

#### Requirement: Metal GPU compute as primary accelerator
`CoreMLBackend` SHALL accelerate inference via `candle-core` Metal device.

- **WHEN** Metal is available on the system
- **THEN** `forward()` delegates to `candle_core::Device::Metal` with tensor validation
- **WHEN** Metal is unavailable
- **THEN** fall back to CPU with latency measurement

#### Requirement: CoreML ANE forward pass (with vendored candle-coreml)
The vendored `candle-coreml` fork (at `candle-coreml/`) provides `candle_coreml::CoreMLModel` for real ANE inference. Integration is wired via the `coreml` feature flag on `atheer-accel`.

- **WHEN** building on macOS with `--features coreml`
- **THEN** `candle-coreml` is available as a dependency
- **WHEN** `CoreMLBackend::with_model()` is called with an `.mlpackage`
- **THEN** returns `Ok` with a `CoreMLBackend` ready for ANE-forward

#### Requirement: Fallback chain
- **WHEN** ANE forward panics (caught via `catch_unwind`)
- **THEN** fall back to `candle_core::Device::Metal` with `tracing::warn!`

### NNAPI Full Compiler (`nnapi-full-compiler`)

#### Requirement: NNAPI graph builder
`NnapiGraphBuilder` SHALL map Candle operations to NNAPI operation codes and construct a model graph via `ANeuralNetworksModel_addOperation`.

- **WHEN** `NnapiGraphBuilder::new()` is called
- **THEN** create and hold a new `ANeuralNetworksModel`
- **WHEN** a Candle op is ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE, or FULLY_CONNECTED
- **THEN** map to the corresponding `ANEURALNETWORKS_*` code and call `addOperation`
- **WHEN** a Candle op has no NNAPI equivalent
- **THEN** return `Err(AccelError::UnsupportedOperation(...))`

#### Requirement: Model compilation from graph
`.finish()` SHALL call `ANeuralNetworksModel_finish`. `.compile()` SHALL return an `NnapiCompiledModel` handle.

- **WHEN** `.finish()` succeeds and `.compile()` is called
- **THEN** return a compilation handle ready for `ANeuralNetworksExecution_compute`

#### Requirement: Integration tests
Tests SHALL cover ADD, MUL, SOFTMAX, RELU, builder patterns, and compiled model lifecycle — 10 tests delivered.

- **WHEN** run on Android via `cargo ndk -t arm64-v8a --platform 26 test`
- **THEN** verify correct model graph construction for each operation

---

## Task List

### Phase 1: iOS Telemetry — objc2 FFI Setup

- [x] 1.1 Add `objc2` and `objc2-foundation` to `atheer-hardware/Cargo.toml` under `[target.'cfg(any(target_os = "ios", target_os = "macos"))'.dependencies]`
- [x] 1.2 Create `atheer-hardware/src/ios.rs` module with `#[cfg]` gate, public `IosMonitor` struct, and module skeleton
- [x] 1.3 Implement thermal state reading via `objc2` `NSProcessInfo.processInfo.thermalState` → `ThermalState` mapping
- [x] 1.4 Implement memory reading via `os_proc_available_memory()` C FFI + `NSProcessInfo.physicalMemory` → `MemoryStatus`
- [x] 1.5 Implement battery reading via `objc2` `UIDevice.current.batteryLevel`/`batteryState` → `PowerState`
- [x] 1.6 Implement `IosMonitor` struct with 1 Hz sampling thread, `HardwareMonitor` trait impl
- [x] 1.7 Write 6+ unit tests for iOS telemetry — 9 unit tests delivered, all passing on macOS

### Phase 2: CoreML/ANE Real Inference

- [x] 2.1 Fork `candle-coreml` to `github.com/achmadk/candle-coreml`, bump dep to `candle-core` 0.10.2, vendor via git subtree at `candle-coreml/`
- [x] 2.2 Add `candle-coreml` as workspace member and macOS/iOS-only path dependency in `atheer-accel/Cargo.toml`
- [x] 2.3 Implement ANE hardware detection via sysctl (`AneCapability` enum)
- [x] 2.4 Implement Metal GPU compute path as primary accelerator with tensor validation
- [x] 2.5 Add tests for ANE detection, Metal availability, fallback chain, model compatibility — 16+ tests delivered

### Phase 3: NNAPI Full Model Compiler

- [x] 3.1 Create `NnapiGraphBuilder` struct in `nnapi.rs` with `ANeuralNetworksModel` management
- [x] 3.2 Implement operation mapping for 9 NNAPI operation codes (ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE, FULLY_CONNECTED)
- [x] 3.3 Implement operand ordering and type validation for each mapped operation
- [x] 3.4 Implement `.finish()` → `ANeuralNetworksModel_finish` and `.compile()` → `NnapiCompiledModel`
- [x] 3.5 Write integration tests for NNAPI graph builder — 10 tests: op codes, builder, validation, ADD/MUL/SOFTMAX/RELU, compiled model

### Phase 4: Documentation

- [x] 4.1 Document macOS build requirements in README.md (Xcode, `objc2`, CoreML.framework)
- [x] 4.2 Document NNAPI compiler current operation coverage in `nnapi.rs` module docs
- [x] 4.3 Add "Remaining Work" section to README covering CoreML ANE, Metal stability, production readiness

---

## Resolved Gaps

The two implementation gaps below were addressed in change `openspec/changes/archive/2026-07-09-fix-remaining-gaps/` (13 tasks, all complete, change archived):

1. **ANE forward pass wired as default** — Added `coreml_model_path: Option<String>` to `AtheerConfig`. When set, `BackendManager` uses `CoreMLBackend::with_model()` via `with_coreml_model()` builder, activating the ANE forward pass. Metal GPU remains the fallback when `.mlpackage` is not provided. The ANE path requires a CoreML-converted `.mlpackage` model file (separate from the `.gguf` model path — `candle_coreml::CoreMLModel` processes the ANE compute graph independently, while the main inference path uses the GGUF model for logit decoding on the selected device).
2. **macOS battery panic fixed** — `read_battery()` now checks `AnyClass::get(c"UIDevice")` at runtime and returns safe defaults `(0, true)` on macOS instead of panicking when `UIDevice` is absent. Test `test_read_battery_macos_fallback` verifies the fallback produces valid `(level ≤ 100, bool)` values on any platform. The IosMonitor sampling thread no longer silently dies on macOS.

## Remaining Gaps (platform testing, not code)

1. **NNAPI real device testing** — Graph builder and compiled model tests are verified on non-Android (stubs). Real Android device testing needed to validate `ANeuralNetworksModel_addOperation` and `execute()` produce correct outputs.
2. **iOS telemetry on-device** — `IosMonitor` works on macOS. Testing on physical iOS devices is needed to validate `UIDevice` and `NSProcessInfo` selectors behave as expected (battery path now safe via macOS fallback).

---

*Change `macos-blocked-tasks` (21 tasks) delivered the initial implementation. Change `fix-remaining-gaps` (13 tasks) resolved the ANE wiring and macOS battery panic — both archived under `openspec/changes/archive/`.*
