# macOS-Blocked Tasks — Product Requirements Document

**Status:** Draft — work cannot begin until a macOS build environment is available
**Blocks:** 8 remaining OpenSpec tasks (63/71 complete across 7 crates)
**Related change:** `openspec/changes/macos-blocked-tasks/`

---

## Why

The atheer-rust engine has three remaining implementation gaps that require macOS build tooling:

1. **iOS hardware telemetry** — `objc2` FFI for thermal, memory, battery monitoring
2. **CoreML/ANE real inference** — `candle-coreml` integration for on-device NPU acceleration
3. **Full NNAPI model compiler** — extending the FULLY_CONNECTED-only bridge to general NNAPI operation support (not macOS-blocked, but deferred)

All development infrastructure currently runs on Linux CI, which lacks `objc2` compilation support, macOS SDK headers, and Xcode toolchain. This PRD establishes a phased plan so the work can be picked up immediately when a macOS environment becomes available.

---

## What Changes

- **Phase 1 — iOS Hardware Telemetry (objc2 FFI)**: New `atheer-hardware/src/ios.rs` module with `objc2` bindings for `ProcessInfo.thermalState`, `os_proc_available_memory()`, `NSProcessInfo.physicalMemory`, `UIDevice.batteryLevel`/`batteryState`. `iOSMonitor` implementing `HardwareMonitor`. Conditionally compiled behind `#[cfg(target_os = "ios")]`.
- **Phase 2 — candle-coreml integration**: Real ANE inference via `candle-coreml` crate (blocked by version mismatch: `candle-core` 0.10.2 vs `candle-coreml` ^0.9). Requires macOS build environment for compilation and testing.
- **Phase 3 — NNAPI full model compiler**: Extend the current FULLY_CONNECTED-only NNAPI bridge to support additional operation types (ADD, MUL, SOFTMAX, LOGISTIC, RELU, etc.) with a Candle-op-to-NNAPI-op mapper.
- **No code changes on Linux**: All new modules are `#[cfg]`-gated to their target platform. Linux CI remains unaffected.

---

## Impact

| File | Change |
|------|--------|
| `atheer-hardware/src/ios.rs` | New module — `iOSMonitor` with `objc2` FFI (macOS-only build) |
| `atheer-hardware/Cargo.toml` | Add `objc2` + `objc2-foundation` under `[target.'cfg(...)'.dependencies]` |
| `atheer-accel/src/nnapi_ndk.rs` | Extend with additional operation codes and operand types (Linux+buildable) |
| `atheer-accel/src/nnapi.rs` | Add `NnapiGraphBuilder` op-to-NNAPI mapper dispatch (Linux+buildable) |
| `Cargo.toml` (workspace) | Conditional `candle-coreml` dependency (macOS-only) |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Phase ordering: telemetry → CoreML → NNAPI** | Telemetry has a clear API (`HardwareMonitor`) and no new crate deps beyond `objc2`; CoreML requires resolving a version mismatch first; NNAPI is lowest priority (FULLY_CONNECTED stub works) |
| **`objc2` v0.6 + `objc2-foundation` for iOS FFI** | Maintains compatibility with existing `objc2` ecosystem; provides ergonomic `extern_category` and `rc` types; avoid `msg_send!` raw messaging where possible |
| **`candle-coreml` as optional workspace dep** | Only resolved on macOS; `Cargo.toml` uses `[target.'cfg(target_os = "macos")'.dependencies]` to prevent Linux resolution failures |
| **NNAPI compiler as separate `NnapiGraphBuilder`** | Separation of concerns — `Executor` handles device management, `GraphBuilder` maps Candle ops to NNAPI `addOperation`; allows swapping backends without touching graph logic |
| **All new modules behind `#[cfg]` + `#[cfg_attr(not(...), allow(dead_code))]`** | Keeps Linux CI green; platform-specific code is opt-in at compile time |

---

## Risks & Trade-offs

| Risk | Mitigation |
|------|------------|
| `objc2` v0.6 API drift from latest v0.7 | Pin to v0.6; upgrade when macOS build env is available and tested |
| `candle-core` 0.10.2 vs `candle-coreml` ^0.9 semver gap | Fork `candle-coreml` to update its dep, or contribute upstream patch (multi-hour task) |
| NNAPI supports ~100+ operation codes; full coverage unrealistic | Target 10-15 high-frequency ops: ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE, TRANSPOSE, BATCH_TO_SPACE_ND |
| No on-device test harness for NNAPI/iOS | Document `cargo ndk` / `xcodebuild` test commands; no CI integration until device runner is available |

---

## Open Questions

1. Should `iOSMonitor` use `objc2`'s `declare_class!` or manual `msg_send!` for `UIDevice` calls? `declare_class!` is more ergonomic but may require `objc2` v0.7+; `msg_send!` is more portable.
2. For `candle-coreml`, should Metal buffer interop use `CVMetalTextureCache` sharing (zero-copy but complex) or direct `MTLDevice.newBufferWithBytes` (simpler but copies)?
3. NNAPI compiler: discover available operations via `ANeuralNetworksDevice_getExtensionSupport` or hardcode against feature level 3 (API 28)?

---

## Specifications

### iOS Telemetry (`ios-telemetry`)

#### Requirement: iOS thermal state reading
The system SHALL read the current iOS thermal state via `ProcessInfo.thermalState` using `objc2` FFI and map it to the crate's `ThermalState` enum.

- **WHEN** `iOSMonitor::sample()` is called
- **THEN** query `NSProcessInfo.processInfo.thermalState` and map:
  - `NSProcessInfoThermalStateCritical` → `ThermalState::Critical`
  - `NSProcessInfoThermalStateSerious` → `ThermalState::Throttled`
  - Other values → `ThermalState::Nominal`

#### Requirement: iOS available memory reading
The system SHALL read available memory via `os_proc_available_memory()` and total memory via `NSProcessInfo.physicalMemory`.

- **WHEN** `iOSMonitor::sample()` is called
- **THEN** return available and total memory in megabytes

#### Requirement: iOS battery state reading
The system SHALL read battery level and charging state via `UIDevice.batteryLevel` and `UIDevice.batteryState`.

- **WHEN** `iOSMonitor::sample()` is called
- **THEN** enable battery monitoring, read `batteryLevel` (0.0–1.0 → 0–100) and `batteryState` (charging if `Charging` or `Full`)
- **WHEN** battery monitoring was already disabled beforehand
- **THEN** restore the previous state after sampling

#### Requirement: iOSMonitor implements HardwareMonitor
`iOSMonitor` SHALL implement `HardwareMonitor`, sampling all three metrics in a single 1 Hz thread, returning `HealthSnapshot` with timestamp.

- **WHEN** `iOSMonitor::new()` is called
- **THEN** spawn a 1 Hz sampling thread storing results in `Arc<Mutex<HealthSnapshot>>`
- **WHEN** `health()` is called but no sample collected yet
- **THEN** return `ThermalState::Nominal`, zeroed `MemoryStatus`, `PowerState { on_battery: true, level: 0 }`

#### Requirement: iOS telemetry tests
At least 6 unit tests covering creation, thermal/memory/battery conversion, fallback behavior, and snapshot freshness.

- **WHEN** compiled on non-macOS targets
- **THEN** tests excluded via `#[cfg]`

### CoreML/ANE Inference (`coreml-ane-inference`)

#### Requirement: candle-coreml dependency resolution
The workspace SHALL resolve `candle-coreml` as an optional macOS-only dependency.

- **WHEN** building on Linux CI
- **THEN** `candle-coreml` SHALL NOT be resolved or compiled
- **WHEN** building on macOS with `--features coreml`
- **THEN** `candle-coreml` SHALL be available as a dependency of `atheer-accel`

#### Requirement: CoreML ANE forward pass
`CoreMLBackend` SHALL offload tensor operations to ANE via `candle_coreml::Device::ANE`, falling back to Metal GPU then CPU.

- **WHEN** a model is ANE-compatible and `candle-coreml` is available
- **THEN** `forward()` delegates to `candle_coreml::Device::ANE` and returns real logits
- **WHEN** model fails ANE compatibility detection
- **THEN** fall back to `candle_core::Device::Metal` with `tracing::warn!`
- **WHEN** neither ANE nor Metal is available
- **THEN** fall back to CPU with latency measurement

#### Requirement: CoreML compatibility detection
Detect model ANE compatibility: supported architectures (LLaMA, Mistral, Falcon), quantization (q4_k_m, f16), size limits (~8 GB NPU ceiling).

- **WHEN** `CoreMLBackend::is_available()` is called
- **THEN** return `true` only when `candle-coreml` is available AND `candle_coreml::Device::ane_if_available()` returns `Some`

### NNAPI Full Compiler (`nnapi-full-compiler`)

#### Requirement: NNAPI graph builder
`NnapiGraphBuilder` SHALL map Candle operations to NNAPI operation codes and construct a model graph via `ANeuralNetworksModel_addOperation`.

- **WHEN** `NnapiGraphBuilder::new()` is called
- **THEN** create and hold a new `ANeuralNetworksModel`
- **WHEN** a Candle op is ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE, TRANSPOSE, or BATCH_TO_SPACE_ND
- **THEN** map to the corresponding `ANEURALNETWORKS_*` code and call `addOperation`
- **WHEN** a Candle op has no NNAPI equivalent
- **THEN** return `Err(AccelError::UnsupportedOperation(...))`

#### Requirement: Model compilation from graph
`.finish()` SHALL call `ANeuralNetworksModel_finish`. `.compile()` SHALL return an `NnapiExecutor`-compatible handle.

- **WHEN** `.finish()` succeeds and `.compile()` is called
- **THEN** return a compilation handle ready for `ANeuralNetworksExecution_compute`

#### Requirement: Integration tests
Tests SHALL cover ADD, MUL, and SOFTMAX operations with operand validation.

- **WHEN** run on Android via `cargo ndk -t arm64-v8a --platform 26 test`
- **THEN** verify correct model graph construction for each operation

---

## Task List

### Phase 1: iOS Telemetry — objc2 FFI Setup

- [ ] 1.1 Add `objc2` and `objc2-foundation` to `atheer-hardware/Cargo.toml` under `[target.'cfg(any(target_os = "ios", target_os = "macos"))'.dependencies]`
- [ ] 1.2 Create `atheer-hardware/src/ios.rs` module with `#[cfg]` gate, public `iOSMonitor` struct, and module skeleton
- [ ] 1.3 Implement thermal state reading via `objc2` `NSProcessInfo.processInfo.thermalState` → `ThermalState` mapping
- [ ] 1.4 Implement memory reading via `os_proc_available_memory()` C FFI + `NSProcessInfo.physicalMemory` → `MemoryStatus`
- [ ] 1.5 Implement battery reading via `objc2` `UIDevice.current.batteryLevel`/`batteryState` → `PowerState`
- [ ] 1.6 Implement `iOSMonitor` struct with 1 Hz sampling thread, `HardwareMonitor` trait impl
- [ ] 1.7 Write 6+ unit tests for iOS telemetry (`#[cfg(target_os = "ios")]`-gated)

### Phase 2: CoreML/ANE Real Inference

- [ ] 2.1 Fork/update `candle-coreml` to compile against `candle-core` 0.10.2, or contribute upstream patch
- [ ] 2.2 Add `candle-coreml` as optional macOS-only dependency in workspace `Cargo.toml`
- [ ] 2.3 Implement `CoreMLBackend::forward()` with ANE tensor offloading via `candle_coreml::Device::ANE`
- [ ] 2.4 Implement Metal GPU fallback in `CoreMLBackend` when ANE unavailable or model incompatible
- [ ] 2.5 Add CoreML integration tests for ANE inference and Metal fallback paths

### Phase 3: NNAPI Full Model Compiler

- [ ] 3.1 Create `NnapiGraphBuilder` struct in `nnapi.rs` with `ANeuralNetworksModel` management
- [ ] 3.2 Implement operation mapping for 8+ NNAPI operation codes (ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION, RESHAPE)
- [ ] 3.3 Implement operand ordering and type validation for each mapped operation
- [ ] 3.4 Implement `.finish()` → `ANeuralNetworksModel_finish` and `.compile()` → compilation handle
- [ ] 3.5 Write integration tests for NNAPI graph builder (ADD, MUL, SOFTMAX with operand validation)

### Phase 4: Documentation

- [ ] 4.1 Document macOS build requirements in README.md (Xcode, `objc2`, CoreML.framework)
- [ ] 4.2 Document NNAPI compiler current operation coverage in `nnapi.rs` module docs
- [ ] 4.3 Add "Remaining Work" section to README linking to this PRD

---

*Generated from `openspec/changes/macos-blocked-tasks/` — proposal, design, specs, and tasks compiled into a single reference document.*
