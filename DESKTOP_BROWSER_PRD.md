# Product Requirement Document (PRD): Atheer-Rust Universal Web & Desktop Core Expansion

## 1. Document Control & Metadata

* **Document Title:** Product Requirement Document (PRD) - Atheer-Rust Web & Desktop Target Expansion
* **Version:** 1.0.0
* **Author:** Atheer Architecture Core Team
* **Status:** Proposed / Review
* **Target Release:** Q3 2026
* **Crates Impacted:** `atheer-accel`, `atheer-orchestrator`, `atheer-core`, `atheer-ffi`

---

## 2. Executive Summary & Objective

Atheer-Rust has successfully validated local, safe-Rust on-device LLM inference on mobile platforms (iOS and Android) via Neural Engine and NNAPI hardware acceleration. To establish Atheer as the premiere cross-platform, deployment-agnostic local inference engine, it must expand to target **Desktop environments (Native & Hybrid)** and **Web browsers**.

The goal of this initiative is to implement a hybrid **WebGPU + WebAssembly (Wasm) SIMD** pipeline within the existing workspace. This expansion will allow the engine to run securely inside traditional browser sandboxes, hybrid runtimes (Tauri, Electron), and native desktop environments with high throughput, strict memory isolation, and bulletproof fallback mechanics.

---

## 3. Scope of Core Features & Architecture Changes

### 3.1 Web & Desktop Hardware Probing Hierarchy

The `atheer-accel` crate will be expanded to support a browser and desktop-native compilation target. The routing engine must dynamically evaluate host runtime capabilities at initialization using the following prioritization:

1. **Priority 1: WebGPU High-Performance Pipeline** (Native WGSL with Subgroups)
* *Fallback:* WebGPU Compatibility Pipeline (`naga`-transpiled GLSL)


2. **Priority 2: Multi-Threaded Wasm SIMD** (`SharedArrayBuffer` Threadpool)
3. **Priority 3: Scalar Wasm CPU Pipeline** (Universal Fallback)

### 3.2 Dual-Path Shader Pipeline

To achieve universal compatibility without sacrificing bleeding-edge desktop performance, the GPU compute pipeline must implement a dual-path architecture:

* **Automated Pipeline (`naga` Compilation):** Reuse the core GLSL kernels (`gemv.glsl`, `attention.glsl`) used in the Android Vulkan backend. Use the `naga` compiler at build time to transpile these into universally compatible WGSL code-blocks.
* **Hand-Optimized WGSL Pipeline:** Write native WGSL shaders featuring hardware-specific enhancements, specifically targeting the WebGPU `subgroups` language extension. This path will bypass Shared Local Memory (SLM) limits using intra-warp shuffles (`subgroupBroadcast`, `subgroupAdd`) to accelerate matrix-vector multiplications during attention prefill/decode loops.

### 3.3 Hybrid WebAssembly Memory Model Configuration

To balance maximum throughput against the memory limits of large models, the compilation targets will support two deterministic memory configurations:

* **Memory32 Configuration (Target: Small Quantized Models 1B–3B):** Compiled with standard 32-bit linear memory. Constrains memory to 4GB. This forces the use of the engine's 16-token `PagedAttention` blocks to prevent dynamic heap fragmentation, securing zero-cost hardware bounds checking from browser JIT engines.
* **Memory64 Configuration (Target: High-Capacity Desktop Contexts 7B+):** Compiled via the WebAssembly `Memory64` proposal (`wasm64-unknown-unknown`). This breaks the 4GB ceiling to allocate larger un-chunked model tensors into the linear heap at the cost of software-injected bounds checks.

---

## 4. Functional Requirements

### FR-001: Async Device Probing & Capabilities Negotiation

* **Description:** The orchestrator must asynchronously probe for WebGPU compatibility via `navigator.gpu` or native desktop `wgpu` adapters.
* **Requirements:**
* Must check for the explicit presence of the `SUBGROUPS` feature extension in the WebGPU adapter.
* If the device initialization fails, throws an error, or lacks WebGPU support entirely, the engine must catch the fault without crashing and immediately switch to initializing the Wasm SIMD worker pool.



### FR-002: Dynamic Cross-Origin Isolation Fallback

* **Description:** Multi-threaded Wasm execution depends on `SharedArrayBuffer`, which requires specific security headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`).
* **Requirements:**
* The engine must test if `globalThis.crossOriginIsolated` is true.
* If false, the engine must bypass the multi-threaded Wasm SIMD threadpool and fall back gracefully to a thread-safe single-threaded scalar execution loop while alerting the user application via standard telemetry.



### FR-003: Bit-Shifted Low-Precision Quantization Kernels

* **Description:** Standard WebGPU implementations lack uniform support for sub-byte native data types (like native `i4` or `i8`).
* **Requirements:**
* The WGSL compute shaders (both `naga`-transpiled and native optimized) must utilize 32-bit integer unpacking mechanics.
* INT4/INT8 model weights must be stored packed within `u32` arrays, and the shader must execute bit-shifting and masking operations at runtime to unpack weights on-the-fly inside the execution registers.



### FR-004: Asynchronous WebGPU Device Recovery

* **Description:** Browser environments routinely dump GPU contexts during system sleep, tab switches, or OS-level hardware priority changes.
* **Requirements:**
* The `atheer-accel` layer must bind a listener to the WebGPU `device.lost` promise event.
* Upon context termination, the engine must immediately serialize the active execution state (including prompt context position and current generation metrics), halt GPU polling, and transfer processing to the Wasm SIMD backend seamlessly mid-token.



---

## 5. Non-Functional Requirements (NFR)

* **NFR-001 (Memory Safety):** Under no circumstances shall an external JS boundary interaction or an accelerator driver panic result in memory access leaks or an unhandled crash of the host window thread. All entries must navigate safe FFI wrappers protected by Rust `catch_unwind`.
* **NFR-002 (Context Isomorphic Execution):** The generated logits of a specific model must remain mathematically identical (within an allowable floating-point epsilon of $10^{-5}$) regardless of whether the execution occurred on a native iOS ANE backend, a desktop native subgroup WGSL shader, or a single-threaded Wasm scalar loop.
* **NFR-003 (Size Footprint Constraint):** The compiled Wasm core engine footprint (excluding model weights) must not exceed 5MB after standard Dead Code Elimination (DCE), LLVM optimization passes, and Brotli compression.

---

## 6. Testing, Verification, & Differential Logit Checkers

To guarantee engine reliability across diverging compilation targets, a specialized differential test suite must be introduced inside `perf-bench`:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_differential_shader_outputs() {
        let input_tensor = generate_mock_logits(1, 512);
        
        let output_naga = run_naga_transpiled_wgsl(&input_tensor);
        let output_optimized = run_subgroup_optimized_wgsl(&input_tensor);
        
        assert_logits_match!(output_naga, output_optimized, epsilon = 1e-5);
    }
}

```

### Verification Criteria

1. **Continuous Integration (CI):** Headless automated browser environments (via Playwright or Puppeteer with WebGPU flags enabled) must run the entire `atheer-core` test block per commit.
2. **Crash Containment Test:** A simulated `DeviceLost` panic must be triggered manually during a heavy batch token prefill loop. The test passes only if the system completes the sequence using Wasm fallback without losing the conversational context or dropping active memory pages.

---

## 7. Risks & Mitigation Strategies

| Risk Factor | Impact | Likelihood | Mitigation Strategy |
| --- | --- | --- | --- |
| Browser environments reject Memory64 due to lack of stable Safari implementation. | High | Medium | Keep Memory32 as the default compilation configuration; enforce PagedAttention block swapping to stream weights in for larger contexts instead of storing them natively in the active Wasm heap. |
| Injected software bounds checks on Memory64 degrade matrix multiplication speeds. | Medium | High | Rely heavier on the WebGPU path for execution whenever Memory64 configurations are deployed on desktop runtimes. |
| Divergent behavior across GPU drivers causing precision drift in handwritten WGSL. | Critical | Medium | Embed the automated differential verification script directly into the startup initialization routine of the browser application package. |