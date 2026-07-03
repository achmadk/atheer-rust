# Atheer-Rust: Improvements for a Reliable, Secure, Privacy-First Edge AI Engine

> **Analysis based on full codebase review** · July 2026
>
> Reviewed: all 6 core crates (~19K LoC), CI, FFI, benchmarks, security/safety modules, whitepaper, and progress docs.

---

## Executive Summary

Atheer-Rust has strong architectural foundations — NPU-first acceleration, predictive thermal management, hierarchical KV cache, grammar-constrained decoding, and a clean Rust codebase. However, to be **market-ready as a reliable, secured, privacy-first edge AI engine**, there are significant gaps across 7 dimensions. The most critical are in **security hardening** (model integrity is rudimentary), **privacy engineering** (no encryption at rest, no differential privacy), and **reliability** (14 failing tests, no real-device validation).

---

## 1. 🔒 Security & Model Integrity

### Current State

The [security.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/security.rs) module is minimal — path allowlisting, size checks, and prompt truncation. There is **no model signature verification** despite the field existing (`enable_signature_verify: bool` is always `false` and has no implementation).

> [!CAUTION]
> Model integrity is the #1 attack surface for on-device AI. A malicious GGUF file can execute arbitrary computation through crafted weight values. Without cryptographic verification, the engine is vulnerable to supply-chain attacks.

### Gaps & Recommendations

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **No model signature verification** | 🔴 Critical | Implement Ed25519 or ECDSA signature verification for GGUF files. The `SecurityAudit.enable_signature_verify` field exists but is dead code. Wire it to a real verification pipeline using `ring` or `ed25519-dalek`. |
| **SHA-256 verification is download-only** | 🟠 High | The [model_registry.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/model_registry.rs#L299-L308) verifies hashes after download, but `Model::from_gguf()` does **not** verify hashes at load time. A file modified post-download passes silently. Add mandatory hash verification at `from_gguf()`. |
| **No GGUF format validation** | 🟠 High | The engine trusts GGUF metadata (tensor shapes, quantization markers) without validation. Malformed GGUF files could cause OOB reads via `mmap`. Add GGUF header/metadata validation before mmap. |
| **No memory-safe tensor bounds checking** | 🟡 Medium | The `mmap` model loading ([mmap_model.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/mmap_model.rs)) trusts file offsets. Add bounds checks to prevent mmap OOB access from malformed files. |
| **HTTP downloads over plain reqwest** | 🟡 Medium | [model_registry.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/model_registry.rs#L117-L124) downloads from HuggingFace without certificate pinning. On mobile networks, MITM is a real risk. Add TLS certificate pinning for model download endpoints. |
| **No sandboxing of model execution** | 🟡 Medium | The NNAPI and Vulkan backends execute compute on shared device resources. Consider seccomp/SELinux policy recommendations for Android deployments. |
| **Prompt truncation is not unicode-safe** | 🟡 Medium | `sanitize_prompt()` in [security.rs:56-61](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/security.rs#L55-L61) slices at byte offset `prompt[..max_len]` which will panic on multi-byte UTF-8. Use `prompt.chars().take(max_len)` or `floor_char_boundary()`. |

### Recommended New Modules

```
atheer-core/src/
├── model_verifier.rs    # Ed25519 signature + SHA-256 at load time
├── gguf_validator.rs    # GGUF header/metadata structural validation  
└── secure_download.rs   # Certificate-pinned HTTPS with retry + integrity
```

---

## 2. 🔐 Privacy Engineering

### Current State

The project positions itself as "privacy-first" by virtue of on-device inference. However, **privacy is more than locality** — it requires active engineering to prevent data leakage.

> [!IMPORTANT]
> "On-device" ≠ "privacy-first." True privacy requires: no data leaves the device, cached data is encrypted, PII is detected and handled, and there are no telemetry side-channels.

### Gaps & Recommendations

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **KV cache stored in plaintext** | 🔴 Critical | L2/L3 cache ([l2_warm.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-memory-bank/src/l2_warm.rs), [l3_compressed.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-memory-bank/src/l3_compressed.rs)) stores conversation context unencrypted. An attacker with device access can extract full conversation history from the cache. **Encrypt L2/L3 cache at rest** using AES-256-GCM with a device-bound key (Keystore on Android, Secure Enclave on iOS). |
| **Checkpoints stored in plaintext** | 🔴 Critical | KV cache checkpoints ([inference.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/inference.rs#L70-L77)) write raw tensor data to disk. Encrypt checkpoint files. |
| **No cache expiry / auto-wipe** | 🟠 High | L2/L3 cache has LRU eviction by size but no **time-based expiry**. Sensitive conversations persist indefinitely. Add configurable TTL (e.g., 24h default) and secure wipe (overwrite before delete). |
| **PII redaction is rudimentary** | 🟠 High | [PiiRedactor](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/safety.rs#L209-L271) uses basic string matching for emails and counts digits for credit cards. The phone number detection loop has a `continue` bug (line 250-253) that skips incrementing `i`, causing an infinite loop on digit sequences. Replace with regex-based detection or an established PII library. |
| **No differential privacy for cached context** | 🟡 Medium | The L2 warm cache stores exact context representations. Consider adding noise injection before L2 storage to prevent exact reconstruction of cached prompts. |
| **Model download reveals user intent** | 🟡 Medium | Downloading specific models from HuggingFace reveals what AI capabilities the user is seeking. Consider supporting pre-bundled models or onion routing for downloads. |
| **No secure memory wiping** | 🟡 Medium | After inference, token sequences and decoded text remain in process memory until reallocation. Use `zeroize` crate for sensitive buffers (prompt tokens, generated output, KV cache tensors when dropped). |
| **No audit logging** | 🟡 Medium | There's no tamper-evident log of what prompts were processed, what models were used, or what content was moderated. Add an append-only local audit log for compliance. |

### Recommended New Modules

```
atheer-core/src/
├── crypto.rs            # AES-256-GCM encryption for cache/checkpoints
├── secure_memory.rs     # Zeroize wrappers for sensitive buffers
└── audit_log.rs         # Append-only local audit trail

atheer-memory-bank/src/
├── encrypted_store.rs   # Encrypted L2/L3 persistence
└── ttl_policy.rs        # Time-based cache expiry with secure wipe
```

---

## 3. 🛡️ Reliability & Production Hardening

### Current State

The project claims "92% complete" with ~394 tests, but has **14 known test failures**, **40+ compiler warnings**, a broken CI lint job, and no real-device testing.

> [!WARNING]
> 14 failing tests, 40+ warnings, and a CI that can't pass clippy means the project cannot reliably detect regressions. This is the #1 reliability blocker.

### Gaps & Recommendations

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **14 test failures** (4 Metal panics + 10 platform-gated) | 🔴 Critical | The Metal tests panic due to `swap_remove` on empty Vec in upstream `candle-core`. Add `catch_unwind` wrapper in [metal.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-accel/src/metal.rs) and `#[cfg_attr(not(target_os = "macos"), ignore)]` on Metal tests. |
| **CI clippy always fails** | 🔴 Critical | `-D warnings` with 40+ existing warnings = permanently red CI. Fix all warnings or configure allowed lints in `Cargo.toml`. |
| **CI accuracy tests have env var bug** | 🟠 High | [ci.yml:150-156](file:///home/achmadkurnianto/PROJECTS/atheer-rust/.github/workflows/ci.yml#L150-L156) uses `$ATHEER_TEST_MODEL` before it's set (line 161). Move the `env:` block above the steps that use it. |
| **No macOS CI runner** | 🟠 High | CoreML, Metal, and iOS telemetry are only verifiable on macOS. Add a `macos-latest` runner for platform-specific crates. |
| **No Android cross-compilation CI** | 🟠 High | NNAPI and Vulkan backends have never been compiled in CI. Add `cargo ndk` build step. |
| **Fuzz harness is skeletal** | 🟡 Medium | Only [3 trivial fuzz targets](file:///home/achmadkurnianto/PROJECTS/atheer-rust/fuzz) with no corpus, no dictionary, no CI integration. Add structured fuzzing for GGUF parsing, tokenizer input, and grammar validation. |
| **No crash recovery for interrupted inference** | 🟡 Medium | If the app is killed mid-generation, the KV cache is lost. The checkpoint mechanism exists but isn't integrated with app lifecycle. Wire checkpoints to iOS `UIApplication.willTerminateNotification` / Android `onTrimMemory`. |
| **No watchdog for runaway inference** | 🟡 Medium | The timeout in `generate()` relies on cooperative checking. A stuck `forward()` call (e.g., Vulkan shader hang) will never return. Add a secondary watchdog thread that force-terminates after 2× the timeout. |
| **`atheer-bindgen` is dead code** | 🟡 Low | [atheer-bindgen](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-bindgen) expects a `.udl` file that doesn't exist. Either remove or rewrite. |
| **`generate-bindings.sh` is a no-op** | 🟡 Low | All binding generation code is commented out. Fix or remove. |

---

## 4. ⚡ Performance & Efficiency

### Current State

Architecture is strong (speculative decoding, NGram cache, per-op device routing), but **no actual benchmark numbers exist** — all entries in BENCHMARKS.md are "TBD."

### Gaps & Recommendations

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **No baseline performance numbers** | 🟠 High | Run the existing `perf-bench` on real hardware and populate [BENCHMARKS.md](file:///home/achmadkurnianto/PROJECTS/atheer-rust/BENCHMARKS.md). Without numbers, you can't prove claims or detect regressions. |
| **No competitive benchmarks** | 🟠 High | The whitepaper claims superiority over llama.cpp, MLC LLM, etc. but has zero comparison data. Run identical models on identical hardware to produce comparison numbers. |
| **Vulkan shaders are unoptimized** | 🟡 Medium | The GEMV and attention shaders use basic int8 quantized multiply. Consider: (a) warp-level reductions, (b) shared memory tiling, (c) async compute overlap for prefill+decode. |
| **No WASM/WebGPU backend** | 🟡 Medium | For browser-based edge deployments, a WebGPU backend would open a new market segment. The `AccelBackend` trait makes this extensible. |
| **No model quantization pipeline** | 🟡 Medium | Users must pre-quantize models externally. An integrated quantization step (FP16→INT4/INT8) at download time would improve UX and storage efficiency. |
| **Context window eviction is naive** | 🟡 Medium | [maybe_evict()](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/inference.rs#L226-L273) drops oldest turns and clears the entire KV cache. This forces full re-prefill of all remaining turns. Implement incremental eviction that only removes the evicted turn's KV entries. |
| **No batched prefill** | 🟡 Low | The inference engine processes one sequence at a time. For agent loops with tool outputs, batched prefill could amortize overhead. |

---

## 5. 🌍 Platform & Ecosystem Maturity

### Current State

UniFFI bindings exist but the generation pipeline is broken, and the pre-generated bindings may be stale.

### Gaps & Recommendations

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **Binding generation is broken** | 🟠 High | `generate-bindings.sh` is a no-op; `atheer-bindgen` expects a non-existent UDL file. Fix the binding generation pipeline so it's automated and CI-verified. |
| **No Linux/desktop backend** | 🟡 Medium | The engine targets iOS/Android but has no desktop GPU path (CUDA, ROCm). For development and testing, a Linux GPU path would accelerate iteration. |
| **No model format abstraction** | 🟡 Medium | Currently hardcoded to GGUF. Supporting GGML, SafeTensors, or ONNX would broaden model compatibility. |
| **No plugin/extension system** | 🟡 Medium | Custom backends, samplers, or grammar constraints require forking. A trait-based plugin registry would allow third-party extensions. |
| **No OTA model update mechanism** | 🟡 Medium | Models are downloaded once. There's no mechanism for incremental model updates (delta patches) or silent background updates. |
| **iOS SDK is stale** | 🟡 Low | [ios/atheer_ffi.swift](file:///home/achmadkurnianto/PROJECTS/atheer-rust/ios) was pre-generated from an earlier build. May not match current API surface. |
| **No example apps** | 🟡 Low | The `android/` directory has a `MainActivity.kt` but no complete runnable demo app. Provide reference apps for both platforms. |

---

## 6. 📋 Compliance & Certification Readiness

### Gaps for Market Readiness

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **No SOC 2 / ISO 27001 readiness** | 🟠 High | Enterprise customers need compliance documentation. Start with: access control for model files, audit logging, key management documentation. |
| **No GDPR data flow documentation** | 🟠 High | Even for on-device inference, GDPR requires documenting what personal data is processed, how it's stored, and how to delete it. The L2/L3 cache stores conversation context that may contain personal data — document the data flow and provide a `delete_all_user_data()` API. |
| **No export control consideration** | 🟡 Medium | Cryptographic components (if added per recommendation #2) may be subject to export controls. Document compliance with EAR/ITAR. |
| **No accessibility testing** | 🟡 Low | If the engine is used in consumer apps, ensure the FFI API supports accessibility metadata in responses (e.g., alt-text generation). |

---

## 7. 🧑‍💻 Developer Experience & Observability

### Gaps & Recommendations

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **No structured telemetry/metrics** | 🟠 High | The engine uses `tracing::info!` for logging but has no structured metrics export (Prometheus, OpenTelemetry). Add a `MetricsCollector` trait for tok/s, latency percentiles, cache hit rate, mode transitions. |
| **No CLAUDE.md / AGENTS.md** | 🟡 Medium | No AI-assistant onboarding documentation. Add agent guidance files for codebase navigation. |
| **No API versioning** | 🟡 Medium | The FFI API has no version number. Breaking changes to `GenerationRequest` or `AtheerConfig` will silently break consumer apps. Add API version negotiation. |
| **Error types don't carry context** | 🟡 Medium | `AtheerCoreError` variants like `GenerationFailed(String)` lose structured context. Use `thiserror` with source chains instead of string formatting. |
| **No configuration validation** | 🟡 Low | `AtheerConfig` accepts arbitrary values without validation. Add a `validate()` method that checks `max_seq_len > 0`, `temperature > 0`, etc. |

---

## Priority Roadmap

### Phase 1: Security & Reliability Foundation (2–3 weeks)
1. Fix all 14 test failures and 40+ warnings
2. Implement model signature verification (Ed25519)
3. Add hash verification at model load time (not just download)
4. Fix the PII redactor infinite loop bug
5. Fix the CI env var bug and add macOS runner
6. Encrypt L2/L3 cache at rest

### Phase 2: Privacy Hardening (2–3 weeks)
7. Add AES-256-GCM encryption for cache/checkpoints
8. Implement secure memory wiping with `zeroize`
9. Add time-based cache expiry with secure deletion
10. Implement audit logging
11. Add `delete_all_user_data()` API

### Phase 3: Performance Validation (2–3 weeks)
12. Run real benchmarks and populate BENCHMARKS.md
13. Run competitive benchmarks vs llama.cpp
14. Fix mode switching benchmark (wire orchestrator)
15. Implement grammar overhead benchmark

### Phase 4: Market Readiness (4–6 weeks)
16. Fix binding generation pipeline
17. Add GDPR compliance documentation
18. Create reference iOS and Android apps
19. Implement structured metrics/telemetry
20. Add API versioning to FFI layer

---

## Summary of Critical Bugs Found

| Bug | File | Line | Impact |
|-----|------|------|--------|
| PII phone detection infinite loop | [safety.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/safety.rs#L249-L253) | 249-253 | `continue` skips `i += 1`, causing infinite loop on digit input |
| Prompt truncation panics on UTF-8 | [security.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-core/src/security.rs#L57-L59) | 57-59 | `prompt[..max_len]` slices at byte boundary, panics on multi-byte chars |
| CI env var used before set | [ci.yml](file:///home/achmadkurnianto/PROJECTS/atheer-rust/.github/workflows/ci.yml#L150-L161) | 150-161 | `$ATHEER_TEST_MODEL` used at line 150/156 but defined at line 161 |
| Metal tests panic on empty device list | [metal.rs](file:///home/achmadkurnianto/PROJECTS/atheer-rust/atheer-accel/src/metal.rs) | upstream | `swap_remove` on empty `Vec` in `candle-core` |

---

> **Bottom line:** Atheer has a genuinely differentiated architecture — no competitor combines NPU-first probing, predictive thermal management, L1/L2/L3 KV cache, grammar decoding, and agent loops in Rust. But "privacy-first" and "reliable" require active engineering beyond on-device execution. The gaps above are the delta between a promising prototype and a market-ready product.
