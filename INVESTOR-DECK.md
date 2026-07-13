# Atheer-Rust — Investor & Partner Deck

> **Slide-by-slide narrative**
> A mobile inference engine for on-device LLMs — built for the NPU era.
> July 2026

---

## Slide 1: Title

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                    ▲THER ▲ R                             │
│                                                         │
│         A mobile inference engine for on-device LLMs     │
│                                                         │
│                 Built for the NPU era                    │
│                                                         │
│                      July 2026                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Elevator Pitch:**
> Atheer is the only LLM inference engine that runs on both iOS and Android NPUs,
> predicts thermal throttling before it happens, manages memory across a 3-tier
> cache hierarchy, and ships a built-in agent loop — all in one Rust codebase.

---

## Slide 2: The Problem

**LLMs are moving to devices. The infrastructure isn't ready.**

```
                    ┌──────────────────────────────────────┐
                    │      The Mobile Inference Gap         │
                    ├──────────────────────────────────────┤
                    │                                      │
                    │  Server        →  100-400W, CUDA     │
                    │  Desktop       →  65-150W, GPU       │
                    │  Laptop        →  15-45W, GPU/CPU    │
                    │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
                    │  Phone         →  3-6W,  Throttle at │
                    │                      45°C,  Shared   │
                    │                      4-8GB RAM       │
                    └──────────────────────────────────────┘
```

**Key constraints that existing engines fail to address:**
- **Thermal**: Phones throttle at ~45°C skin temp. Existing engines run flat-out until the OS forcibly cuts performance by 60%+.
- **Accelerators**: Every phone has a different NPU (ANE, Hexagon, MediaTek APU, Samsung NPU) — but most engines ignore them.
- **Memory**: 4-8GB shared with the OS and apps. A single 7B model at 4K context uses ~6GB. No room for multi-turn agent sessions.
- **Agentic AI**: The industry is moving to agent loops (tool calling, multi-turn). Mobile engines were designed for single-turn Q&A.

**Market signal:** Apple Intelligence, Samsung Galaxy AI, Google Gemini Nano — all major platforms are betting on on-device AI. But they're building proprietary stacks. There's a gap for an open, cross-platform engine.

---

## Slide 3: The Solution

**Atheer-Rust: Mobile inference, reimagined.**

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                    ▲THER ▲ R                             │
│                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │  Auto-select  │  │  Predictive  │  │ L1/L2/L3 KV  │  │
│   │  NPU/GPU/CPU  │  │   Thermal    │  │    Cache     │  │
│   │  per platform │  │  Orchestr.   │  │  Hierarchy   │  │
│   └──────────────┘  └──────────────┘  └──────────────┘  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │   Grammar     │  │  Built-in    │  │  UniFFI FFI  │  │
│   │  Constrained  │  │Agent Loop    │  │  Swift+Kotlin│  │
│   │   Decoding    │  │              │  │              │  │
│   └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│         All in safe Rust. One codebase.                  │
│         MIT / Apache 2.0.                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 4: Technical Moat

**Why this is hard to replicate.**

| Moat | Why It's Defensible | Competitor Status |
|---|---|---|
| **NPU-first acceleration** | Raw NDK FFI (~20 extern fns) + CoreML ANE. Requires deep platform expertise. | llama.cpp doesn't have it. MLC LLM doesn't have it. ExecuTorch has Qualcomm-only. |
| **Predictive thermal orchestration** | Least-squares regression on live telemetry. Pre-emptive downgrade. | No competitor has any thermal awareness at all. |
| **L1/L2/L3 KV cache** | Three-tier with alignment gating and handoff protocols. *(Promotion scoring heuristics not yet implemented — the alignment_score is currently a manually-set placeholder.)* | Every competitor uses a single flat cache. |
| **Grammar-constrained decoding** | Pushdown automaton as a first-class Rust trait. Thread-safe. | Only llama.cpp has comparable GBNF, but it's a C bolt-on. |
| **Built-in agent loop** | Autonomous tool-calling with configurable max steps. | Every other engine requires you to build this yourself. |
| **UniFFI cross-platform** | One UDL → Swift + Kotlin. No platform-specific SDK maintenance. | Competitors require separate binding efforts per platform. |
| **Rust safety** | No GC pauses, no use-after-free, catch_unwind across FFI. | All competitors (except MLX Swift) use C/C++ with manual memory. |

---

## Slide 5: Market Landscape

**Positioning map: Atheer vs. existing engines**

```
                    Specialist                Generalist
                    ──────────                ──────────
              │
    iOS+Android│     Atheer                    ONNX Runtime
    cross-plat  │     (NPU+GPU+CPU,              (no LLM
              │      thermal, agent)             specific)
              │
              │
──────────────┼──────────────────────────────────────────
              │
              │
    iOS only   │     MLX Swift
              │     (no NPU, no
              │      Android)
              │
              │
              │
    Android    │     Qualcomm AI Hub
    only       │     (Qualcomm-only)
              │
              │
              └──────────────────────────────────────────
                    CPU-friendly        Accelerator-first
```

**Adjacent space nobody owns:** Cross-platform, NPU-first, multi-accelerator orchestration with thermal management.

---

## Slide 6: Business Model

**Open source core + commercial offerings.**

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Open Source (MIT / Apache 2.0)                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • Core inference engine (atheer-core)             │  │
│  │ • All acceleration backends (atheer-accel)        │  │
│  │ • Orchestrator, memory bank, hardware telemetry   │  │
│  │ • Perf-bench benchmarking                         │  │
│  │ • UniFFI bindings (Swift + Kotlin)                │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Commercial (Future)                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • Managed model zoo (pre-optimized models)        │  │
│  │ • Custom backend development (new NPU targets)    │  │
│  │ • Enterprise support SLA                          │  │
│  │ • Dashboard & analytics for deployed engines      │  │
│  │ • Priority feature development                    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Market: Any company shipping LLMs on mobile devices     │
│  TAM: On-device AI inference market → $XXB by 2028      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 7: Traction & Velocity

**Current state:**

| Metric | Value |
|---|---|
| Lines of Rust | ~18,000 |
| Tests | ~390 |
| Crates | 7 production + perf-bench |
| Backends | 5 (CoreML, Metal, NNAPI, Vulkan, CPU) |
| Supported model architectures | LLaMA, Mistral, Gemma, Phi, Qwen 2+ |
| Quantization formats | Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q8_0, F16 |
| License | MIT / Apache 2.0 |

**Key milestones achieved:**
- Custom Vulkan compute shaders (GEMV + attention) compiled to SPIR-V via naga
- Raw NNAPI NDK FFI with full graph compiler (9 ops)
- CoreML/ANE compatibility detection with fallback chain
- Predictive thermal model with least-squares trend estimation
- L1/L2/L3 KV cache with HandoffProtocol
- Grammar-constrained decoding (38-state pushdown automaton)
- Built-in agent loop with tool-calling support
- UniFFI-generated Swift and Kotlin bindings

---

## Slide 8: Team

> **[Your team description here]**
>
> *Built by engineers who understand both mobile platforms and ML inference —
> combining Rust systems programming, GPU compute, and LLM optimization.*

**Key expertise areas:**
- Rust systems programming
- Mobile platform engineering (iOS + Android)
- GPU compute (Metal, Vulkan, GLSL/SPIR-V)
- NPU/ML accelerator programming (NNAPI NDK, CoreML)
- LLM inference optimization (quantization, speculative decoding, KV cache)

---

## Slide 9: Ask

**What we're looking for:**

| Type | Ask |
|---|---|
| **Strategic partners** | Mobile OEMs, chip vendors (Qualcomm, MediaTek, Samsung) — early access to NPU hardware for backend optimization |
| **Enterprise customers** | Companies deploying AI assistants on mobile devices — pilot program for integration support |
| **Investment** | Seed round for full-time development, device lab, and commercialization |

---

## Slide 10: Vision

**The future of on-device AI is heterogeneous, adaptive, and open.**

```
    2026                    2027                    2028
    ─────                   ─────                   ─────
    ┌──────────┐           ┌──────────┐           ┌──────────┐
    │ Mobile   │           │ Mobile + │           │ Edge     │
    │ LLMs     │──────────▶│ Wearable │──────────▶│ Every-   │
    │ iOS/     │           │ IoT      │           │ where    │
    │ Android  │           │ Auto     │           │          │
    └──────────┘           └──────────┘           └──────────┘
         │                      │                      │
         ▼                      ▼                      ▼
    single-engine          multi-device            on-device
    per device             orchestrated            training +
                           federation              personalization
```

Atheer's architecture — designed for heterogeneous accelerators, thermal awareness, and autonomous agents — maps directly onto this future. The same code that runs on an iPhone today can run on a car's infotainment system, a smart glasses SoC, or a robot's embedded computer tomorrow.

---

*This deck is a living document. Metrics and milestones updated July 2026.*
