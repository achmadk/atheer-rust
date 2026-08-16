# Kotlin owns the Binder transport; Rust owns the sandbox policy

The Android SDK layer owns service binding, DeathRecipient observation, and AIDL marshalling. The Rust bridge is a pure policy state machine (lifecycle, crash counting, escalation, persistence) fed by events — connected, probe result, death — arriving over the FFI callback channel, with the fallback decision returned via `set_on_sandbox_fallback`. Rust also owns async pre-warm orchestration. This reuses the tested `SandboxedGpuBridge` and the existing FFI callback pattern, and keeps `atheer-core` free of Android framework code.

Status: accepted

## Considered Options

- **Rust implements bindService via JNI inside atheer-core** — rejected: drags Android framework code into the core crate, and async ServiceConnection callbacks fight the synchronous state machine.
