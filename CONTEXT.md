# Atheer Engine

An on-device LLM inference engine for iOS and Android that runs models through NPU/GPU accelerators, with sandboxed execution in an Android isolated process and layered guardrails in the main process.

## Language

### GPU execution sandbox

**Bridge**:
The component in the main process that manages the shard's lifecycle, tracks its crashes, and decides when generation falls back to CPU.
_Avoid_: sandbox manager, supervisor

**Shard**:
The component that runs inside an Android isolated process and owns the model's KV cache and forward pass.
_Avoid_: worker, GPU process, sandbox process

**KV page**:
A queued (token, position) pair awaiting transfer to the shard in a batch.
_Avoid_: batch unit, decode step

**Probe**:
The startup check that forwards a known tensor through the shard's GPU pipeline and verifies the output before the shard accepts inference work.
_Avoid_: attestation, startup test

**Fallback**:
The state in which generation runs on CPU in the main process, entered permanently once the shard exceeds the crash threshold.
_Avoid_: degraded mode, CPU mode

**Sandboxed execution**:
The mode in which inference runs inside an Android isolated process.
_Avoid_: sandboxing (the verb is fine)

**Isolated process**:
An Android process with its own UID, no network access, and no filesystem access except file descriptors passed to it.
_Avoid_: sandbox process

**Turbo mode**:
The orchestrator mode that uses speculative decoding; excluded from sandboxed execution in v1.
