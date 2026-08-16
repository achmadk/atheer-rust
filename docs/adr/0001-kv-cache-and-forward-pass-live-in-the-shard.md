# KV cache and forward pass live in the shard

The shard owns the KV cache and forward pass; the main process keeps sampling, grammar-constrained decoding, guardrails, and the L2/L3 memory-bank hierarchy. The AIDL contract (token IDs + positions in, logits out) is only coherent if the shard can decode against its own KV cache — shipping KV tensors over Binder would defeat the "batch KV pages to amortize IPC" design. Consequences: the main process's L1 (active context) is a shadow in sandboxed mode, a shard crash loses worker-held context (fallback re-decodes from the prompt), and checkpoint/restore must eventually cross the process boundary.

Status: accepted

## Considered Options

- **Parent-owned KV (stateless compute worker)** — rejected: per-call KV shipping over Binder is too slow and contradicts the batched-IPC premise.
- **Shard-owned sampling** — rejected: breaks grammar-constrained output and the L3 output guard, which live in the main process.
