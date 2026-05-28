# Accuracy Regression Test Suite

The accuracy regression suite detects unintended changes in model output behavior caused by code changes. It uses deterministic logit fingerprinting to compare outputs across commits.

## Overview

The core idea: for a fixed prompt and fixed random seed, the same model binary should produce identical logits. If logits change (e.g., due to a tensor layout bug, incorrect quantization, or operator substitution), accuracy regression tests catch it.

## Logit Fingerprinting

### `LogitFingerprint`

Captures the output behavior of a model for a fixed prompt:

```rust
pub struct LogitFingerprint {
    pub prompt: String,                // The input prompt
    pub prefill_fingerprint: String,   // SHA-256 of top-K prefill logits
    pub max_tokens: u32,               // Number of generation steps
    pub generation_fingerprint: String,// SHA-256 of all generation fingerprints (accumulated)
}
```

### `fingerprint_topk(logits, k) -> Result<String>`

Extracts a deterministic fingerprint from a logit tensor:

1. Flattens the tensor to a 1-D `Vec<f32>`
2. Sorts values descending
3. Truncates to top-K
4. SHA-256 hashes the little-endian byte representation of each value
5. Returns hex-encoded hash

Only top-K logits are fingerprinted to make the digest insensitive to numerical noise in very small logits that don't affect sampling decisions.

### `capture_fingerprint(model, prompt, max_tokens, k) -> Result<LogitFingerprint>`

Runs a model with a fixed input and captures fingerprints:

1. **Prefill**: Runs a BOS token (id=1) through the model, fingerprints the output logits
2. **Generation**: Autoregressively generates `max_tokens` tokens by greedy argmax decoding. At each step, fingerprints the logit output and accumulates into a running SHA-256 hash
3. Returns both the prefill and accumulated generation fingerprint

### `cosine_similarity(a, b) -> Option<f64>`

Standard cosine similarity between two `&[f32]` vectors. Returns `None` on length mismatch or zero vectors.

### `compare_accuracy(baseline, candidate) -> AccuracyComparison`

Compares two `LogitFingerprint` values:

```rust
pub struct AccuracyComparison {
    pub prefill_match: bool,                  // Exact SHA-256 match
    pub generation_match: bool,               // Exact SHA-256 match
    pub prefill_cosine_similarity: Option<f64>,// Reserved for future use
    pub token_match_count: usize,             // Reserved for future use
    pub token_total_count: usize,             // Reserved for future use
}
```

## Test Suite

### Unit Tests (no model file required)

| Test | Description |
|------|-------------|
| `test_fingerprint_empty_logit` | Single zero-value logit → 64-char hex digest |
| `test_fingerprint_deterministic` | Same input → same fingerprint (twice) |
| `test_fingerprint_k_smaller_than_vocab` | Works when K < vocabulary size |
| `test_cosine_similarity_identical` | Identical vectors → similarity ~1.0 |
| `test_cosine_similarity_orthogonal` | Orthogonal vectors → similarity ~0.0 |
| `test_cosine_similarity_mismatched_length` | Different lengths → `None` |
| `test_compare_accuracy_mismatch` | Prefill matches, generation differs → partial match |

### Integration Tests (require a real GGUF model)

All integration tests are marked `#[ignore]` because they require a real GGUF model file:

| Test | Description |
|------|-------------|
| `test_accuracy_with_real_model` | Load GGUF, run prefill + 5 tokens, verify 64-char fingerprints |
| `test_kv_cache_roundtrip_accuracy` | Snapshot → restore → forward, verify output logits are identical |
| `test_quantization_accuracy` | Load GGUF, run 10-token generation, capture fingerprint |

## Running Tests

```bash
# Unit tests only (no model required)
cargo test -p atheer-core -- accuracy

# Download the test model (one-time setup, ~350 MB)
scripts/download-test-model.sh

# Integration tests (auto-downloads if missing)
ATHEER_TEST_MODEL=./models/LFM2-700M-Q4_0.gguf cargo test -p atheer-core -- --ignored accuracy
```

## Baselining

The integration tests verify that fingerprinting works (output is 64 hex chars) but do not currently assert against stored baselines. To establish baselines for a known-good model:

1. Run the full accuracy suite against a trusted build:
   ```bash
   ATHEER_TEST_MODEL=/path/to/model.gguf cargo test -p atheer-core -- --ignored --test-threads=1
   ```
2. Capture the output to record fingerprints:
   ```bash
   ATHEER_TEST_MODEL=/path/to/model.gguf cargo test -p atheer-core -- --ignored accuracy 2>&1 | tee baseline-output.txt
   ```
3. Store baseline fingerprints alongside the model reference (or embed them as constants in `accuracy.rs`)

When adding baseline assertion tests, use the pattern:

```rust
let fp = capture_fingerprint(&mut model, "Hello world", 5, 100).unwrap();
let expected = "abcdef123456...";  // stored baseline
assert_eq!(fp.prefill_fingerprint, expected, "prefill regression detected");
```

## Error Budgets

| Quantization | Expected Cosine Similarity | Fingerprint Stability |
|-------------|--------------------------|----------------------|
| FP16 (no quantization) | 1.0 (exact) | Identical across runs on same binary |
| INT8 | ~0.99+ | May produce different fingerprint from FP16 |

The fingerprint uses SHA-256 of top-K logit values, so even a single differing logit near the top-K boundary will produce a completely different hash. For tolerance-aware comparisons, prefer cosine similarity over exact hash matching when comparing across quantization levels.

## CI Integration

Integration tests now run automatically on every push/PR to `main`. The CI workflow downloads the LFM2-700M-Q4_0 GGUF model (via `scripts/download-test-model.sh`) and caches it using `actions/cache`. Both the accuracy regression tests and multi-turn conversation tests execute with the downloaded model:

```yaml
- name: Download test model
  run: scripts/download-test-model.sh
- name: Run accuracy tests
  run: ATHEER_TEST_MODEL=/tmp/models/LFM2-700M-Q4_0.gguf cargo test -p atheer-core -- --ignored
```
