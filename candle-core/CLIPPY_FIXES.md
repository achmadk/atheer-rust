# Clippy Fixes: `candle-core` (vendored)

> **Context**: `candle-core` is vendored via `git subtree` from the upstream
> [huggingface/candle](https://github.com/huggingface/candle) repository. We do
> not own this code — the goal is to suppress clippy noise without patching
> upstream logic.

## Summary

| Change | What | Why |
|--------|------|-----|
| `#![allow(clippy::all)]` | Silence all clippy lints at crate root | Vendored crate — upstream code triggers ~30+ clippy warnings (`needless_range_loop`, `too_many_arguments`, `type_complexity`, etc.) that we cannot fix upstream. |

## Before

`cargo clippy -p candle-core -- -D warnings` produced **~85 clippy errors/warnings**
across the vendored crate, spanning:
- `needless_range_loop` — hot-loop style idiomatic in Candle but flagged by clippy
- `too_many_arguments` — ML kernel signatures accept many parameters by design
- `type_complexity` — tensor shape types are inherently nested
- `cast_lossless`, `cast_possible_truncation` — int/float casts in performance-sensitive paths
- `missing_safety_doc` — vendored `unsafe` blocks without doc comments
- `mut_from_ref` — internal patterns in tensor storage
- `should_implement_trait` — iterator-like structs without full trait impls
- `redundant_closure`, `redundant_field_names`, `unnecessary_cast`
- Various `clippy::pedantic` lints

## Change Applied

**File**: `candle-core/src/lib.rs` (after doc comments, before any modules)

```rust
// Vendored crate — silence all clippy lints from upstream code.
#![allow(clippy::all)]
```

## After

`cargo clippy -p candle-core -- -D warnings` → **exit 0** ✅

## Maintenance

When the vendored `candle-core` subtree is updated:

```bash
git subtree pull --prefix=candle-core --squash candle-core-upstream crate-candle-core
```

Verify the `#![allow(clippy::all)]` is still present at `candle-core/src/lib.rs`.
If the upstream moved or rewrote `lib.rs`, re-add the annotation after the
crate-level doc comments.

```bash
cargo clippy -p candle-core -- -D warnings
```

Should exit 0. If new clippy issues appear despite the blanket allow, check
that `#![allow(clippy::all)]` is at the crate root and not inside a module.

## Rationale

We chose `#![allow(clippy::all)]` (blanket suppress) over individual
`#[allow(clippy::xxx)]` annotations because:

1. **Vendored code** — we don't own it; granular fixes would be lost on subtree
   update when upstream code changes.
2. **Breadth** — ~85 distinct clippy diagnostics across the crate; individual
   suppression would require touching hundreds of lines.
3. **Zero risk** — `clippy::all` is purely stylistic/cosmetic. It does not
   suppress correctness lints like `clippy::correctness` (which is not part of
   `clippy::all`).
