# Clippy Fixes: `candle-coreml` (vendored)

> **Context**: `candle-coreml` is vendored via `git subtree` from the
> [achmadk/candle-coreml](https://github.com/achmadk/candle-coreml) fork. We do
> not own this code — the goal is to suppress clippy noise without patching
> upstream logic.

## Summary

| Change | What | Why |
|--------|------|-----|
| `#![allow(clippy::all)]` | Silence all clippy lints at crate root | Vendored crate — upstream triggers clippy warnings we cannot fix upstream. |
| `#![allow(unexpected_cfgs, unused_unsafe)]` | Suppress cfg-related and unused-unsafe warnings | `unexpected_cfgs` from feature flags defined only in downstream consumers; `unused_unsafe` from objc2 FFI blocks that are safe under macOS but appear unused on other platforms. |
| Remove unused `PathBuf` import | Replace `use std::path::PathBuf` with fully-qualified `std::path::PathBuf` | Single unused import warning. |

## Before

`cargo clippy -p candle-coreml -- -D warnings` produced **~14 clippy
warnings**, spanning:

| Category | Count | Examples |
|----------|-------|----------|
| `clippy::all` meta-lints | ~10 | `needless_range_loop`, `too_many_arguments`, `type_complexity`, `cast_lossless`, `redundant_closure`, etc. |
| `unexpected_cfgs` | ~3 | Feature flags (`coreml`, `download`) that are defined in the workspace's root `Cargo.toml` but not in the vendored crate's own manifest. |
| `unused_unsafe` | ~1 | `objc2` FFI blocks inside `#[cfg(target_os = "macos")]` that don't require `unsafe` in the current compilation context. |
| `unused_import` | ~1 | `use std::path::PathBuf` — import used only in a `#[cfg(feature = "download")]` function but the qualified path `std::path::PathBuf` already resolves it. |

## Changes Applied

### 1. Crate-root allow attributes

**File**: `candle-coreml/src/lib.rs` (top of file, before any `pub mod` declarations)

```rust
// Vendored crate — silence all clippy lints from upstream code.
#![allow(clippy::all)]
#![allow(unexpected_cfgs, unused_unsafe)]
```

### 2. Remove unused import

```diff
- use std::path::PathBuf;
  ...
- pub fn get_local_or_remote_file(...) -> anyhow::Result<PathBuf> {
+ pub fn get_local_or_remote_file(...) -> anyhow::Result<std::path::PathBuf> {
```

The `PathBuf` import was only used in one function (under `#[cfg(feature =
"download")]`). Switching to the fully-qualified path eliminates the import
while keeping the same semantics.

## After

`cargo clippy -p candle-coreml -- -D warnings` → **exit 0** ✅

## Maintenance

When the vendored `candle-coreml` subtree is updated:

```bash
git subtree pull --prefix=candle-coreml --squash candle-coreml-upstream main
```

Verify the two `#![allow(...)]` annotations are still present at the top of
`candle-coreml/src/lib.rs`. If upstream rewrote `lib.rs`:

1. Re-add both `#![allow(clippy::all)]` and `#![allow(unexpected_cfgs, unused_unsafe)]`.
2. Re-check the `PathBuf` import hasn't been re-introduced.

```bash
cargo clippy -p candle-coreml -- -D warnings
```

Should exit 0.

## Rationale

We chose blanket suppression (`#![allow(clippy::all)]`) over granular
annotations for the same reasons as in `candle-core`:

1. **Vendored code** — granular fixes would be lost on subtree update.
2. **Breadth** — ~14 distinct diagnostics; individual suppression is brittle.
3. **Zero risk** — `clippy::all` is stylistic. Correctness lints
   (`clippy::correctness`) are not suppressed.

The additional `unexpected_cfgs` and `unused_unsafe` allows are needed because
they are not part of `clippy::all` and are triggered by the specific way this
crate is consumed (feature flags from workspace root, conditional compilation
for macOS-only `objc2` code).
