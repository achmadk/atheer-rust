use crate::Result;

/// Interface for reading and restoring GPU-side KV cache state.
///
/// Implementations provide access to the per-layer key/value cache that
/// lives on the accelerator device (Metal, Vulkan, CPU) during autoregressive
/// inference.  The flat `Vec<f32>` buffers can be transferred between cache
/// tiers (GPU↔L1↔L2↔L3) or persisted for model handoff / session restore.
pub trait KvCacheBridge {
    /// Return a copy of every layer's KV cache as flat CPU buffers.
    ///
    /// Each entry is `(keys_flat, values_flat)`.  Layers with no cached entries
    /// return `(vec![], vec![])`.
    fn kv_cache_snapshot(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>>;

    /// Overwrite every layer's KV cache from a previous snapshot.
    ///
    /// The snapshot must have exactly one entry per model layer.  Empty buffers
    /// clear the corresponding layer.
    fn kv_cache_restore(&mut self, snapshot: &[(Vec<f32>, Vec<f32>)]) -> Result<()>;
}

// ── blanket impl for any type that derefs to KvCacheBridge ──────
// (no blanket needed – callers use the concrete impl directly)
