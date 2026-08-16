//! Inline WGSL compute shaders for the wgpu Vulkan backend.
//!
//! WGSL is the **canonical** shader source for this backend. The GLSL files
//! under `candle-core/shaders/*.glsl` are unbuilt reference material only (kept
//! for the Phase 2 quantized-forward-pass port); they are not compiled by
//! `build.rs` nor loaded at runtime. Any new GPU kernel must be added here as
//! WGSL and dispatched from `device.rs`. See change `vulkan-wgsl-gpu-matmul`.

pub const ADD_SHADER: &str = r#"
struct Buffer {
    data: array<f32>,
};

@group(0) @binding(0)
var<storage, read> lhs : Buffer;

@group(0) @binding(1)
var<storage, read> rhs : Buffer;

@group(0) @binding(2)
var<storage, read_write> output : Buffer;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&lhs.data) && idx < arrayLength(&rhs.data)) {
        output.data[idx] = lhs.data[idx] + rhs.data[idx];
    }
}
"#;

pub const AFFINE_SHADER: &str = r#"
struct Buffer {
    data: array<f32>,
};

struct Uniforms {
    mul: f32,
    add: f32,
};

@group(0) @binding(0)
var<storage, read> input : Buffer;

@group(0) @binding(1)
var<storage, read_write> output : Buffer;

@group(0) @binding(2)
var<storage, read> uniforms : Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input.data)) {
        output.data[idx] = input.data[idx] * uniforms.mul + uniforms.add;
    }
}
"#;

pub const BROADCAST_ADD_SHADER: &str = r#"
struct Buffer {
    data: array<f32>,
};

@group(0) @binding(0)
var<storage, read> lhs : Buffer;

@group(0) @binding(1)
var<storage, read> rhs : Buffer;

@group(0) @binding(2)
var<storage, read_write> output : Buffer;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let lhs_size = arrayLength(&lhs.data);
    let rhs_size = arrayLength(&rhs.data);
    let out_size = arrayLength(&output.data);

    if (idx < out_size) {
        // For broadcasting, we compute the effective index in each tensor
        // This is a simplified version - full implementation would handle
        // multi-dimensional broadcasting
        let lhs_val = if idx < lhs_size { lhs.data[idx] } else { 0.0 };
        let rhs_val = if idx < rhs_size { rhs.data[idx] } else { 0.0 };
        output.data[idx] = lhs_val + rhs_val;
    }
}
"#;

pub const GEMM_SHADER: &str = r#"
struct Buffer {
    data: array<f32>,
};

// GEMM: C = A * B, shapes [b, m, k] x [b, k, n] -> [b, m, n].
//
// Correctness-first (naive, one invocation per output element). Operand
// indexing is fully strided/offset so that transposed or offset layouts (e.g.
// x.matmul(&w.t())) compute correctly. Stride semantics mirror candle's CPU
// MatMul: per-batch skip strides plus row/col strides for each operand, with a
// contiguous destination. All accumulation is in f32 regardless of I/O dtype.
struct Metadata {
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    dst_offset: u32,
    lhs_batch_stride: u32,
    rhs_batch_stride: u32,
    lhs_stride_m: u32,   // stride between rows of A (elements)
    lhs_stride_k: u32,   // stride between cols of A (elements)
    rhs_stride_k: u32,   // stride between rows of B (elements)
    rhs_stride_n: u32,   // stride between cols of B (elements)
    dst_stride_m: u32,   // contiguous dst row stride (= n)
    dst_stride_n: u32,   // contiguous dst col stride (= 1)
}

@group(0) @binding(0)
var<storage, read> a : Buffer;

@group(0) @binding(1)
var<storage, read> b : Buffer;

@group(0) @binding(2)
var<storage, read_write> c : Buffer;

@group(0) @binding(3)
var<uniform> meta : Metadata;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;

    if (row >= meta.m || col >= meta.n || batch >= meta.batch) {
        return;
    }

    let a_base = meta.lhs_offset + batch * meta.lhs_batch_stride + row * meta.lhs_stride_m;
    let b_base = meta.rhs_offset + batch * meta.rhs_batch_stride + col * meta.rhs_stride_n;

    var sum = 0.0;
    for (var kk = 0u; kk < meta.k; kk = kk + 1u) {
        let a_idx = a_base + kk * meta.lhs_stride_k;
        let b_idx = b_base + kk * meta.rhs_stride_k;
        sum = sum + a.data[a_idx] * b.data[b_idx];
    }

    let c_idx = meta.dst_offset + batch * (meta.m * meta.n)
        + row * meta.dst_stride_m + col * meta.dst_stride_n;
    c.data[c_idx] = sum;
}
"#;
