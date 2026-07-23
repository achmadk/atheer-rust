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

// Gemm shader for matrix multiplication
// A: [M, K], B: [K, N], C: [M, N]
// Sequential matmul implementation for correctness, optimized tiling can come later
@group(0) @binding(0)
var<storage, read> a : Buffer;

@group(0) @binding(1)
var<storage, read> b : Buffer;

@group(0) @binding(2)
var<storage, read_write> c : Buffer;

@group(0) @binding(3)
var<uniform> meta : Metadata;

struct Metadata {
    M: u32,
    N: u32,
    K: u32,
    batch: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;

    let M = meta.M;
    let N = meta.N;
    let K = meta.K;

    let global_row = batch * M + row;
    let global_col = col;

    if (row < M && col < N && batch < meta.batch) {
        var sum = 0.0;
        for (var k = 0u; k < K; k = k + 1u) {
            let a_idx = global_row * K + k;
            let b_idx = k * N + global_col;
            sum = sum + a.data[a_idx] * b.data[b_idx];
        }
        let c_idx = global_row * N + global_col;
        c.data[c_idx] = sum;
    }
}
"#;
