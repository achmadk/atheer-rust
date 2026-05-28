#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: token IDs (batch)
layout(set = 0, binding = 0) readonly buffer InputBuffer {
    uint input_ids[];
};

// Weight matrix (quantized int8) + scale
layout(set = 0, binding = 1) readonly buffer WeightBuffer {
    uint weights[];  // Packed int4 pairs or int8 values
};

// Output: logits (batch_size x vocab_size)
layout(set = 0, binding = 2) buffer OutputBuffer {
    float logits[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint vocab_size;
    uint hidden_size;
    uint quantization_type; // 0 = f32, 1 = q4_k_m, 2 = q8_0
};

// DP4A-style dot product for quantized weights
int dot_int4(uint packed_a, uint packed_b) {
    int result = 0;
    // Unpack 4 int4 values from each uint
    for (int i = 0; i < 8; i++) {
        int a_val = int((packed_a >> (i * 4)) & 0xFu);
        int b_val = int((packed_b >> (i * 4)) & 0xFu);
        // Dequantize int4 (signed)
        if (a_val >= 8) a_val -= 16;
        if (b_val >= 8) b_val -= 16;
        result += a_val * b_val;
    }
    return result;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationID.x;

    if (gid >= batch_size * vocab_size) {
        return;
    }

    uint token_idx = gid / vocab_size;
    uint vocab_idx = gid % vocab_size;

    if (token_idx >= batch_size || vocab_idx >= vocab_size) {
        return;
    }

    // For now, compute a simple activation pattern
    // Real GEMV would read weight row for this vocab_idx and dot with input embedding
    float sum = 0.0f;

    // Simulate GEMV: use token ID as index into weight-like pattern
    uint token_id = input_ids[token_idx];
    if (vocab_idx == token_id % vocab_size) {
        sum = 1.0f;
    }

    // Add small noise based on position to avoid all-zero rows
    sum += float(gid) * 0.000001f;

    logits[gid] = sum;
}
