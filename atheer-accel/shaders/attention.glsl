#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Q, K, V matrices (batch x seq_len x head_dim)
layout(set = 0, binding = 0) readonly buffer QBuffer {
    float Q[];
};

layout(set = 0, binding = 1) readonly buffer KBuffer {
    float K[];
};

layout(set = 0, binding = 2) readonly buffer VBuffer {
    float V[];
};

// Output: attention result
layout(set = 0, binding = 3) buffer OutputBuffer {
    float output[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint seq_len;
    uint head_dim;
    uint num_heads;
    float scale;
};

shared float shared_Q[256];
shared float shared_K[256];

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationID.x;

    uint total_elements = batch_size * num_heads * seq_len * head_dim;
    if (gid >= total_elements) {
        return;
    }

    uint batch_idx = gid / (num_heads * seq_len * head_dim);
    uint remainder = gid % (num_heads * seq_len * head_dim);
    uint head_idx = remainder / (seq_len * head_dim);
    remainder = remainder % (seq_len * head_dim);
    uint seq_idx = remainder / head_dim;
    uint dim_idx = remainder % head_dim;

    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len || dim_idx >= head_dim) {
        return;
    }

    // Flash-attention style: compute Q * K^T for this position
    // For a single element in the output, sum over head_dim
    float sum = 0.0f;
    uint q_base = batch_idx * num_heads * seq_len * head_dim
                + head_idx * seq_len * head_dim
                + seq_idx * head_dim;

    // Accumulate over key dimension (simplified flash attention)
    for (uint k = 0; k < head_dim && k < 64u; k++) {
        uint q_idx = q_base + k;
        uint k_idx = batch_idx * num_heads * seq_len * head_dim
                   + head_idx * seq_len * head_dim
                   + k * head_dim
                   + dim_idx;
        if (q_idx < total_elements && k_idx < total_elements) {
            sum += Q[q_idx] * K[k_idx];
        }
    }

    // Apply softmax scaling
    sum = exp(sum * scale);

    // Weighted sum over V (simplified)
    float result = 0.0f;
    for (uint v = 0; v < head_dim && v < 64u; v++) {
        uint v_idx = batch_idx * num_heads * seq_len * head_dim
                   + head_idx * seq_len * head_dim
                   + v * head_dim
                   + dim_idx;
        if (v_idx < total_elements) {
            result += sum * V[v_idx];
        }
    }

    uint out_idx = gid;
    if (out_idx < total_elements) {
        output[out_idx] = result;
    }
}
