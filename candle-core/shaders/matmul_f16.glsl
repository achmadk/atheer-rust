#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer LhsBuffer {
    uint lhs[];
};

layout(set = 0, binding = 1) readonly buffer RhsBuffer {
    uint rhs[];
};

layout(set = 0, binding = 2) writeonly buffer DstBuffer {
    uint dst[];
};

layout(push_constant) uniform PushConstants {
    uint b;
    uint m;
    uint n;
    uint k;
    uint lhs_offset;
    uint rhs_offset;
    uint dst_offset;
    uint lhs_stride_m;
    uint lhs_stride_k;
    uint rhs_stride_k;
    uint rhs_stride_n;
    uint dst_stride_m;
    uint dst_stride_n;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = b * m * n;
    if (gid >= total) {
        return;
    }

    uint b_idx = gid / (m * n);
    uint mn_idx = gid % (m * n);
    uint m_idx = mn_idx / n;
    uint n_idx = mn_idx % n;

    float result = 0.0;

    for (uint k_idx = 0; k_idx < k; k_idx++) {
        uint lhs_elem_idx = b_idx * lhs_stride_m * m + m_idx * lhs_stride_k + k_idx + lhs_offset;
        uint lhs_word_idx = lhs_elem_idx / 2u;

        uint rhs_elem_idx = b_idx * rhs_stride_k * k + k_idx * rhs_stride_n + n_idx + rhs_offset;
        uint rhs_word_idx = rhs_elem_idx / 2u;

        float lhs_val;
        float rhs_val;

        if (lhs_elem_idx % 2u == 0) {
            lhs_val = unpackHalf2x16(lhs[lhs_word_idx]).x;
        } else {
            lhs_val = unpackHalf2x16(lhs[lhs_word_idx]).y;
        }

        if (rhs_elem_idx % 2u == 0) {
            rhs_val = unpackHalf2x16(rhs[rhs_word_idx]).x;
        } else {
            rhs_val = unpackHalf2x16(rhs[rhs_word_idx]).y;
        }

        result += lhs_val * rhs_val;
    }

    uint dst_elem_idx = b_idx * dst_stride_m * m + m_idx * dst_stride_n + n_idx + dst_offset;
    uint dst_word_idx = dst_elem_idx / 2u;
    uint dst_half_idx = dst_elem_idx % 2u;

    if (dst_half_idx == 0) {
        uvec2 existing = uvec2(dst[dst_word_idx], 0);
        dst[dst_word_idx] = packHalf2x16(vec2(result, unpackHalf2x16(existing.x).y));
    } else {
        uvec2 existing = uvec2(dst[dst_word_idx], 0);
        dst[dst_word_idx] = packHalf2x16(vec2(unpackHalf2x16(existing.x).x, result));
    }
}
