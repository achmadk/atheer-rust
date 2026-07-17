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
    uint offset;
    uint count;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= count) {
        return;
    }

    uint elem_idx = gid + offset;
    uint word_idx = elem_idx / 2u;
    uint half_idx = elem_idx % 2u;

    float lhs_val = (half_idx == 0)
        ? unpackHalf2x16(lhs[word_idx]).x
        : unpackHalf2x16(lhs[word_idx]).y;
    float rhs_val = (half_idx == 0)
        ? unpackHalf2x16(rhs[word_idx]).x
        : unpackHalf2x16(rhs[word_idx]).y;

    float result = lhs_val + rhs_val;

    if (half_idx == 0) {
        uvec2 existing = uvec2(dst[word_idx], 0);
        dst[word_idx] = packHalf2x16(vec2(result, unpackHalf2x16(existing.x).y));
    } else {
        uvec2 existing = uvec2(dst[word_idx], 0);
        dst[word_idx] = packHalf2x16(vec2(unpackHalf2x16(existing.x).x, result));
    }
}
