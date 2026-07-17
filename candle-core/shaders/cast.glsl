#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer SrcBuffer {
    uint src[];
};

layout(set = 0, binding = 1) writeonly buffer DstBuffer {
    uint dst[];
};

layout(push_constant) uniform PushConstants {
    uint src_offset;
    uint dst_offset;
    uint count;
    uint src_dtype; // 0=F16, 1=F32, etc
    uint dst_dtype;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= count) {
        return;
    }

    uint src_elem_idx = gid + src_offset;
    uint dst_elem_idx = gid + dst_offset;
    uint src_word_idx = src_elem_idx / 2u;
    uint dst_word_idx = dst_elem_idx / 2u;
    uint src_half_idx = src_elem_idx % 2u;
    uint dst_half_idx = dst_elem_idx % 2u;

    float value;
    if (src_dtype == 0) { // F16
        value = (src_half_idx == 0)
            ? unpackHalf2x16(src[src_word_idx]).x
            : unpackHalf2x16(src[src_word_idx]).y;
    } else {
        value = 0.0; // TODO: handle other dtypes
    }

    if (dst_dtype == 0) { // F16
        if (dst_half_idx == 0) {
            uvec2 existing = uvec2(dst[dst_word_idx], 0);
            dst[dst_word_idx] = packHalf2x16(vec2(value, unpackHalf2x16(existing.x).y));
        } else {
            uvec2 existing = uvec2(dst[dst_word_idx], 0);
            dst[dst_word_idx] = packHalf2x16(vec2(unpackHalf2x16(existing.x).x, value));
        }
    }
}
