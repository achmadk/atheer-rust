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
    uint total_count;
    uint reduce_count;
    uint reduce_dim;
    uint op; // 0=sum, 1=max
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint dst_count = total_count / reduce_count;

    if (gid >= dst_count) {
        return;
    }

    uint block_base = gid * reduce_count;
    float result;

    if (op == 0) { // sum
        result = 0.0;
        for (uint i = 0; i < reduce_count; i++) {
            uint elem_idx = block_base + i + src_offset;
            uint word_idx = elem_idx / 2u;
            uint half_idx = elem_idx % 2u;
            float val = (half_idx == 0)
                ? unpackHalf2x16(src[word_idx]).x
                : unpackHalf2x16(src[word_idx]).y;
            result += val;
        }
    } else { // max
        result = -3.402823466e+38; // -INF
        for (uint i = 0; i < reduce_count; i++) {
            uint elem_idx = block_base + i + src_offset;
            uint word_idx = elem_idx / 2u;
            uint half_idx = elem_idx % 2u;
            float val = (half_idx == 0)
                ? unpackHalf2x16(src[word_idx]).x
                : unpackHalf2x16(src[word_idx]).y;
            result = max(result, val);
        }
    }

    uint dst_elem_idx = gid + dst_offset;
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
