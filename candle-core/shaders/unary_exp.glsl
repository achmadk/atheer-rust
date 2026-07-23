#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer SrcBuffer {
    uint src[];
};

layout(set = 0, binding = 1) writeonly buffer DstBuffer {
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

    float value = (half_idx == 0)
        ? unpackHalf2x16(src[word_idx]).x
        : unpackHalf2x16(src[word_idx]).y;

    value = exp(value);

    if (half_idx == 0) {
        uvec2 existing = uvec2(dst[word_idx], 0);
        dst[word_idx] = packHalf2x16(vec2(value, unpackHalf2x16(existing.x).y));
    } else {
        uvec2 existing = uvec2(dst[word_idx], 0);
        dst[word_idx] = packHalf2x16(vec2(unpackHalf2x16(existing.x).x, value));
    }
}
