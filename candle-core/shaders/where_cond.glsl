#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer CondBuffer {
    uint cond[];
};

layout(set = 0, binding = 1) readonly buffer TrueBuffer {
    uint t[];
};

layout(set = 0, binding = 2) readonly buffer FalseBuffer {
    uint f[];
};

layout(set = 0, binding = 3) writeonly buffer DstBuffer {
    uint dst[];
};

layout(push_constant) uniform PushConstants {
    uint cond_offset;
    uint t_offset;
    uint f_offset;
    uint dst_offset;
    uint count;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= count) {
        return;
    }

    uint cond_elem_idx = gid + cond_offset;
    uint cond_word_idx = cond_elem_idx / 32u;
    uint cond_bit_idx = cond_elem_idx % 32u;
    uint cond_word = cond[cond_word_idx];
    bool cond_val = (cond_word & (1u << cond_bit_idx)) != 0;

    uint t_elem_idx = gid + t_offset;
    uint t_word_idx = t_elem_idx / 2u;
    uint t_half_idx = t_elem_idx % 2u;
    float t_val = (t_half_idx == 0)
        ? unpackHalf2x16(t[t_word_idx]).x
        : unpackHalf2x16(t[t_word_idx]).y;

    uint f_elem_idx = gid + f_offset;
    uint f_word_idx = f_elem_idx / 2u;
    uint f_half_idx = f_elem_idx % 2u;
    float f_val = (f_half_idx == 0)
        ? unpackHalf2x16(f[f_word_idx]).x
        : unpackHalf2x16(f[f_word_idx]).y;

    float result = cond_val ? t_val : f_val;

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
