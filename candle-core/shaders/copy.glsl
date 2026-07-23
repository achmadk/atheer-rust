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
    uint elem_count;
    uint src_stride;
    uint dst_stride;
    uint dim0_size;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= elem_count) {
        return;
    }

    uint src_idx = gid * src_stride + src_offset;
    uint dst_idx = gid * dst_stride + dst_offset;

    dst[dst_idx] = src[src_idx];
}
