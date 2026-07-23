#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer Q4KBuffer {
    uint data[];
};

layout(set = 0, binding = 1) writeonly buffer F16Buffer {
    uint dst[];
};

layout(push_constant) uniform PushConstants {
    uint block_idx;
    uint num_blocks;
};

const uint QK_K = 256u;
const uint K_SCALE_SIZE = 12u;
const uint BYTES_PER_BLOCK = 2u + 2u + K_SCALE_SIZE + QK_K / 2u;

float dequant_q4k(uint block_base, uint elem_idx) {
    uint byte_idx = elem_idx / 2u;
    uint q = (data[block_base + 2u + K_SCALE_SIZE / 4u + byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
    uint nibble = (q >> ((elem_idx % 2u) * 4u)) & 0xFu;

    uint scale_idx = elem_idx / 16u;
    uint scale_byte_idx = 2u + (scale_idx / 4u);
    uint scale_nibble_idx = scale_idx % 4u;

    uint scales_word = data[block_base + 2u + scale_byte_idx / 4u];
    uint scales_byte = (scales_word >> ((scale_byte_idx % 4u) * 8u)) & 0xFFu;

    uint scale = (scales_byte >> (scale_nibble_idx * 2u)) & 0x3Fu;
    if (scale >= 32u) {
        scale = scale | 0xFFFFFFC0u;
    }

    float d = uintBitsToFloat(data[block_base]);
    float dmin = uintBitsToFloat(data[block_base + 1u]);

    float result = (float(nibble) * float(scale) / 16.0f) * d + dmin;
    return result;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint block = gid / QK_K;
    uint elem = gid % QK_K;

    if (block >= num_blocks) {
        return;
    }

    uint block_base = block * (BYTES_PER_BLOCK / 4u);

    float value = dequant_q4k(block_base, elem);

    dst[gid] = floatBitsToUint(value);
}
