#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputBuffer {
    uint input_ids[];
};

layout(set = 0, binding = 1) buffer OutputBuffer {
    float logits[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint vocab_size;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;

    if (gid >= batch_size) {
        return;
    }

    uint token_id = input_ids[gid];
    uint offset = gid * vocab_size + token_id;

    // Bounds check
    if (offset < vocab_size * batch_size) {
        logits[offset] = 1.0;
    }
}
