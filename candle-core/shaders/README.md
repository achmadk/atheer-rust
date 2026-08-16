# GLSL shaders — reference only (NOT built)

These `*.glsl` files are **not part of the build** and are **not loaded at
runtime**. As of the `vulkan-wgsl-gpu-matmul` change, the wgpu Vulkan backend
standardized on inline **WGSL** as its single canonical shader source; see
`candle-core/src/vulkan_backend_wgpu/shaders.rs`.

The `build.rs` GLSL→SPIR-V (naga) compilation step was removed. Nothing consumes
`$OUT_DIR/shaders/*.spv`.

These files are retained only as a porting reference for the **Phase 2**
quantized-forward-pass work (on-GPU dequantization + fused quantized matmul):

- `dequant_q4k.glsl` — q4_k dequantization logic to be ported to WGSL for
  `to_dtype` / `QVulkanStorage::fwd`.
- `matmul_f16.glsl` — stride/offset push-constant model that informed the WGSL
  `GEMM_SHADER` metadata layout.

Do not wire these into the build. Add new kernels as WGSL in the backend module.
