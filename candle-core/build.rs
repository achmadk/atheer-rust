use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&out_path).unwrap();

    // NOTE: The GLSL -> SPIR-V compilation pipeline (naga) was removed here.
    // The wgpu Vulkan backend uses inline WGSL as its single canonical shader
    // source (see `candle-core/src/vulkan_backend_wgpu/shaders.rs`). The GLSL
    // files under `candle-core/shaders/*.glsl` are retained only as unbuilt
    // reference material for the Phase 2 quantized-forward-pass port (notably
    // `dequant_q4k.glsl` and the stride model in `matmul_f16.glsl`). Nothing at
    // runtime loads `$OUT_DIR/shaders/*.spv`. See change `vulkan-wgsl-gpu-matmul`.

    println!("cargo:rerun-if-changed=build.rs");
}
