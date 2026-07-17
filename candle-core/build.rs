use std::env;
use std::fs;
use std::path::Path;

#[cfg(all(feature = "vulkan", target_os = "android"))]
fn compile_shader(shader_path: &Path, out_path: &Path) {
    let glsl_source = fs::read_to_string(shader_path)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", shader_path));

    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Compute,
        defines: Default::default(),
    };
    let module = frontend
        .parse(&options, &glsl_source)
        .unwrap_or_else(|_| panic!("Failed to parse GLSL: {:?}", shader_path));

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|_| panic!("Failed to validate SPIR-V: {:?}", shader_path));

    let spv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 0),
            ..Default::default()
        },
        None,
    )
    .unwrap_or_else(|_| panic!("Failed to write SPIR-V: {:?}", shader_path));

    let output_bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();

    let stem = shader_path.file_stem().unwrap().to_str().unwrap();
    let spv_path = out_path.join(format!("{}.spv", stem));
    fs::write(&spv_path, &output_bytes)
        .unwrap_or_else(|_| panic!("Failed to write SPIR-V: {:?}", spv_path));
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&out_path).unwrap();

    #[cfg(all(feature = "vulkan", target_os = "android"))]
    {
        let shader_dir = Path::new("shaders");
        let shaders = [
            "dequant_q4k.glsl",
            "matmul_f16.glsl",
            "copy.glsl",
            "affine.glsl",
            "binary_add.glsl",
            "binary_mul.glsl",
            "binary_div.glsl",
            "unary_silu.glsl",
            "unary_exp.glsl",
            "cast.glsl",
            "reduce.glsl",
            "where_cond.glsl",
        ];
        for shader_name in &shaders {
            let shader_path = shader_dir.join(shader_name);
            if shader_path.exists() {
                compile_shader(&shader_path, out_path.as_path());
                println!("cargo:rerun-if-changed=shaders/{}", shader_name);
            }
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
