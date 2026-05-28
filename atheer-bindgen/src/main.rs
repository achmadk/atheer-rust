use camino::Utf8PathBuf;
use std::path::PathBuf;
use uniffi_bindgen::ComponentInterface;
use uniffi_udl::parse_udl;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: aether-bindgen <udl-path> <language> [output-dir]");
        eprintln!("Languages: swift, kotlin");
        std::process::exit(1);
    }

    let udl_path =
        Utf8PathBuf::from_path_buf(PathBuf::from(&args[1])).expect("Invalid UTF-8 in path");
    let language = &args[2];
    let output_dir = if args.len() > 3 {
        Utf8PathBuf::from_path_buf(PathBuf::from(&args[3])).expect("Invalid UTF-8 in output path")
    } else {
        Utf8PathBuf::from_path_buf(std::env::current_dir().unwrap()).unwrap()
    };

    println!("Parsing UDL from {}", udl_path);

    // Read UDL content
    let udl_content = std::fs::read_to_string(&udl_path).expect("Failed to read UDL file");

    // Parse UDL to ComponentInterface
    let metadata = parse_udl(&udl_content, "aether_ffi").expect("Failed to parse UDL");

    let ci =
        ComponentInterface::from_metadata(metadata).expect("Failed to create component interface");

    println!("Created component interface: {}", ci.namespace());

    match language.as_str() {
        "swift" => {
            let swift_dir = output_dir.join("swift");
            std::fs::create_dir_all(&swift_dir).expect("Failed to create swift dir");

            let config = uniffi_bindgen::bindings::swift::gen_swift::Config::default();
            uniffi_bindgen::bindings::swift::write_bindings(&config, &ci, &swift_dir, true)
                .expect("Failed to generate Swift bindings");

            println!("Swift bindings written to {}", swift_dir);
        }
        "kotlin" => {
            let kotlin_dir = output_dir.join("kotlin");
            std::fs::create_dir_all(&kotlin_dir).expect("Failed to create kotlin dir");

            let config = uniffi_bindgen::bindings::kotlin::gen_kotlin::Config::default();
            uniffi_bindgen::bindings::kotlin::write_bindings(&config, &ci, &kotlin_dir, true)
                .expect("Failed to generate Kotlin bindings");

            println!("Kotlin bindings written to {}", kotlin_dir);
        }
        _ => {
            eprintln!("Unknown language: {}. Use 'swift' or 'kotlin'", language);
            std::process::exit(1);
        }
    }

    println!("Done!");
}
