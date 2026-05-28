fn main() {
    // Use uniffi's CLI to generate bindings
    if let Err(e) = uniffi::cli::run() {
        eprintln!("Binding generation failed: {}", e);
        std::process::exit(1);
    }
}
