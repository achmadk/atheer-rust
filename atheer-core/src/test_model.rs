use std::path::{Path, PathBuf};
use std::process::Command;

/// Ensure a test model is available, downloading it if necessary.
///
/// Behavior:
/// - If `ATHEER_TEST_MODEL` is set and the file exists → return its path.
/// - If `ATHEER_TEST_MODEL` is set but the file is missing → attempt to
///   download the model to that path by spawning `scripts/download-test-model.sh`.
/// - If `ATHEER_TEST_MODEL` is not set → panic with a helpful message.
pub fn ensure_test_model() -> PathBuf {
    let var = std::env::var("ATHEER_TEST_MODEL")
        .expect("ATHEER_TEST_MODEL env var must be set to run integration tests");

    let path = PathBuf::from(&var);
    if path.exists() {
        return path;
    }

    // File doesn't exist — try to auto-download
    eprintln!(
        "ATHEER_TEST_MODEL set to {var} but file not found. Attempting download..."
    );

    let script_path = find_download_script();
    let status = Command::new(&script_path)
        .status()
        .expect("Failed to spawn download-test-model.sh");

    if !status.success() {
        panic!(
            "Model download failed for {var}. Run `{}` manually and retry.",
            script_path.display()
        );
    }

    // Check if download placed the model at the expected path
    if path.exists() {
        eprintln!("Model downloaded to {var}");
        path
    } else {
        // Script downloaded to default location; check there
        let default_path = default_model_path();
        if default_path.exists() {
            eprintln!("Model downloaded to default location: {}", default_path.display());
            default_path
        } else {
            panic!(
                "Model download script ran but file not found at {var} or {}. \
                 Run `{}` manually.",
                default_path.display(),
                script_path.display()
            );
        }
    }
}

/// Locate `scripts/download-test-model.sh` relative to the project root.
fn find_download_script() -> PathBuf {
    // Check CARGO_MANIFEST_DIR first (works during `cargo test`)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidate = Path::new(&manifest_dir)
            .parent()
            .map(|p| p.join("scripts").join("download-test-model.sh"))
            .unwrap_or_else(|| PathBuf::from("scripts/download-test-model.sh"));
        if candidate.exists() {
            return candidate;
        }
    }

    // Fallback: check relative to current directory
    let candidates = [
        "scripts/download-test-model.sh",
        "../scripts/download-test-model.sh",
        "../../scripts/download-test-model.sh",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return PathBuf::from(c);
        }
    }

    // Default — let the Command fail with a clear error
    PathBuf::from("scripts/download-test-model.sh")
}

/// Default model path (matching the download script).
fn default_model_path() -> PathBuf {
    // Relative to workspace root (parent of atheer-core)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        Path::new(&manifest_dir)
            .parent()
            .map(|p| p.join("models").join("LFM2-700M-Q4_0.gguf"))
            .unwrap_or_else(|| PathBuf::from("models/LFM2-700M-Q4_0.gguf"))
    } else {
        PathBuf::from("models/LFM2-700M-Q4_0.gguf")
    }
}
