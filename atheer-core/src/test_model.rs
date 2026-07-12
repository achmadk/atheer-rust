use std::path::{Path, PathBuf};
use std::process::Command;

/// Try to obtain a test model path, downloading it if necessary.
///
/// Returns `Some(path)` if a model is available, or `None` if `ATHEER_TEST_MODEL`
/// is unset (local-dev scenario — tests skip gracefully).
///
/// When `ATHEER_TEST_MODEL` IS set but the model cannot be obtained, panics with
/// a descriptive message.  This ensures CI fails loudly when the download fails
/// despite explicit configuration, rather than silently skipping tests.
///
/// Behavior:
/// - If `ATHEER_TEST_MODEL` is unset → return `None`.
/// - If `ATHEER_TEST_MODEL` is set and the file exists → return `Some(path)`.
/// - If `ATHEER_TEST_MODEL` is set but the file is missing → attempt to
///   download the model by spawning `scripts/download-test-model.sh`.  Returns
///   `Some(path)` on success, panics on failure.
pub fn ensure_test_model() -> Option<PathBuf> {
    let var = std::env::var("ATHEER_TEST_MODEL").ok()?;

    let path = PathBuf::from(&var);
    if path.exists() {
        return Some(path);
    }

    // File doesn't exist — try to auto-download
    eprintln!("ATHEER_TEST_MODEL set to {var} but file not found. Attempting download...");

    let script_path = find_download_script();
    let status = Command::new(&script_path).status().unwrap_or_else(|e| {
        panic!(
            "Failed to spawn download-test-model.sh at {}: {e}",
            script_path.display()
        );
    });

    if !status.success() {
        panic!(
            "Model download script failed for {var}. Run `{}` manually.",
            script_path.display()
        );
    }

    // Check if download placed the model at the expected path
    if path.exists() {
        eprintln!("Model downloaded to {var}");
        Some(path)
    } else {
        // Script downloaded to default location; check there
        let default_path = default_model_path();
        if default_path.exists() {
            eprintln!(
                "Model downloaded to default location: {}",
                default_path.display()
            );
            Some(default_path)
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

/// Like [`ensure_test_model`] but panics if no model is available.
///
/// Use in CI or other contexts where a model is expected to always be present.
pub fn require_test_model() -> PathBuf {
    ensure_test_model()
        .expect("ATHEER_TEST_MODEL env var must be set to run tests that require a real GGUF model")
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
