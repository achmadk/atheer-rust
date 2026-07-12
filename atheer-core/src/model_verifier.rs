use crate::security::SecurityError;
use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Verifies Ed25519 signatures over model files.
///
/// The signing scheme is "hash-then-sign": the model file is SHA-256 hashed
/// via streaming reads (64 KB chunks), then the 32-byte hash is verified
/// against the Ed25519 signature using the trusted public key.
#[derive(Debug)]
pub struct ModelVerifier {
    trusted_key: VerifyingKey,
}

impl ModelVerifier {
    /// Construct a verifier from raw 32-byte Ed25519 public key bytes.
    pub fn new(public_key: &[u8]) -> Result<Self, SecurityError> {
        let key_bytes: [u8; 32] = public_key.try_into().map_err(|_| {
            SecurityError::KeyParseFailed("Ed25519 public key must be exactly 32 bytes".into())
        })?;
        let trusted_key = VerifyingKey::from_bytes(&key_bytes).map_err(|e| {
            SecurityError::KeyParseFailed(format!("Invalid Ed25519 public key: {e}"))
        })?;
        Ok(Self { trusted_key })
    }

    /// Verify a model file against its detached Ed25519 signature.
    ///
    /// 1. Streaming SHA-256 hash of the model file (64 KB chunks)
    /// 2. Read the 64-byte Ed25519 signature from `signature_path`
    /// 3. `Ed25519::verify_strict(hash, signature, trusted_key)`
    pub fn verify_detached(
        &self,
        model_path: &Path,
        signature_path: &Path,
    ) -> Result<(), SecurityError> {
        // 1. Streaming SHA-256 hash of the model file
        let hash = hash_file(model_path)?;

        // 2. Read the detached signature file
        if !signature_path.exists() {
            return Err(SecurityError::SignatureVerificationFailed(format!(
                "Signature file not found: {:?}",
                signature_path
            )));
        }
        let sig_bytes = std::fs::read(signature_path).map_err(|e| {
            SecurityError::SignatureVerificationFailed(format!(
                "Failed to read signature file {:?}: {e}",
                signature_path
            ))
        })?;

        let signature =
            Signature::from_slice(&sig_bytes).map_err(|_| SecurityError::SignatureInvalid)?;

        // 3. Verify the Ed25519 signature over the SHA-256 hash
        self.trusted_key
            .verify_strict(&hash, &signature)
            .map_err(|_| SecurityError::SignatureInvalid)
    }
}

/// Compute the SHA-256 hash of a file via streaming reads (64 KB chunks).
///
/// Returns the raw 32-byte hash as a `Vec<u8>`.
pub fn hash_file(path: &Path) -> Result<Vec<u8>, SecurityError> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path).map_err(|e| {
        SecurityError::SignatureVerificationFailed(format!(
            "Failed to open model file for hashing {:?}: {e}",
            path
        ))
    })?;
    let mut buf = [0u8; 65536]; // 64 KB chunks
    loop {
        let n = file.read(&mut buf).map_err(|e| {
            SecurityError::SignatureVerificationFailed(format!(
                "Failed to read model file for hashing {:?}: {e}",
                path
            ))
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        let counter = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "atheer_sig_test_{}_{}",
            std::process::id(),
            counter
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("test_model.gguf");
        let sig = dir.join("test_model.gguf.sig");
        (model, sig)
    }

    /// Convenience: hash content bytes as if they were a file.
    fn hash_bytes(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    fn test_keypair() -> (ed25519_dalek::SigningKey, ed25519_dalek::VerifyingKey) {
        let signing = ed25519_dalek::SigningKey::from_bytes(&[0xabu8; 32]);
        let verifying = signing.verifying_key();
        (signing, verifying)
    }

    fn test_keypair_alt() -> (ed25519_dalek::SigningKey, ed25519_dalek::VerifyingKey) {
        let signing = ed25519_dalek::SigningKey::from_bytes(&[0xbcu8; 32]);
        let verifying = signing.verifying_key();
        (signing, verifying)
    }

    #[test]
    fn test_verify_valid_signature() {
        let (signing_key, verifying_key) = test_keypair();

        let (model_path, sig_path) = test_paths();

        // Write test model data
        let model_data = b"Hello, this is a test model file!";
        std::fs::write(&model_path, model_data).unwrap();

        // Sign the SHA-256 hash
        let hash = hash_file(&model_path).unwrap();
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(&hash);
        std::fs::write(&sig_path, signature.to_bytes()).unwrap();

        // Verify
        let verifier = ModelVerifier::new(&verifying_key.to_bytes()).unwrap();
        let result = verifier.verify_detached(&model_path, &sig_path);
        assert!(
            result.is_ok(),
            "Valid signature should verify: {:?}",
            result
        );

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&sig_path);
    }

    #[test]
    fn test_verify_tampered_file() {
        let (signing_key, verifying_key) = test_keypair();

        let (model_path, sig_path) = test_paths();

        // Write and sign original data
        let model_data = b"Original model file content here";
        std::fs::write(&model_path, model_data).unwrap();
        let hash = hash_file(&model_path).unwrap();
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(&hash);
        std::fs::write(&sig_path, signature.to_bytes()).unwrap();

        // Tamper with model file (single byte changed)
        let tampered = b"Tampered model file content!!!";
        std::fs::write(&model_path, tampered).unwrap();

        // Verify should fail
        let verifier = ModelVerifier::new(&verifying_key.to_bytes()).unwrap();
        let result = verifier.verify_detached(&model_path, &sig_path);
        assert!(result.is_err(), "Tampered file should fail verification");
        assert!(
            matches!(result.unwrap_err(), SecurityError::SignatureInvalid),
            "Wrong error variant for tampered file"
        );

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&sig_path);
    }

    #[test]
    fn test_verify_wrong_key() {
        let (signing_key, _verifying_key) = test_keypair();
        let (_wrong_signing, wrong_verifying) = test_keypair_alt();

        let (model_path, sig_path) = test_paths();

        let model_data = b"Data signed by one key, verified by another";
        std::fs::write(&model_path, model_data).unwrap();
        let hash = hash_file(&model_path).unwrap();
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(&hash);
        std::fs::write(&sig_path, signature.to_bytes()).unwrap();

        // Verify with wrong key
        let verifier = ModelVerifier::new(&wrong_verifying.to_bytes()).unwrap();
        let result = verifier.verify_detached(&model_path, &sig_path);
        assert!(result.is_err(), "Wrong key should fail verification");

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&sig_path);
    }

    #[test]
    fn test_verify_invalid_sig_bytes() {
        let (_signing_key, verifying_key) = test_keypair();

        let (model_path, sig_path) = test_paths();

        let model_data = b"Garbage signature test";
        std::fs::write(&model_path, model_data).unwrap();

        std::fs::write(&sig_path, b"not a valid ed25519 signature!!!").unwrap();

        let verifier = ModelVerifier::new(&verifying_key.to_bytes()).unwrap();
        let result = verifier.verify_detached(&model_path, &sig_path);
        assert!(result.is_err(), "Invalid signature bytes should fail");

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&sig_path);
    }

    #[test]
    fn test_verify_missing_sig_file() {
        let (_signing_key, verifying_key) = test_keypair();

        let (model_path, sig_path) = test_paths();

        let model_data = b"Missing sig file test";
        std::fs::write(&model_path, model_data).unwrap();

        // Don't create sig file at all

        let verifier = ModelVerifier::new(&verifying_key.to_bytes()).unwrap();
        let result = verifier.verify_detached(&model_path, &sig_path);
        assert!(result.is_err(), "Missing sig file should fail");
        assert!(
            matches!(
                result.unwrap_err(),
                SecurityError::SignatureVerificationFailed(_)
            ),
            "Wrong error variant for missing sig file"
        );

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_key_parse_failure() {
        // Too short
        let too_short = b"too short";
        let result = ModelVerifier::new(too_short);
        assert!(
            matches!(result.unwrap_err(), SecurityError::KeyParseFailed(_)),
            "Too-short key should return KeyParseFailed"
        );

        // Too long
        let too_long = &[0u8; 64];
        let result = ModelVerifier::new(too_long);
        assert!(
            matches!(result.unwrap_err(), SecurityError::KeyParseFailed(_)),
            "Too-long key should return KeyParseFailed"
        );
    }

    #[test]
    fn test_hash_file_equivalence() {
        let (model_path, _sig_path) = test_paths();

        let data = b"Test data for hash consistency check";
        std::fs::write(&model_path, data).unwrap();

        let hash_result = hash_file(&model_path).unwrap();
        let hash_direct = hash_bytes(data);

        assert_eq!(
            hash_result, hash_direct,
            "Streaming hash should match direct hash"
        );

        let _ = std::fs::remove_file(&model_path);
    }
}
