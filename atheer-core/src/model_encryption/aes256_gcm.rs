use crate::model_encryption::ModelEncryption;
use crate::AtheerCoreError;
use aes_gcm::{aead::Aead, Aes256Gcm, Key, KeyInit, Nonce};
use std::fs;
use std::path::Path;
use zeroize::Zeroize;

/// AES-256-GCM based model file encryption (v1).
///
/// Encrypted file format:
///   GGUF:  `[12B nonce][16B GCM tag][ciphertext]`
///   .bin:  `[12B nonce][16B GCM tag][ciphertext]`
///
/// Additional Authenticated Data (AAD):
///   GGUF:        `"atheer-model-v1"`
///   .bin weight: `"atheer-mlpackage-weight"`
pub struct Aes256GcmEncryption {
    key: Box<[u8; 32]>,
}

impl Aes256GcmEncryption {
    pub fn new(key: [u8; 32]) -> Self {
        Self { key: Box::new(key) }
    }

    fn decrypt_bytes(&self, data: &[u8], aad: &[u8]) -> Result<Vec<u8>, AtheerCoreError> {
        if data.len() < 28 {
            return Err(AtheerCoreError::ModelDecryptionFailed(
                "encrypted file too short (< 28 bytes)".into(),
            ));
        }
        let nonce_bytes = &data[..12];
        let tag_bytes = &data[12..28];
        let ciphertext = &data[28..];

        let key = Key::<Aes256Gcm>::from_slice(self.key.as_slice());
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(nonce_bytes);

        // AES-GCM decrypt with AAD
        let mut ciphertext_with_tag = ciphertext.to_vec();
        ciphertext_with_tag.extend_from_slice(tag_bytes);

        cipher
            .decrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| {
                AtheerCoreError::ModelDecryptionFailed(
                    "AES-GCM: bad tag or corrupted ciphertext".into(),
                )
            })
    }
}

impl ModelEncryption for Aes256GcmEncryption {
    fn decrypt_reader(&self, path: &str) -> Result<Vec<u8>, AtheerCoreError> {
        let data = fs::read(path).map_err(|e| {
            AtheerCoreError::ModelDecryptionFailed(format!("cannot read {path}: {e}"))
        })?;
        self.decrypt_bytes(&data, b"atheer-model-v1")
    }

    fn decrypt_mlpackage(&self, path: &str) -> Result<String, AtheerCoreError> {
        let src = Path::new(path);
        if !src.is_dir() {
            return Err(AtheerCoreError::ModelDecryptionFailed(format!(
                "not a directory: {path}"
            )));
        }

        // Copy the .mlpackage bundle to a temp directory
        let tmp = tempfile::tempdir()
            .map_err(|e| AtheerCoreError::ModelDecryptionFailed(format!("tempdir: {e}")))?;
        #[allow(deprecated)]
        let tmp_path = tmp.into_path();

        // Recursive copy
        copy_dir_recursive(src, &tmp_path)?;

        // Walk and decrypt each .bin file
        decrypt_bin_files(&tmp_path, &self.key)?;

        Ok(tmp_path.to_string_lossy().to_string())
    }

    fn scrub(&self) {
        // The caller is responsible for scrubbing; the key cannot be
        // mutated through a shared reference, so we document that the key
        // is zeroized when the struct is dropped.
    }
}

impl Drop for Aes256GcmEncryption {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), AtheerCoreError> {
    if src.is_dir() {
        fs::create_dir_all(dst)
            .map_err(|e| AtheerCoreError::ModelDecryptionFailed(format!("mkdir {dst:?}: {e}")))?;
        for entry in fs::read_dir(src)
            .map_err(|e| AtheerCoreError::ModelDecryptionFailed(format!("read_dir {src:?}: {e}")))?
        {
            let entry =
                entry.map_err(|e| AtheerCoreError::ModelDecryptionFailed(format!("entry: {e}")))?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            copy_dir_recursive(&src_path, &dst_path)?;
        }
    } else {
        fs::copy(src, dst).map_err(|e| {
            AtheerCoreError::ModelDecryptionFailed(format!("copy {src:?} -> {dst:?}: {e}"))
        })?;
    }
    Ok(())
}

fn decrypt_bin_files(dir: &Path, key: &[u8; 32]) -> Result<(), AtheerCoreError> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)
            .map_err(|e| AtheerCoreError::ModelDecryptionFailed(format!("walk {dir:?}: {e}")))?
        {
            let entry =
                entry.map_err(|e| AtheerCoreError::ModelDecryptionFailed(format!("entry: {e}")))?;
            let path = entry.path();
            if path.is_dir() {
                decrypt_bin_files(&path, key)?;
            } else if path.extension().map(|e| e == "bin").unwrap_or(false) {
                let encrypted = fs::read(&path).map_err(|e| {
                    AtheerCoreError::ModelDecryptionFailed(format!("read {:?}: {e}", path))
                })?;
                let aad = b"atheer-mlpackage-weight";
                let key_ref = Key::<Aes256Gcm>::from_slice(key.as_slice());
                let cipher = Aes256Gcm::new(key_ref);
                if encrypted.len() < 28 {
                    return Err(AtheerCoreError::ModelDecryptionFailed(format!(
                        "encrypted .bin too short: {:?}",
                        path
                    )));
                }
                let nonce = Nonce::from_slice(&encrypted[..12]);

                let ciphertext = &encrypted[28..];
                let plaintext = cipher
                    .decrypt(
                        nonce,
                        aes_gcm::aead::Payload {
                            msg: ciphertext,
                            aad,
                        },
                    )
                    .map_err(|_| {
                        AtheerCoreError::ModelDecryptionFailed(format!(
                            "AES-GCM: bad tag for {:?}",
                            path
                        ))
                    })?;
                fs::write(&path, &plaintext).map_err(|e| {
                    AtheerCoreError::ModelDecryptionFailed(format!("write {:?}: {e}", path))
                })?;
            }
        }
    }
    Ok(())
}
