use crate::l3_compressed::L3CompressedStorage;
use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Key, Nonce,
};
use std::path::PathBuf;
use zeroize::Zeroize;

/// Wraps `L3CompressedStorage` with AES-256-GCM encryption of snapshot data.
///
/// On-disk format per snapshot:
///   `[12 B nonce][AES-256-GCM ciphertext]`
///
/// Data flow:
///   `snapshot()`  : serialize → LZ4 compress → AES-256-GCM encrypt → write
///   `restore()`   : read → AES-256-GCM decrypt → LZ4 decompress → parse
///
/// AAD tag: `b"atheer-cache-v1"` — distinct from model encryption
/// (`"atheer-model-v1"`) so cross-type swap attacks cause a decryption failure.
pub struct EncryptedStore {
    inner: L3CompressedStorage,
    key: Box<[u8; 32]>,
}

impl EncryptedStore {
    /// Create a new `EncryptedStore` backed by the given directory.
    ///
    /// The directory is created if it does not exist.  The caller is
    /// responsible for providing a suitable 32-byte AES-256 key.
    pub fn new(storage_dir: PathBuf, key: [u8; 32]) -> std::io::Result<Self> {
        let inner = L3CompressedStorage::new(storage_dir)?;
        Ok(Self {
            inner,
            key: Box::new(key),
        })
    }

    /// Compress then encrypt `plaintext`.
    ///
    /// Returns `[12 B nonce || ciphertext + GCM tag]`.
    fn encrypt(&self, plaintext: &[u8]) -> std::io::Result<Vec<u8>> {
        // 1. LZ4 compress
        let compressed = lz4::block::compress(plaintext, None, true)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // 2. AES-256-GCM encrypt with a fresh random nonce
        let key = Key::<Aes256Gcm>::from_slice(self.key.as_ref());
        let cipher = Aes256Gcm::new(key);
        let nonce_bytes: [u8; 12] = {
            // Use a random nonce — `aes_gcm` 0.10 does not export a
            // `generate_nonce()` convenience fn, so we generate bytes
            // ourselves (same approach as the upstream `aes_gcm` examples).
            use rand::Rng;
            let mut buf = [0u8; 12];
            rand::thread_rng().fill(&mut buf);
            buf
        };
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(
                nonce,
                Payload {
                    msg: &compressed,
                    aad: b"atheer-cache-v1",
                },
            )
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // 3. Prepend nonce
        let mut out = Vec::with_capacity(12 + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt then decompress `data`.
    ///
    /// Expects `data` to be `[12 B nonce || ciphertext + GCM tag]`.
    fn decrypt(&self, data: &[u8]) -> std::io::Result<Vec<u8>> {
        if data.len() < 12 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "encrypted data too short: missing nonce",
            ));
        }

        // 1. Split nonce and ciphertext
        let (nonce_bytes, ciphertext) = data.split_at(12);

        // 2. AES-256-GCM decrypt
        let key = Key::<Aes256Gcm>::from_slice(self.key.as_ref());
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(nonce_bytes);

        let compressed = cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad: b"atheer-cache-v1",
                },
            )
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "AES-GCM decryption failed: bad key, corrupted data, or nonce mismatch",
                )
            })?;

        // 3. LZ4 decompress
        let plaintext = lz4::block::decompress(&compressed, None)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(plaintext)
    }

    /// Write a snapshot: compress → encrypt → persist to disk.
    ///
    /// The snapshot file is written to `{storage_dir}/{model_id}_{uuid}.snap` —
    /// same naming convention as `L3CompressedStorage` but with encrypted content.
    pub fn snapshot(&mut self, model_id: &str, data: &[u8]) -> std::io::Result<String> {
        let encrypted = self.encrypt(data)?;
        self.inner.snapshot_raw(model_id, &encrypted)
    }

    /// Read and decrypt a snapshot from disk.
    pub fn restore(&self, snapshot_id: &str) -> std::io::Result<Vec<u8>> {
        let encrypted = self.inner.restore_raw(snapshot_id)?;
        self.decrypt(&encrypted)
    }

    /// Total size on disk of all snapshot files in bytes.
    pub fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }
}

impl Drop for EncryptedStore {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        k.copy_from_slice(b"0123456789abcdef0123456789abcdef");
        k
    }

    fn other_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        k.copy_from_slice(b"fedcba9876543210fedcba9876543210");
        k
    }

    fn temp_store(key: [u8; 32]) -> (EncryptedStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = EncryptedStore::new(dir.path().join("l3"), key).unwrap();
        (store, dir)
    }

    // ── 4.1 Roundtrip with known key ────────────────────────────────────────

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let (store, _dir) = temp_store(test_key());
        let data = b"Hello, encrypted world!";
        let encrypted = store.encrypt(data).unwrap();
        assert!(encrypted.len() > 12, "must include nonce");
        let decrypted = store.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let (mut store, _dir) = temp_store(test_key());
        let data = b"snapshot roundtrip data";
        let id = store.snapshot("test-model", data).unwrap();
        let restored = store.restore(&id).unwrap();
        assert_eq!(restored, data);
    }

    // ── 4.2 Wrong key returns error ─────────────────────────────────────────

    #[test]
    fn test_wrong_key_fails() {
        let (store, _dir) = temp_store(test_key());
        let data = b"secret data";
        let encrypted = store.encrypt(data).unwrap();
        // Decrypt with a different store that has a different key
        let other = EncryptedStore::new(_dir.path().join("l3"), other_key()).unwrap();
        let result = other.decrypt(&encrypted);
        assert!(result.is_err(), "decrypt with wrong key should fail");
    }

    // ── 4.3 Corrupted file returns error ────────────────────────────────────

    #[test]
    fn test_corrupted_file_fails() {
        let (mut store, dir) = temp_store(test_key());
        let data = b"data for corruption test";
        let id = store.snapshot("test-model", data).unwrap();

        // Find the snapshot file and corrupt it
        let snap_path = dir.path().join("l3");
        // Walk the directory for the snapshot file
        let mut file_path = None;
        for entry in fs::read_dir(&snap_path).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            if name.contains(&id) {
                file_path = Some(entry.path());
                break;
            }
        }
        let path = file_path.expect("snapshot file not found");

        // Corrupt the file — flip a byte in the ciphertext region (after nonce)
        let mut contents = fs::read(&path).unwrap();
        if contents.len() > 12 {
            // Flip a byte somewhere after the nonce
            let pos = 12 + (contents.len() - 12) / 2;
            contents[pos] ^= 0xFF;
            fs::write(&path, &contents).unwrap();
        }

        let result = store.restore(&id);
        assert!(result.is_err(), "restore of corrupted file should fail");
    }

    // ── 4.4 Nonce uniqueness across sequential snapshots ────────────────────

    #[test]
    fn test_nonce_uniqueness() {
        let (store, _dir) = temp_store(test_key());
        let payload = b"identical payload";

        let enc1 = store.encrypt(payload).unwrap();
        let enc2 = store.encrypt(payload).unwrap();

        // First 12 bytes = nonce
        assert_ne!(&enc1[..12], &enc2[..12], "nonces MUST differ");
        // Ciphertext must also differ (due to different nonce)
        assert_ne!(enc1, enc2, "full ciphertext MUST differ");
    }

    // ── 4.5 Large payload roundtrip ─────────────────────────────────────────

    #[test]
    fn test_large_payload_roundtrip() {
        let (store, _dir) = temp_store(test_key());
        let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let encrypted = store.encrypt(&data).unwrap();
        let decrypted = store.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    // ── 4.6 Empty payload roundtrip ─────────────────────────────────────────

    #[test]
    fn test_empty_payload_roundtrip() {
        let (store, _dir) = temp_store(test_key());
        let data = b"";
        let encrypted = store.encrypt(data).unwrap();
        let decrypted = store.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }
}
