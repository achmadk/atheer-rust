/// Configuration for decrypting model files at load time.
///
/// This enum is designed to be UniFFI-compatible (all variants use
/// record-like fields with `Vec<u8>` instead of `[u8; N]`).
/// It is re-exported by `atheer-ffi` for use from Swift and Kotlin.
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum ModelCredential {
    /// Key is provided by a remote server and stored in the platform
    /// secure enclave (Keychain / KeyStore) under `key_id`.
    ServerDistributed {
        key_id: String,
        nonce: Vec<u8>,
        wrapped_key: Option<Vec<u8>>,
    },
    /// Key is derived from the device identity (device UID + model hash + salt).
    DeviceDerived {
        salt: Vec<u8>,
        nonce: Vec<u8>,
    },
    /// Custom encryption scheme registered by the host app (Swift/Kotlin).
    Custom {
        scheme_name: String,
        config: Vec<u8>,
    },
}
