// ---------------------------------------------------------------------------
// hex_literal! macro — converts a hex string to [u8; 32] at compile time.
// Must be defined before its use in const initializers.
// ---------------------------------------------------------------------------

/// Compile-time hex decoding for pinned hash literals.
macro_rules! hex_literal {
    ($s:expr) => {{
        const HASH: [u8; 32] = decode_hex_bytes($s);
        HASH
    }};
}

/// Decode a hex string (without `0x` prefix, lowercase ASCII) into `[u8; 32]`.
/// Panics if the input is not exactly 64 hex characters.
const fn decode_hex_bytes(s: &str) -> [u8; 32] {
    let bytes = s.as_bytes();
    assert!(bytes.len() == 64, "hex literal must be exactly 64 chars");
    let mut result = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        let hi = hex_val(bytes[2 * i]);
        let lo = hex_val(bytes[2 * i + 1]);
        result[i] = (hi << 4) | lo;
        i += 1;
    }
    result
}

const fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        _ => panic!("invalid hex character"),
    }
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::WebPkiServerVerifier;
use rustls::pki_types::{CertificateDer, UnixTime};
use rustls::{ClientConfig, DigitallySignedStruct, Error, RootCertStore, SignatureScheme};
use sha2::{Digest, Sha256};

/// SHA-256 hashes of DER-encoded SubjectPublicKeyInfo for pinned certificates.
///
/// Current pins (as of July 2026):
/// - Amazon RSA 2048 M04 (intermediate CA for huggingface.co)
/// - huggingface.co leaf certificate
pub const PINNED_PUBLIC_KEY_HASHES: &[[u8; 32]] = &[
    hex_literal!("1bd2cd340aa5f3dede818b1a6dab219335024c4264581ce0a04b64f17ffaefc7"),
    hex_literal!("67b9f2ef1be135d67f970de5cd02ce6669ea60738f2900e46673c712ba6ee492"),
];

/// A certificate public key pinner that builds a rustls `ClientConfig` with
/// pinned public key hashes.
///
/// During TLS handshake, the peer's certificate chain is checked: if any
/// certificate's SubjectPublicKeyInfo SHA-256 hash matches a pinned hash,
/// the connection proceeds. Otherwise, it fails with a pinning error.
pub struct CertificatePinner {
    pinned_hashes: Vec<[u8; 32]>,
}

impl CertificatePinner {
    /// Creates a new `CertificatePinner` with the given SHA-256 public key hashes.
    ///
    /// Each hash should be the SHA-256 digest of the DER-encoded
    /// SubjectPublicKeyInfo from a trusted certificate.
    pub fn new(pinned_hashes: Vec<[u8; 32]>) -> Self {
        Self { pinned_hashes }
    }

    /// Creates a `CertificatePinner` with the default HuggingFace pins:
    /// - Amazon RSA 2048 M04 (intermediate CA)
    /// - huggingface.co leaf certificate
    pub fn default_huggingface() -> Self {
        Self {
            pinned_hashes: PINNED_PUBLIC_KEY_HASHES.to_vec(),
        }
    }

    /// Builds a rustls `ClientConfig` that enforces the pinned public key hashes.
    ///
    /// The config performs standard certificate chain validation AND checks
    /// that at least one certificate in the peer's chain has a public key
    /// whose SHA-256 hash matches a pinned hash.
    pub fn build_tls_config(&self) -> Result<ClientConfig, String> {
        let crypto_provider = rustls::crypto::ring::default_provider();
        let time_provider = Arc::new(rustls::time_provider::DefaultTimeProvider);

        let mut root_store = RootCertStore::empty();
        root_store.roots = webpki_roots::TLS_SERVER_ROOTS.to_vec();

        let verifier = WebPkiServerVerifier::builder(Arc::new(root_store))
            .build()
            .map_err(|e| format!("TLS verifier builder: {e}"))?;

        let pinning_verifier = PinningVerifier {
            inner: verifier,
            pinned_hashes: self.pinned_hashes.clone(),
        };

        let config = ClientConfig::builder_with_details(Arc::new(crypto_provider), time_provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("TLS protocol versions: {e}"))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(pinning_verifier))
            .with_no_client_auth();

        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// PinningVerifier — custom ServerCertVerifier that delegates standard
// validation and then checks pinned public key hashes.
// ---------------------------------------------------------------------------

struct PinningVerifier {
    inner: Arc<WebPkiServerVerifier>,
    pinned_hashes: Vec<[u8; 32]>,
}

impl std::fmt::Debug for PinningVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinningVerifier")
            .field("pinned_hashes_count", &self.pinned_hashes.len())
            .finish()
    }
}

impl ServerCertVerifier for PinningVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        // 1. Standard certificate chain validation first.
        self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        )?;

        // 2. Collect SPKI hashes from the entire chain (leaf + intermediates).
        let peer_hashes = collect_spki_hashes(end_entity, intermediates);

        // 3. Check if ANY peer hash matches ANY pinned hash.
        let matched = peer_hashes
            .iter()
            .any(|peer_hash| self.pinned_hashes.contains(peer_hash));

        if !matched {
            let peer_hex: Vec<String> = peer_hashes.iter().map(|h| hex::encode(h)).collect();
            let pinned_hex: Vec<String> =
                self.pinned_hashes.iter().map(|h| hex::encode(h)).collect();
            let hostname = server_name.to_str().to_string();
            return Err(Error::General(format!(
                "TLS certificate pinning failed for {hostname}: \
                 peer key hashes [{peer_hashes}] do not match any pinned key [{pinned}]",
                hostname = hostname,
                peer_hashes = peer_hex.join(", "),
                pinned = pinned_hex.join(", "),
            )));
        }

        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

// ---------------------------------------------------------------------------
// Helper: extract SHA-256 hash of SubjectPublicKeyInfo from DER certs.
// ---------------------------------------------------------------------------

/// Collects the SHA-256 hashes of every certificate's SubjectPublicKeyInfo
/// in the chain (leaf first, then intermediates).
fn collect_spki_hashes(
    end_entity: &CertificateDer<'_>,
    intermediates: &[CertificateDer<'_>],
) -> Vec<[u8; 32]> {
    let mut hashes = Vec::new();
    if let Some(hash) = cert_spki_sha256(end_entity) {
        hashes.push(hash);
    }
    for cert in intermediates {
        if let Some(hash) = cert_spki_sha256(cert) {
            hashes.push(hash);
        }
    }
    hashes
}

/// Parses a DER-encoded X.509 certificate and returns the SHA-256 hash of
/// its SubjectPublicKeyInfo, or `None` if parsing fails.
fn cert_spki_sha256(cert: &CertificateDer<'_>) -> Option<[u8; 32]> {
    let end_entity = webpki::EndEntityCert::try_from(cert).ok()?;
    let spki = end_entity.subject_public_key_info();
    let mut hasher = Sha256::new();
    hasher.update(spki.as_ref());
    let result = hasher.finalize();
    Some(result.into())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_huggingface_returns_two_hashes() {
        let pinner = CertificatePinner::default_huggingface();
        assert_eq!(pinner.pinned_hashes.len(), 2);
    }

    #[test]
    fn test_new_stores_hashes() {
        let hash = [0u8; 32];
        let pinner = CertificatePinner::new(vec![hash]);
        assert_eq!(pinner.pinned_hashes.len(), 1);
        assert_eq!(pinner.pinned_hashes[0], hash);
    }

    #[test]
    fn test_build_tls_config_ok() {
        let pinner = CertificatePinner::default_huggingface();
        let config = pinner.build_tls_config();
        assert!(config.is_ok());
    }

    #[test]
    fn test_empty_pins_still_builds() {
        let pinner = CertificatePinner::new(vec![]);
        let config = pinner.build_tls_config();
        assert!(config.is_ok());
    }

    #[test]
    fn test_hex_literal_decode() {
        let hash = hex_literal!("1bd2cd340aa5f3dede818b1a6dab219335024c4264581ce0a04b64f17ffaefc7");
        assert_eq!(hash[0], 0x1b);
        assert_eq!(hash[1], 0xd2);
        assert_eq!(hash[31], 0xc7);
    }

    #[test]
    fn test_pinned_hashes_const_is_valid() {
        for hash in PINNED_PUBLIC_KEY_HASHES {
            assert!(hash.iter().any(|&b| b != 0), "hash should not be all zeros");
        }
    }

    #[test]
    fn test_cert_spki_sha256_none_for_invalid_der() {
        let invalid = CertificateDer::from(vec![0, 1, 2, 3]);
        assert!(cert_spki_sha256(&invalid).is_none());
    }

    #[test]
    fn test_tls_pinning_failed_error_display() {
        let err = crate::AtheerCoreError::TlsPinningFailed {
            hostname: "huggingface.co".to_string(),
            peer_hash: "abcdef".to_string(),
            pinned_hashes: "123456,789abc".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("huggingface.co"));
        assert!(msg.contains("abcdef"));
        assert!(msg.contains("123456"));
    }
}
