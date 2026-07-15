use crate::{AtheerCoreError, Result};
use filetime::FileTime;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::cert_pinner::CertificatePinner;

/// Configuration describing a downloadable model from HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub sha256: String,
    pub quantization: String,
    pub context_size: usize,
    pub recommended_backend: Option<String>,
}

impl ModelConfig {
    pub fn download_url(&self) -> String {
        let parts: Vec<&str> = self.name.splitn(2, '/').collect();
        if parts.len() == 2 {
            format!(
                "https://huggingface.co/{org}/{repo}/resolve/main/{file}",
                org = parts[0],
                repo = parts[1],
                file = self.filename()
            )
        } else {
            format!(
                "https://huggingface.co/{name}/{name}/resolve/main/{file}",
                name = self.name,
                file = self.filename()
            )
        }
    }

    pub fn filename(&self) -> String {
        format!("{}-{}.gguf", self.name.replace('/', "_"), self.quantization)
    }
}

/// A registry that downloads, caches, and manages HuggingFace GGUF models.
///
/// Cache location: `~/.atheer/models/` (configurable)
/// Max cache size: 10 GB (configurable)
pub struct ModelRegistry {
    cache_dir: PathBuf,
    max_cache_size: u64,
    client: reqwest::blocking::Client,
}

impl ModelRegistry {
    pub fn new(
        cache_dir: Option<PathBuf>,
        max_cache_size: Option<u64>,
        pinner: Option<&CertificatePinner>,
    ) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(Self::default_cache_dir);
        let max_cache_size = max_cache_size.unwrap_or(10_737_418_240);
        let client = match pinner {
            Some(pinner) => {
                let tls_config = pinner
                    .build_tls_config()
                    .expect("CertificatePinner TLS config should build");
                reqwest::blocking::Client::builder()
                    .user_agent("atheer-rust/0.1.0")
                    .use_preconfigured_tls(tls_config)
                    .build()
                    .expect("reqwest Client should build")
            }
            None => reqwest::blocking::Client::builder()
                .user_agent("atheer-rust/0.1.0")
                .build()
                .expect("reqwest Client should build"),
        };
        Self {
            cache_dir,
            max_cache_size,
            client,
        }
    }

    /// Creates a new `ModelRegistry` with HuggingFace certificate pinning enabled.
    ///
    /// Uses the default pinned public key hashes (Amazon RSA 2048 M04 intermediate
    /// CA + huggingface.co leaf certificate) to prevent MITM attacks on model
    /// downloads.
    pub fn with_pinning(cache_dir: Option<PathBuf>, max_cache_size: Option<u64>) -> Result<Self> {
        let cache_dir = cache_dir.unwrap_or_else(Self::default_cache_dir);
        let max_cache_size = max_cache_size.unwrap_or(10_737_418_240);
        let pinner = CertificatePinner::default_huggingface();
        let tls_config = pinner
            .build_tls_config()
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("build TLS config: {e}")))?;
        let client = reqwest::blocking::Client::builder()
            .user_agent("atheer-rust/0.1.0")
            .use_preconfigured_tls(tls_config)
            .build()
            .expect("reqwest Client should build");
        Ok(Self {
            cache_dir,
            max_cache_size,
            client,
        })
    }

    pub fn default_cache_dir() -> PathBuf {
        let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
        base.join(".atheer").join("models")
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn cache_hit(&self, model_name: &str) -> Option<PathBuf> {
        let model_dir = self.model_dir(model_name);
        if !model_dir.exists() {
            return None;
        }
        let entries = fs::read_dir(&model_dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "gguf") {
                let _ = Self::touch_timestamp(&path);
                return Some(path);
            }
        }
        None
    }

    pub fn resolve(&self, model_name: &str, config: &ModelConfig) -> Result<PathBuf> {
        if let Some(cached) = self.cache_hit(model_name) {
            tracing::info!(model = model_name, "cache hit");
            return Ok(cached);
        }
        tracing::info!(model = model_name, "cache miss, downloading");
        self.download(model_name, config)
    }

    pub fn download(&self, model_name: &str, config: &ModelConfig) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);
        fs::create_dir_all(&model_dir)
            .map_err(|e| AtheerCoreError::CacheError(format!("create cache dir: {e}")))?;

        let final_path = model_dir.join(config.filename());

        if final_path.exists() {
            tracing::info!(path = %final_path.display(), "partial download exists, attempting resume");
            return self.resume_download(model_name, config);
        }

        let url = config.download_url();
        tracing::info!(url = %url, "starting download");

        let response = self
            .client
            .get(&url)
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("HTTP request to {url}: {e}")))?;

        let total_size = response.content_length().unwrap_or(0);
        let mut hasher = Sha256::new();
        let mut downloaded: u64 = 0;
        let mut file = fs::File::create(&final_path)
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("create file: {e}")))?;

        let body = response
            .bytes()
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("read response: {e}")))?;
        hasher.update(&body);
        file.write_all(&body)
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("write body: {e}")))?;
        downloaded += body.len() as u64;
        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            tracing::debug!(model = model_name, pct = %format!("{:.1}", pct), "download progress");
        }

        self.verify_hash(&final_path, &config.sha256)?;

        if let Err(e) = Self::touch_timestamp(&final_path) {
            tracing::warn!("touch timestamp: {e}");
        }

        self.enforce_cache_limit()?;

        tracing::info!(model = model_name, path = %final_path.display(), "download complete");
        Ok(final_path)
    }

    pub fn resume_download(&self, model_name: &str, config: &ModelConfig) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);
        let final_path = model_dir.join(config.filename());
        let existing_len = fs::metadata(&final_path).ok().map(|m| m.len()).unwrap_or(0);

        let url = config.download_url();
        let range_header = format!("bytes={}-", existing_len);

        let response = self
            .client
            .get(&url)
            .header("Range", &range_header)
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| {
                AtheerCoreError::DownloadFailed(format!("HTTP Range request to {url}: {e}"))
            })?;

        let total_size = response
            .content_length()
            .map(|cl| cl + existing_len)
            .unwrap_or(0);

        let mut hasher = Sha256::new();
        if existing_len > 0 {
            let existing_data = fs::read(&final_path)
                .map_err(|e| AtheerCoreError::DownloadFailed(format!("read existing: {e}")))?;
            hasher.update(&existing_data);
        }

        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&final_path)
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("open for append: {e}")))?;

        let mut downloaded = existing_len;
        let body = response
            .bytes()
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("read resume chunk: {e}")))?;
        hasher.update(&body);
        file.write_all(&body)
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("write resume: {e}")))?;
        downloaded += body.len() as u64;
        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            tracing::debug!(model = model_name, pct = %format!("{:.1}", pct), "download progress");
        }

        self.verify_hash(&final_path, &config.sha256)?;

        if let Err(e) = Self::touch_timestamp(&final_path) {
            tracing::warn!("touch timestamp: {e}");
        }

        self.enforce_cache_limit()?;

        tracing::info!(model = model_name, path = %final_path.display(), "resume complete");
        Ok(final_path)
    }

    pub fn verify_sha256(&self, path: &Path, expected: &str) -> Result<()> {
        self.verify_hash(path, expected)
    }

    pub fn load_model(
        &self,
        model_name: &str,
        config: &ModelConfig,
        device: &candle_core::Device,
    ) -> Result<crate::Model> {
        let path = self.resolve(model_name, config)?;
        crate::Model::from_gguf(&path, device, None)
    }

    fn enforce_cache_limit(&self) -> Result<()> {
        let total = self.cache_size()?;
        if total <= self.max_cache_size {
            return Ok(());
        }
        let overage = total - self.max_cache_size;
        let evicted = self.evict_bytes(overage)?;
        tracing::info!(evicted_mb = %(evicted / 1_048_576), "LRU eviction");
        Ok(())
    }

    fn evict_bytes(&self, target_bytes: u64) -> Result<u64> {
        let mut candidates: Vec<(PathBuf, SystemTime)> = Vec::new();
        self.collect_gguf_files(&self.cache_dir, &mut candidates)
            .map_err(|e| AtheerCoreError::CacheError(e.to_string()))?;

        candidates.sort_by_key(|(_, time)| *time);

        let mut freed: u64 = 0;
        for (path, _) in &candidates {
            if freed >= target_bytes {
                break;
            }
            if let Ok(meta) = fs::metadata(path) {
                let len = meta.len();
                if fs::remove_file(path).is_ok() {
                    freed += len;
                    tracing::debug!(file = %path.display(), "evicted");
                }
            }
        }
        Ok(freed)
    }

    fn cache_size(&self) -> Result<u64> {
        let mut files = Vec::new();
        self.collect_gguf_files(&self.cache_dir, &mut files)
            .map_err(|e| AtheerCoreError::CacheError(e.to_string()))?;
        let total: u64 = files
            .iter()
            .filter_map(|(p, _)| fs::metadata(p).ok().map(|m| m.len()))
            .sum();
        Ok(total)
    }

    fn collect_gguf_files(
        &self,
        dir: &Path,
        acc: &mut Vec<(PathBuf, SystemTime)>,
    ) -> io::Result<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.collect_gguf_files(&path, acc)?;
            } else if path.extension().map_or(false, |e| e == "gguf") {
                let modified = entry
                    .metadata()?
                    .accessed()
                    .unwrap_or_else(|_| SystemTime::UNIX_EPOCH);
                acc.push((path, modified));
            }
        }
        Ok(())
    }

    fn verify_hash(&self, path: &Path, expected: &str) -> Result<()> {
        let actual = Self::sha256_file(path)?;
        if actual != expected {
            return Err(AtheerCoreError::ChecksumMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
        Ok(())
    }

    fn sha256_file(path: &Path) -> Result<String> {
        let data = fs::read(path)
            .map_err(|e| AtheerCoreError::DownloadFailed(format!("read for hash: {e}")))?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        Ok(hex::encode(hasher.finalize()))
    }

    fn touch_timestamp(path: &Path) -> io::Result<()> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let ft = FileTime::from_unix_time(now as i64, 0);
        filetime::set_file_times(path, ft, ft)
    }

    fn model_dir(&self, model_name: &str) -> PathBuf {
        self.cache_dir.join(model_name.replace('/', "_"))
    }
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("cache_dir", &self.cache_dir)
            .field("max_cache_size", &self.max_cache_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test/placeholder".to_string(),
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
            quantization: "Q4_K_M".to_string(),
            context_size: 4096,
            recommended_backend: Some("cpu".to_string()),
        }
    }

    #[test]
    fn test_model_config_download_url() {
        let cfg = test_config();
        let url = cfg.download_url();
        assert!(url.contains("huggingface.co"));
        assert!(url.contains("test/placeholder"));
        assert!(url.contains("Q4_K_M"));
    }

    #[test]
    fn test_model_config_filename() {
        let cfg = test_config();
        let name = cfg.filename();
        assert!(name.contains("test_placeholder"));
        assert!(name.contains("Q4_K_M"));
        assert!(name.ends_with(".gguf"));
    }

    #[test]
    fn test_registry_default_cache_dir() {
        let dir = ModelRegistry::default_cache_dir();
        assert!(dir.to_string_lossy().ends_with(".atheer/models"));
    }

    #[test]
    fn test_registry_new_with_defaults() {
        let reg = ModelRegistry::new(None, None, None);
        assert_eq!(reg.max_cache_size, 10_737_418_240);
        assert!(reg
            .cache_dir()
            .to_string_lossy()
            .ends_with(".atheer/models"));
    }

    #[test]
    fn test_cache_hit_miss() {
        let dir = tempdir().unwrap();
        let reg = ModelRegistry::new(Some(dir.path().join("cache")), None, None);
        assert!(reg.cache_hit("test/model").is_none());

        let model_dir = dir.path().join("cache").join("test_model");
        fs::create_dir_all(&model_dir).unwrap();
        let gguf_path = model_dir.join("test_model-Q4_K_M.gguf");
        fs::write(&gguf_path, &[0u8; 32]).unwrap();
        assert!(reg.cache_hit("test/model").is_some());
    }

    #[test]
    fn test_verify_sha256_mismatch() {
        let dir = tempdir().unwrap();
        let reg = ModelRegistry::new(Some(dir.path().to_path_buf()), None, None);

        let file_path = dir.path().join("test.bin");
        fs::write(&file_path, b"hello world").unwrap();

        let result = reg.verify_sha256(
            &file_path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            AtheerCoreError::ChecksumMismatch { expected, actual } => {
                assert_eq!(
                    actual,
                    "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
                );
                assert_eq!(
                    expected,
                    "0000000000000000000000000000000000000000000000000000000000000000"
                );
            }
            e => panic!("wrong error type: {e}"),
        }
    }

    #[test]
    fn test_verify_sha256_match() {
        let dir = tempdir().unwrap();
        let reg = ModelRegistry::new(Some(dir.path().to_path_buf()), None, None);

        let file_path = dir.path().join("test.bin");
        fs::write(&file_path, b"hello world").unwrap();

        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(reg.verify_sha256(&file_path, expected).is_ok());
    }

    #[test]
    fn test_empty_cache_size() {
        let dir = tempdir().unwrap();
        let reg = ModelRegistry::new(Some(dir.path().to_path_buf()), None, None);
        assert_eq!(reg.cache_size().unwrap(), 0);
    }
}
