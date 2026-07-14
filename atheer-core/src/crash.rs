use crate::privacy::PrivacyMode;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Encodes `Option<PrivacyMode>` as `u64` for atomic storage:
/// `None → u64::MAX`, `Normal → 0`, `Ephemeral → 1`, `Audited → 2`.
fn mode_to_u64(mode: Option<PrivacyMode>) -> u64 {
    match mode {
        None => u64::MAX,
        Some(PrivacyMode::Normal) => 0,
        Some(PrivacyMode::Ephemeral) => 1,
        Some(PrivacyMode::Audited) => 2,
    }
}

fn u64_to_mode(val: u64) -> Option<PrivacyMode> {
    match val {
        u64::MAX => None,
        0 => Some(PrivacyMode::Normal),
        1 => Some(PrivacyMode::Ephemeral),
        2 => Some(PrivacyMode::Audited),
        _ => None,
    }
}

#[allow(dead_code)]
pub struct CrashReporter {
    crash_count: AtomicU64,
    log_path: Option<PathBuf>,
    max_crashes: usize,
    mode: AtomicU64,
}

impl CrashReporter {
    pub fn new() -> Self {
        Self {
            crash_count: AtomicU64::new(0),
            log_path: None,
            max_crashes: 10,
            mode: AtomicU64::new(u64::MAX),
        }
    }

    pub fn with_log_path(mut self, path: PathBuf) -> Self {
        self.log_path = Some(path);
        self
    }

    pub fn with_privacy_mode(self, mode: PrivacyMode) -> Self {
        self.set_privacy_mode(Some(mode));
        self
    }

    pub fn set_privacy_mode(&self, mode: Option<PrivacyMode>) {
        self.mode.store(mode_to_u64(mode), Ordering::Release);
    }

    pub fn crash_log_path(&self) -> Option<&PathBuf> {
        self.log_path.as_ref()
    }

    pub fn record_crash(&self, error: &str, context: &str) -> u64 {
        let crash_id = self.crash_count.fetch_add(1, Ordering::SeqCst) + 1;

        // In Ephemeral mode, skip the file write entirely
        if u64_to_mode(self.mode.load(Ordering::Acquire)) == Some(PrivacyMode::Ephemeral) {
            return crash_id;
        }

        if let Some(ref path) = self.log_path {
            let entry = CrashEntry {
                id: crash_id,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                error: error.to_string(),
                context: context.to_string(),
            };
            let _ = Self::write_crash_log(path, &entry);
        }

        crash_id
    }

    pub fn crash_count(&self) -> u64 {
        self.crash_count.load(Ordering::SeqCst)
    }

    /// Record a crash with sensitive key identifiers redacted.
    ///
    /// Any occurrence of `key_id_to_redact` in `context` is replaced with
    /// `"KEY_REDACTED"` before the entry is written to the crash log.
    /// In Ephemeral mode, the file write is skipped (same as record_crash).
    pub fn record_crash_scrubbed(&self, error: &str, context: &str, key_id_to_redact: &str) -> u64 {
        let scrubbed = if key_id_to_redact.is_empty() {
            context.to_string()
        } else {
            context.replace(key_id_to_redact, "KEY_REDACTED")
        };
        self.record_crash(error, &scrubbed)
    }

    pub fn reset_crashes(&self) {
        self.crash_count.store(0, Ordering::SeqCst);
    }

    fn write_crash_log(path: &PathBuf, entry: &CrashEntry) -> std::io::Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        writeln!(
            file,
            "[{}] crash_id={} context={} error={}",
            entry.timestamp, entry.id, entry.context, entry.error
        )?;

        Ok(())
    }
}

impl Default for CrashReporter {
    fn default() -> Self {
        Self::new()
    }
}

struct CrashEntry {
    id: u64,
    timestamp: u64,
    error: String,
    context: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_crash_reporter_creation() {
        let reporter = CrashReporter::new();
        assert_eq!(reporter.crash_count(), 0);
    }

    #[test]
    fn test_crash_recording() {
        let reporter = CrashReporter::new();
        let id = reporter.record_crash("OutOfMemory", "model loading");
        assert_eq!(id, 1);
        assert_eq!(reporter.crash_count(), 1);
    }

    #[test]
    fn test_crash_reset() {
        let reporter = CrashReporter::new();
        reporter.record_crash("Error", "test");
        assert_eq!(reporter.crash_count(), 1);

        reporter.reset_crashes();
        assert_eq!(reporter.crash_count(), 0);
    }

    #[test]
    fn test_ephemeral_skips_write_but_increments_count() {
        let dir = std::env::temp_dir().join("aether_test_ephemeral");
        let log_path = dir.join("crash.log");
        fs::create_dir_all(&dir).ok();

        let reporter = CrashReporter::new()
            .with_log_path(log_path.clone())
            .with_privacy_mode(PrivacyMode::Ephemeral);

        let id = reporter.record_crash("TestError", "ephemeral test");
        assert_eq!(id, 1);
        assert_eq!(reporter.crash_count(), 1);

        // Log file should not exist (nothing was written)
        assert!(!log_path.exists(), "log file should not exist in Ephemeral mode");

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_normal_writes_crash() {
        let dir = std::env::temp_dir().join("aether_test_normal_write");
        let log_path = dir.join("crash.log");
        fs::create_dir_all(&dir).ok();

        let reporter = CrashReporter::new()
            .with_log_path(log_path.clone())
            .with_privacy_mode(PrivacyMode::Normal);

        let id = reporter.record_crash("TestError", "normal test");
        assert_eq!(id, 1);
        assert_eq!(reporter.crash_count(), 1);
        assert!(log_path.exists(), "log file should exist in Normal mode");

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_audited_writes_crash() {
        let dir = std::env::temp_dir().join("aether_test_audited_write");
        let log_path = dir.join("crash.log");
        fs::create_dir_all(&dir).ok();

        let reporter = CrashReporter::new()
            .with_log_path(log_path.clone())
            .with_privacy_mode(PrivacyMode::Audited);

        let id = reporter.record_crash("TestError", "audited test");
        assert_eq!(id, 1);
        assert_eq!(reporter.crash_count(), 1);
        assert!(log_path.exists(), "log file should exist in Audited mode");

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_with_privacy_mode_builder() {
        let reporter = CrashReporter::new().with_privacy_mode(PrivacyMode::Ephemeral);
        let id = reporter.record_crash("Test", "builder test");
        assert_eq!(id, 1);
        // No crash log path set, but mode shouldn't cause issues
    }

    #[test]
    fn test_ephemeral_skips_scrubbed_write() {
        let dir = std::env::temp_dir().join("aether_test_ephemeral_scrub");
        let log_path = dir.join("crash.log");
        fs::create_dir_all(&dir).ok();

        let reporter = CrashReporter::new()
            .with_log_path(log_path.clone())
            .with_privacy_mode(PrivacyMode::Ephemeral);

        let id = reporter.record_crash_scrubbed("TestError", "sensitive data", "sensitive");
        assert_eq!(id, 1);
        assert!(!log_path.exists(), "log file should not exist in Ephemeral mode (scrubbed)");

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_set_privacy_mode() {
        let reporter = CrashReporter::new();
        reporter.set_privacy_mode(Some(PrivacyMode::Ephemeral));
        let id = reporter.record_crash("Test", "set mode test");
        assert_eq!(id, 1);
        // Counter incremented but no log path set — nothing to assert beyond no panic
    }

    #[test]
    fn test_privacy_mode_default_is_none() {
        let reporter = CrashReporter::new();
        // In Normal-equivalent mode (no mode set), crash should still record
        let id = reporter.record_crash("Test", "no mode test");
        assert_eq!(id, 1);
    }
}
