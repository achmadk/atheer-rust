use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[allow(dead_code)]
pub struct CrashReporter {
    crash_count: AtomicU64,
    log_path: Option<PathBuf>,
    max_crashes: usize,
}

impl CrashReporter {
    pub fn new() -> Self {
        Self {
            crash_count: AtomicU64::new(0),
            log_path: None,
            max_crashes: 10,
        }
    }

    pub fn with_log_path(mut self, path: PathBuf) -> Self {
        self.log_path = Some(path);
        self
    }

    pub fn crash_log_path(&self) -> Option<&PathBuf> {
        self.log_path.as_ref()
    }

    pub fn record_crash(&self, error: &str, context: &str) -> u64 {
        let crash_id = self.crash_count.fetch_add(1, Ordering::SeqCst) + 1;

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
}
