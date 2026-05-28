use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

pub struct KvSyncWorker {
    running: Arc<AtomicBool>,
    sync_interval: Arc<AtomicUsize>,
    tokens_since_sync: Arc<AtomicUsize>,
}

impl KvSyncWorker {
    pub fn new(sync_interval_tokens: usize) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            sync_interval: Arc::new(AtomicUsize::new(sync_interval_tokens)),
            tokens_since_sync: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn record_tokens(&self, count: usize) -> bool {
        self.tokens_since_sync.fetch_add(count, Ordering::Relaxed);
        self.tokens_since_sync.load(Ordering::Relaxed) >= self.sync_interval.load(Ordering::Relaxed)
    }

    pub fn reset_counter(&self) {
        self.tokens_since_sync.store(0, Ordering::SeqCst);
    }

    pub fn set_interval(&self, tokens: usize) {
        self.sync_interval.store(tokens, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_worker_lifecycle() {
        let worker = KvSyncWorker::new(10);
        assert!(!worker.is_running());

        worker.start();
        assert!(worker.is_running());

        worker.stop();
        assert!(!worker.is_running());
    }

    #[test]
    fn test_sync_interval() {
        let worker = KvSyncWorker::new(5);

        assert!(!worker.record_tokens(3));
        assert!(worker.record_tokens(3));

        worker.reset_counter();
        assert!(!worker.record_tokens(4));
    }
}
