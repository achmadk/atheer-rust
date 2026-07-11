use std::sync::Arc;

/// Live state of an in-progress generation.
#[derive(Debug, Clone)]
pub struct GenerationState {
    /// Number of tokens generated so far (excluding the prompt).
    pub tokens_so_far: u32,
    /// Running average decode step duration (ms).
    pub avg_decode_ms: f64,
    /// Estimated P99 per-step latency (ms). `None` before 10 decode steps.
    pub p99_estimate_ms: Option<f64>,
    /// Duration of the initial prompt prefill (ms). `None` before prefill completes.
    pub prefill_ms: Option<f64>,
}

/// Callback invoked after each decode step during streaming generation.
///
/// The callback receives the newly decoded token id and the current
/// [`GenerationState`].  If the callback returns `false`, generation
/// is aborted immediately.
pub trait StreamingCallback: Send {
    fn on_token(&mut self, token: u32, state: &GenerationState) -> bool;
}

// -----------------------------------------------------------------------
// Helper: wrap a boxed closure as a StreamingCallback.
// -----------------------------------------------------------------------

struct ClosureCallback<F: FnMut(u32, &GenerationState) -> bool + Send>(F);

impl<F: FnMut(u32, &GenerationState) -> bool + Send> StreamingCallback for ClosureCallback<F> {
    fn on_token(&mut self, token: u32, state: &GenerationState) -> bool {
        (self.0)(token, state)
    }
}

/// Convenience: create a [`StreamingCallback`] from a closure.
pub fn callback_from_fn<F>(f: F) -> Box<dyn StreamingCallback>
where
    F: FnMut(u32, &GenerationState) -> bool + Send + 'static,
{
    Box::new(ClosureCallback(f))
}

/// A no-op streaming callback that always returns `true`.
pub struct NullCallback;

impl StreamingCallback for NullCallback {
    fn on_token(&mut self, _token: u32, _state: &GenerationState) -> bool {
        true
    }
}

// -----------------------------------------------------------------------
// Arc-wrapped callback for sharing across threads.
// -----------------------------------------------------------------------

/// Thread-safe streaming callback wrapper.
///
/// Useful when the callback needs to be shared between an inference
/// thread and a UI thread (e.g., send tokens over a channel).
pub struct SharedCallback {
    inner: Arc<tokio::sync::Mutex<Box<dyn StreamingCallback>>>,
}

impl SharedCallback {
    pub fn new(callback: Box<dyn StreamingCallback>) -> Self {
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(callback)),
        }
    }

    /// Invoke the wrapped callback. Returns `false` if the callback
    /// signals abort.
    pub async fn on_token(&self, token: u32, state: &GenerationState) -> bool {
        let mut guard = self.inner.lock().await;
        guard.on_token(token, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_state_defaults() {
        let state = GenerationState {
            tokens_so_far: 0,
            avg_decode_ms: 0.0,
            p99_estimate_ms: None,
            prefill_ms: None,
        };
        assert_eq!(state.tokens_so_far, 0);
        assert!(state.p99_estimate_ms.is_none());
    }

    #[test]
    fn test_callback_from_closure() {
        let tokens = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let tokens_clone = tokens.clone();
        let mut cb = callback_from_fn(move |token, _state| {
            tokens_clone.lock().unwrap().push(token);
            true
        });

        let state = GenerationState {
            tokens_so_far: 0,
            avg_decode_ms: 10.0,
            p99_estimate_ms: Some(15.0),
            prefill_ms: Some(200.0),
        };
        assert!(cb.on_token(42, &state));
        assert!(cb.on_token(43, &state));
        assert_eq!(*tokens.lock().unwrap(), vec![42, 43]);
    }

    #[test]
    fn test_callback_abort() {
        let mut cb = callback_from_fn(|_token, state| state.tokens_so_far < 3);

        let state = GenerationState {
            tokens_so_far: 0,
            avg_decode_ms: 0.0,
            p99_estimate_ms: None,
            prefill_ms: None,
        };
        assert!(cb.on_token(1, &state));

        let state2 = GenerationState {
            tokens_so_far: 3,
            avg_decode_ms: 0.0,
            p99_estimate_ms: None,
            prefill_ms: None,
        };
        assert!(!cb.on_token(4, &state2));
    }

    #[test]
    fn test_null_callback_never_aborts() {
        let mut cb = NullCallback;
        let state = GenerationState {
            tokens_so_far: 100,
            avg_decode_ms: 999.0,
            p99_estimate_ms: Some(9999.0),
            prefill_ms: Some(9999.0),
        };
        assert!(cb.on_token(0, &state));
        assert!(cb.on_token(1, &state));
    }
}
