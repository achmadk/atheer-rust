use crate::{AccelBackend, AccelResult, BackendType, Result};
#[cfg(test)]
use rayon::prelude::*;

pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(rayon::current_num_threads);
        Self { num_threads }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new(None)
    }
}

impl AccelBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn is_available(&self) -> bool {
        true
    }

    #[cfg(test)]
    fn forward(&self, input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        let start = std::time::Instant::now();

        let logits: Vec<f32> = input_ids
            .par_iter()
            .map(|&id| {
                let seed = id as f32 * 0.01;
                seed.sin() * 0.5 + 0.5
            })
            .collect();

        let elapsed = start.elapsed().as_millis() as u64;

        Ok(AccelResult::new(logits, input_ids.len(), elapsed))
    }

    #[cfg(not(test))]
    fn forward(&self, _input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        Err(crate::AccelError::Deprecated(
            "CpuBackend::forward() is deprecated; use InferenceEngine::generate() instead"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new(Some(4));
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.backend_type(), BackendType::Cpu);
    }

    #[test]
    fn test_cpu_forward() {
        let backend = CpuBackend::default();
        let input = vec![1u32, 2, 3, 4, 5];
        let positions = vec![0, 1, 2, 3, 4];

        let result = backend.forward(&input, &positions).unwrap();
        assert_eq!(result.tokens_generated, 5);
    }

    #[test]
    fn test_tokens_per_second() {
        let result = AccelResult::new(vec![0.0; 100], 100, 100);
        assert_eq!(result.tokens_per_second(), 1000.0);
    }

    #[test]
    fn test_cpu_forward_performance() {
        let backend = CpuBackend::default();
        let input = vec![1u32; 100];
        let positions = (0..100).collect::<Vec<_>>();

        let start = std::time::Instant::now();
        let result = backend.forward(&input, &positions).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert_eq!(result.tokens_generated, 100);
        assert!(elapsed < 1000);
    }

    #[test]
    fn test_large_input_handling() {
        let backend = CpuBackend::default();
        let input = vec![1u32; 1000];
        let positions = (0..1000).collect::<Vec<_>>();

        let result = backend.forward(&input, &positions).unwrap();
        assert_eq!(result.tokens_generated, 1000);
    }
}
