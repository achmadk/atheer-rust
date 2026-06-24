use atheer_core::InferenceEngine;
use std::sync::{Arc, Mutex};

/// Error states during an agent execution session.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Maximum iterations ({0}) exceeded without a final answer")]
    MaxIterationsExceeded(usize),
    #[error("Generation error: {0}")]
    GenerationError(String),
}

/// A simple autonomous agent that loops over a prompt, detecting tool calls
/// and requesting their execution from the host application.
///
/// In a real scenario, this would use `atheer_memory_bank` to handle context window limits
/// if the multi-turn session exceeds `max_seq_len`.
pub struct Agent {
    engine: Arc<Mutex<Option<InferenceEngine>>>,
    max_steps: usize,
}

impl Agent {
    pub fn new(engine: Arc<Mutex<Option<InferenceEngine>>>) -> Self {
        Self {
            engine,
            max_steps: 5, // Default cutoff
        }
    }

    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// Run a single turn of generation.
    /// In a full implementation, this parses the raw text for `<tool_call>` syntax
    /// and extracts it, or returns the final answer.
    pub fn step(&self, prompt: &str, max_tokens: u32) -> Result<String, AgentError> {
        let mut guard = self
            .engine
            .lock()
            .map_err(|_| AgentError::GenerationError("Engine lock failed".to_string()))?;

        let engine = guard
            .as_mut()
            .ok_or_else(|| AgentError::GenerationError("Engine not initialized".to_string()))?;

        let (text, _, _) = engine
            .generate(prompt, max_tokens, None)
            .map_err(|e| AgentError::GenerationError(e.to_string()))?;

        Ok(text)
    }
}
