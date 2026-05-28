pub mod json;
pub mod sampler;

pub use json::JsonGrammar;
pub use sampler::GrammarSampler;

/// Trait for grammar constraints that can validate token sequences.
pub trait GrammarConstraint: Send + Sync {
    /// Check whether appending `text` to the current partial output
    /// keeps the output as a valid prefix of the grammar's language.
    fn is_valid_prefix(&self, text: &str) -> bool;

    /// Advance the internal state by `text`. Only call after `is_valid_prefix` returns true.
    fn advance(&mut self, text: &str);

    /// Reset the grammar to its initial state.
    fn reset(&mut self);

    /// Clone the current state into a boxed trait object.
    fn clone_box(&self) -> Box<dyn GrammarConstraint>;
}

impl Clone for Box<dyn GrammarConstraint> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
