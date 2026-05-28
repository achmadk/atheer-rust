use crate::{AtheerCoreError, Result};

/// Wrapper around the Hugging Face `tokenizers` crate.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| AtheerCoreError::TokenizerLoadFailed(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .unwrap_or_else(|_| self.inner.encode("", false).unwrap());
        encoding.get_ids().to_vec()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> String {
        self.inner
            .decode(tokens, skip_special_tokens)
            .unwrap_or_default()
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Look up a token string and return its integer id, if defined in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn clone_inner(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
