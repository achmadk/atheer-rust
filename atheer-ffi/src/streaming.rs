pub trait StreamingCallback: Send + Sync {
    fn on_token(&self, token: String);
    fn on_mode_change(&self, mode: String);
    fn on_error(&self, error: String);
}

#[derive(Debug, Clone)]
pub struct StreamingResult {
    pub tokens: Vec<String>,
    pub complete: bool,
    pub error: Option<String>,
}

impl StreamingResult {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            complete: false,
            error: None,
        }
    }

    pub fn with_error(error: String) -> Self {
        Self {
            tokens: Vec::new(),
            complete: false,
            error: Some(error),
        }
    }

    pub fn finish(mut self) -> Self {
        self.complete = true;
        self
    }

    pub fn add_token(&mut self, token: String) {
        self.tokens.push(token);
    }
}

impl Default for StreamingResult {
    fn default() -> Self {
        Self::new()
    }
}
