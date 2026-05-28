use uuid::Uuid;

pub struct Session {
    id: String,
    model_id: String,
    created_at: std::time::Instant,
    token_count: usize,
}

impl Session {
    pub fn new(model_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            model_id,
            created_at: std::time::Instant::now(),
            token_count: 0,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn add_tokens(&mut self, count: usize) {
        self.token_count += count;
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = Session::new("test-model".to_string());
        assert!(!session.id().is_empty());
        assert_eq!(session.model_id(), "test-model");
        assert_eq!(session.token_count(), 0);
    }

    #[test]
    fn test_session_token_counting() {
        let mut session = Session::new("test-model".to_string());
        session.add_tokens(100);
        assert_eq!(session.token_count(), 100);
        session.add_tokens(50);
        assert_eq!(session.token_count(), 150);
    }
}
