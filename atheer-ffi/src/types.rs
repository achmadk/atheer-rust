use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// A JSON Schema string defining the parameters
    pub parameters_schema: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct ToolCall {
    pub name: String,
    /// A JSON string containing the arguments
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub json_schema: Option<String>,
    pub tools: Vec<ToolDefinition>,
}

impl GenerationRequest {
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            max_tokens: 512,
            temperature: 0.7,
            json_schema: None,
            tools: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct GenerationResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub inference_time_ms: u64,
    pub mode: String,
    pub tool_calls: Vec<ToolCall>,
}

impl GenerationResponse {
    pub fn new(text: String, tokens: u32, time_ms: u64, mode: &str) -> Self {
        Self {
            text,
            tokens_generated: tokens,
            inference_time_ms: time_ms,
            mode: mode.to_string(),
            tool_calls: Vec::new(),
        }
    }

    pub fn tokens_per_second(&self) -> f32 {
        if self.inference_time_ms == 0 {
            return 0.0;
        }
        (self.tokens_generated as f32 / self.inference_time_ms as f32) * 1000.0
    }
}
