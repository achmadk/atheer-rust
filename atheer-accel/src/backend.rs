use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    NNAPI,
    CoreML,
}

impl BackendType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendType::Cpu => "cpu",
            BackendType::Metal => "metal",
            BackendType::Cuda => "cuda",
            BackendType::Vulkan => "vulkan",
            BackendType::NNAPI => "nnapi",
            BackendType::CoreML => "coreml",
        }
    }
}

pub struct Backend {
    backend_type: BackendType,
    name: String,
}

impl Backend {
    pub fn new(backend_type: BackendType) -> Self {
        Self {
            backend_type,
            name: backend_type.as_str().to_string(),
        }
    }

    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::new(BackendType::Cpu)
    }
}
