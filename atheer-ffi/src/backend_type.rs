use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, uniffi::Enum)]
pub enum AtheerBackendType {
    Cpu,
    Metal,
    Vulkan,
    NNAPI,
    CoreML,
}

impl From<atheer_accel::BackendType> for AtheerBackendType {
    fn from(bt: atheer_accel::BackendType) -> Self {
        match bt {
            atheer_accel::BackendType::Cpu => AtheerBackendType::Cpu,
            atheer_accel::BackendType::Metal => AtheerBackendType::Metal,
            atheer_accel::BackendType::Vulkan => AtheerBackendType::Vulkan,
            atheer_accel::BackendType::NNAPI => AtheerBackendType::NNAPI,
            atheer_accel::BackendType::CoreML => AtheerBackendType::CoreML,
            atheer_accel::BackendType::Cuda => AtheerBackendType::Cpu,
        }
    }
}

impl From<AtheerBackendType> for atheer_accel::BackendType {
    fn from(bt: AtheerBackendType) -> Self {
        match bt {
            AtheerBackendType::Cpu => atheer_accel::BackendType::Cpu,
            AtheerBackendType::Metal => atheer_accel::BackendType::Metal,
            AtheerBackendType::Vulkan => atheer_accel::BackendType::Vulkan,
            AtheerBackendType::NNAPI => atheer_accel::BackendType::NNAPI,
            AtheerBackendType::CoreML => atheer_accel::BackendType::CoreML,
        }
    }
}

impl AtheerBackendType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AtheerBackendType::Cpu => "cpu",
            AtheerBackendType::Metal => "metal",
            AtheerBackendType::Vulkan => "vulkan",
            AtheerBackendType::NNAPI => "nnapi",
            AtheerBackendType::CoreML => "coreml",
        }
    }
}
