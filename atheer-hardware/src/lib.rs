pub mod error;
pub mod health;
pub mod memory;
pub mod monitor;
pub mod power;
pub mod thermal;

pub use health::HealthStatus;
pub use memory::MemoryStatus;
pub use monitor::HardwareMonitor;
pub use power::PowerState;
pub use thermal::ThermalState;

#[cfg(target_os = "android")]
pub mod android;
#[cfg(target_os = "ios")]
pub mod ios;
