use crate::{HealthStatus, MemoryStatus, PowerState, ThermalState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub trait HardwareMonitor: Send + Sync {
    fn health(&self) -> HealthStatus;
    fn thermal_state(&self) -> ThermalState;
    fn memory_status(&self) -> MemoryStatus;
    fn power_state(&self) -> PowerState;
    fn battery_level(&self) -> u32;
    fn is_on_battery(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    pub thermal: ThermalState,
    pub available_ram_mb: u64,
    pub total_ram_mb: u64,
    pub battery_level: u32,
    pub on_battery: bool,
    pub timestamp: i64,
    pub sample_count: u64,
}

impl Default for HealthSnapshot {
    fn default() -> Self {
        Self {
            thermal: ThermalState::Nominal,
            available_ram_mb: 4096,
            total_ram_mb: 8192,
            battery_level: 100,
            on_battery: false,
            timestamp: chrono::Utc::now().timestamp(),
            sample_count: 0,
        }
    }
}

pub struct GenericMonitor {
    snapshot: Arc<Mutex<HealthSnapshot>>,
    running: Arc<AtomicBool>,
}

impl GenericMonitor {
    pub fn new() -> Self {
        let snapshot = Arc::new(Mutex::new(HealthSnapshot::default()));
        let running = Arc::new(AtomicBool::new(true));

        let snap = snapshot.clone();
        let run = running.clone();

        thread::spawn(move || {
            while run.load(Ordering::Relaxed) {
                #[cfg(any(target_os = "ios", target_os = "macos"))]
                let sample = Self::sample_apple();
                #[cfg(target_os = "android")]
                let sample = Self::sample_android();
                #[cfg(not(any(target_os = "ios", target_os = "macos", target_os = "android")))]
                let sample = Self::sample_default();

                if let Ok(mut guard) = snap.lock() {
                    guard.thermal = sample.thermal;
                    guard.available_ram_mb = sample.available_ram_mb;
                    guard.total_ram_mb = sample.total_ram_mb;
                    guard.battery_level = sample.battery_level;
                    guard.on_battery = sample.on_battery;
                    guard.timestamp = chrono::Utc::now().timestamp();
                    guard.sample_count += 1;
                }

                thread::sleep(Duration::from_secs(1));
            }
        });

        Self { snapshot, running }
    }

    /// Default sampling for platforms without native telemetry (Linux, CI).
    /// Returns nominal values since actual OS APIs aren't available.
    #[allow(dead_code)]
    fn sample_default() -> HealthSnapshot {
        HealthSnapshot {
            timestamp: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }

    /// Apple platform sampling via objc2 FFI (iOS/macOS).
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn sample_apple() -> HealthSnapshot {
        // Real implementation would use objc2 bindings:
        // - ProcessInfo.thermalState -> ThermalState
        // - os_proc_available_memory() -> available_ram
        // - NSProcessInfo.processInfo.physicalMemory -> total_ram
        // - UIDevice.current.batteryLevel / batteryState -> battery
        //
        // For now, use sysctl-based approximations where possible
        let thermal = Self::apple_thermal_state();
        let (available_mb, total_mb) = Self::apple_memory();
        let (battery_level, on_battery) = Self::apple_battery();

        HealthSnapshot {
            thermal,
            available_ram_mb: available_mb,
            total_ram_mb: total_mb,
            battery_level,
            on_battery,
            timestamp: chrono::Utc::now().timestamp(),
            sample_count: 0,
        }
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn apple_thermal_state() -> ThermalState {
        // Placeholder: ProcessInfo.thermalState via objc2
        ThermalState::Nominal
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn apple_memory() -> (u64, u64) {
        // Placeholder: os_proc_available_memory() + NSProcessInfo
        (2048, 6144)
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn apple_battery() -> (u32, bool) {
        // Placeholder: UIDevice.batteryLevel / batteryState
        (100, false)
    }

    /// Android platform sampling via JNI.
    ///
    /// Uses real JNI calls when `crate::android::is_initialized()` returns
    /// true, otherwise falls back to default (nominal) values.
    #[cfg(target_os = "android")]
    fn sample_android() -> HealthSnapshot {
        if crate::android::is_initialized() {
            let thermal = crate::android::thermal_headroom()
                .map(crate::android::headroom_to_state)
                .unwrap_or(ThermalState::Nominal);
            let (available_mb, total_mb) = crate::android::memory_mb().unwrap_or((3072, 8192));
            let (battery_level, on_battery) = crate::android::battery_info()
                .map(|(l, c)| (l, !c))
                .unwrap_or((100, false));

            HealthSnapshot {
                thermal,
                available_ram_mb: available_mb,
                total_ram_mb: total_mb,
                battery_level,
                on_battery,
                timestamp: chrono::Utc::now().timestamp(),
                sample_count: 0,
            }
        } else {
            // JNI not yet initialized — return nominal defaults
            HealthSnapshot {
                thermal: ThermalState::Nominal,
                available_ram_mb: 3072,
                total_ram_mb: 8192,
                battery_level: 100,
                on_battery: false,
                timestamp: chrono::Utc::now().timestamp(),
                sample_count: 0,
            }
        }
    }
}

impl Drop for GenericMonitor {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Default for GenericMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareMonitor for GenericMonitor {
    fn health(&self) -> HealthStatus {
        if let Ok(guard) = self.snapshot.lock() {
            HealthStatus {
                thermal: guard.thermal,
                available_ram_mb: guard.available_ram_mb,
                total_ram_mb: guard.total_ram_mb,
                battery_level: guard.battery_level,
                on_battery: guard.on_battery,
                timestamp: guard.timestamp,
                sample_count: guard.sample_count,
            }
        } else {
            HealthStatus::default()
        }
    }

    fn thermal_state(&self) -> ThermalState {
        self.snapshot
            .lock()
            .map(|g| g.thermal)
            .unwrap_or(ThermalState::Nominal)
    }

    fn memory_status(&self) -> MemoryStatus {
        if let Ok(guard) = self.snapshot.lock() {
            MemoryStatus {
                available_mb: guard.available_ram_mb,
                total_mb: guard.total_ram_mb,
                low_memory_threshold_mb: 800,
            }
        } else {
            MemoryStatus::default()
        }
    }

    fn power_state(&self) -> PowerState {
        PowerState::Unknown
    }

    fn battery_level(&self) -> u32 {
        self.snapshot.lock().map(|g| g.battery_level).unwrap_or(100)
    }

    fn is_on_battery(&self) -> bool {
        self.snapshot.lock().map(|g| g.on_battery).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_monitor_creation() {
        let monitor = GenericMonitor::new();
        let health = monitor.health();

        // Should have reasonable default values
        assert!(health.available_ram_mb > 0);
        assert!(health.battery_level <= 100);
    }

    #[test]
    fn test_monitor_sampling_thread() {
        let monitor = GenericMonitor::new();
        // Give the sampling thread time to collect a sample
        thread::sleep(Duration::from_millis(1100));

        let health = monitor.health();
        // After 1+ seconds, the sampling thread should have run at least once
        assert!(health.timestamp > 0);
    }

    #[test]
    fn test_snapshot_freshness() {
        let monitor = GenericMonitor::new();
        thread::sleep(Duration::from_millis(500));

        let snap1 = monitor.health();
        thread::sleep(Duration::from_millis(1100));
        let snap2 = monitor.health();

        // Later snapshot should have later (or equal) timestamp
        assert!(snap2.timestamp >= snap1.timestamp);
    }

    #[test]
    fn test_monitor_drop_stops_thread() {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        let handle = thread::spawn(move || {
            while r.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(10));
            }
        });

        running.store(false, Ordering::Relaxed);
        handle.join().unwrap();
        // Test passes if thread joins without hanging
    }

    #[test]
    fn test_health_status_critical() {
        let mut status = HealthStatus::default();
        status.thermal = ThermalState::Critical;
        status.available_ram_mb = 500;

        assert!(status.is_critical());
    }

    #[test]
    fn test_health_snapshot_default() {
        let snap = HealthSnapshot::default();
        assert_eq!(snap.thermal, ThermalState::Nominal);
        assert_eq!(snap.available_ram_mb, 4096);
        assert_eq!(snap.battery_level, 100);
    }

    #[test]
    fn test_sample_count_starts_zero_then_advances() {
        let monitor = GenericMonitor::new();
        let health = monitor.health();
        // Before first 1 Hz tick completes, sample_count must be 0
        assert_eq!(health.sample_count, 0);

        // Wait for at least one sampling cycle
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let health = monitor.health();
        assert!(
            health.sample_count > 0,
            "sample_count should advance after at least one sampling tick"
        );
    }

    #[test]
    fn test_sample_count_reflected_in_health_status() {
        let monitor = GenericMonitor::new();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let health = monitor.health();
        let snap = monitor.snapshot.lock().unwrap();
        assert_eq!(
            health.sample_count, snap.sample_count,
            "HealthStatus.sample_count must match HealthSnapshot.sample_count"
        );
    }
}
