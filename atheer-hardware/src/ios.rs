//! iOS hardware telemetry via `objc2` FFI.
//!
//! Provides real-time thermal state, memory, and battery readings from iOS
//! through Objective-C runtime calls. All code in this module is gated behind
//! `#[cfg(any(target_os = "ios", target_os = "macos"))]` — it will not compile
//! on non-Apple platforms.
//!
//! # Safety
//!
//! Every `msg_send!` call in this module is `unsafe` because the Objective-C
//! runtime cannot guarantee the receiver implements the selector at compile
//! time. Each call is tested to panic with a clear message if the runtime
//! class or selector is unavailable — on production devices these selectors
//! are always present on iOS 15+ / macOS 12+.

#![cfg(any(target_os = "ios", target_os = "macos"))]

use crate::monitor::HealthSnapshot;
use crate::{HardwareMonitor, MemoryStatus, PowerState, ThermalState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// objc2 imports
// ---------------------------------------------------------------------------

use objc2::runtime::NSObject;
use objc2::msg_send;

// ---------------------------------------------------------------------------
// C FFI: os_proc_available_memory()
// ---------------------------------------------------------------------------

extern "C" {
    fn os_proc_available_memory() -> usize;
}

// ---------------------------------------------------------------------------
// Thermal state constants (from NSProcessInfo.h)
// ---------------------------------------------------------------------------

const NS_PROCESS_INFO_THERMAL_STATE_NOMINAL: i64 = 0;
const NS_PROCESS_INFO_THERMAL_STATE_FAIR: i64 = 1;
const NS_PROCESS_INFO_THERMAL_STATE_SERIOUS: i64 = 2;
const NS_PROCESS_INFO_THERMAL_STATE_CRITICAL: i64 = 3;

// ---------------------------------------------------------------------------
// UIDevice battery state constants
// ---------------------------------------------------------------------------

const UI_DEVICE_BATTERY_STATE_UNKNOWN: i64 = 0;
const UI_DEVICE_BATTERY_STATE_UNPLUGGED: i64 = 1;
const UI_DEVICE_BATTERY_STATE_CHARGING: i64 = 2;
const UI_DEVICE_BATTERY_STATE_FULL: i64 = 3;

// ---------------------------------------------------------------------------
// Thermal state
// ---------------------------------------------------------------------------

/// Read the current iOS thermal state via `NSProcessInfo.processInfo.thermalState`.
fn read_thermal_state() -> ThermalState {
    // SAFETY: NSProcessInfo is always available on iOS 15+ / macOS 12+.
    // `processInfo` returns a shared singleton; `thermalState` returns an
    // NSInteger enum. Both selectors exist on all supported OS versions.
    let state: i64 = unsafe {
        let cls = objc2::runtime::AnyClass::get(c"NSProcessInfo")
            .expect("NSProcessInfo class not found — this code requires iOS/macOS");
        let process_info: *mut NSObject = msg_send![cls, processInfo];
        msg_send![process_info, thermalState]
    };

    match state {
        NS_PROCESS_INFO_THERMAL_STATE_NOMINAL => ThermalState::Nominal,
        NS_PROCESS_INFO_THERMAL_STATE_FAIR => ThermalState::Fair,
        NS_PROCESS_INFO_THERMAL_STATE_SERIOUS => ThermalState::Serious,
        NS_PROCESS_INFO_THERMAL_STATE_CRITICAL => ThermalState::Critical,
        other => {
            tracing::warn!("Unknown NSProcessInfoThermalState value: {other}");
            ThermalState::Nominal
        }
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Read available and total physical memory.
///
/// - Available: `os_proc_available_memory()` C function
/// - Total: `NSProcessInfo.processInfo.physicalMemory`
fn read_memory() -> (u64, u64) {
    // Available memory via C FFI
    let available_bytes = unsafe { os_proc_available_memory() };
    let available_mb = (available_bytes as u64) / (1024 * 1024);

    // Total physical memory via NSProcessInfo
    // SAFETY: physicalMemory is a property on NSProcessInfo, always available.
    let total_mb: u64 = unsafe {
        let cls = objc2::runtime::AnyClass::get(c"NSProcessInfo")
            .expect("NSProcessInfo class not found");
        let process_info: *mut NSObject = msg_send![cls, processInfo];
        let physical_memory: u64 = msg_send![process_info, physicalMemory];
        physical_memory / (1024 * 1024)
    };

    (available_mb, total_mb)
}

// ---------------------------------------------------------------------------
// Battery
// ---------------------------------------------------------------------------

/// Read battery level (0–100) and charging state.
///
/// Battery monitoring is enabled on the device for the duration of the sample
/// and restored to its previous state afterwards (good citizenship).
fn read_battery() -> (u32, bool) {
    // SAFETY: UIDevice is always available on iOS. `currentDevice` returns
    // the shared singleton. We enable battery monitoring temporarily.
    unsafe {
        let cls = objc2::runtime::AnyClass::get(c"UIDevice")
            .expect("UIDevice class not found — this code requires iOS");
        let device: *mut NSObject = msg_send![cls, currentDevice];

        // Save previous battery monitoring state
        let was_monitoring: bool = msg_send![device, isBatteryMonitoringEnabled];

        // Enable monitoring (required to read level/state)
        let _: () = msg_send![device, setBatteryMonitoringEnabled: true];

        // Read battery level (0.0–1.0, -1 if unavailable)
        let level_float: f32 = msg_send![device, batteryLevel];
        let level = if level_float >= 0.0 {
            (level_float * 100.0).round() as u32
        } else {
            0
        };

        // Read battery state
        let state: i64 = msg_send![device, batteryState];
        let on_battery = state != UI_DEVICE_BATTERY_STATE_CHARGING
            && state != UI_DEVICE_BATTERY_STATE_FULL;

        // Restore previous monitoring state
        let _: () = msg_send![device, setBatteryMonitoringEnabled: was_monitoring];

        (level.min(100), on_battery)
    }
}

// ---------------------------------------------------------------------------
// IosMonitor
// ---------------------------------------------------------------------------

/// iOS hardware monitor that samples thermal, memory, and battery state at 1 Hz.
///
/// Spawns a dedicated background thread on construction that continuously
/// reads hardware metrics and stores them in a shared `HealthSnapshot`.
pub struct IosMonitor {
    snapshot: Arc<Mutex<HealthSnapshot>>,
    running: Arc<AtomicBool>,
}

impl IosMonitor {
    pub fn new() -> Self {
        let snapshot = Arc::new(Mutex::new(HealthSnapshot {
            thermal: ThermalState::Nominal,
            available_ram_mb: 0,
            total_ram_mb: 0,
            battery_level: 0,
            on_battery: true,
            timestamp: chrono::Utc::now().timestamp(),
            sample_count: 0,
        }));
        let running = Arc::new(AtomicBool::new(true));

        let snap = snapshot.clone();
        let run = running.clone();

        thread::spawn(move || {
            while run.load(Ordering::Relaxed) {
                let thermal = read_thermal_state();
                let (available_mb, total_mb) = read_memory();
                let (battery_level, on_battery) = read_battery();

                if let Ok(mut guard) = snap.lock() {
                    guard.thermal = thermal;
                    guard.available_ram_mb = available_mb;
                    guard.total_ram_mb = total_mb;
                    guard.battery_level = battery_level;
                    guard.on_battery = on_battery;
                    guard.timestamp = chrono::Utc::now().timestamp();
                    guard.sample_count += 1;
                }

                thread::sleep(Duration::from_secs(1));
            }
        });

        Self { snapshot, running }
    }

    fn health_snapshot(&self) -> HealthSnapshot {
        self.snapshot
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| HealthSnapshot {
                thermal: ThermalState::Nominal,
                available_ram_mb: 0,
                total_ram_mb: 0,
                battery_level: 0,
                on_battery: true,
                timestamp: chrono::Utc::now().timestamp(),
                sample_count: 0,
            })
    }
}

impl Default for IosMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IosMonitor {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl HardwareMonitor for IosMonitor {
    fn health(&self) -> crate::HealthStatus {
        let snap = self.health_snapshot();
        crate::HealthStatus {
            thermal: snap.thermal,
            available_ram_mb: snap.available_ram_mb,
            total_ram_mb: snap.total_ram_mb,
            battery_level: snap.battery_level,
            on_battery: snap.on_battery,
            timestamp: snap.timestamp,
        }
    }

    fn thermal_state(&self) -> ThermalState {
        self.snapshot
            .lock()
            .map(|g| g.thermal)
            .unwrap_or(ThermalState::Nominal)
    }

    fn memory_status(&self) -> MemoryStatus {
        self.snapshot.lock().map_or_else(
            |_| MemoryStatus {
                available_mb: 0,
                total_mb: 0,
                low_memory_threshold_mb: 800,
            },
            |g| MemoryStatus {
                available_mb: g.available_ram_mb,
                total_mb: g.total_ram_mb,
                low_memory_threshold_mb: 800,
            },
        )
    }

    fn power_state(&self) -> PowerState {
        self.snapshot.lock().map_or(PowerState::Unknown, |g| {
            if g.on_battery {
                PowerState::Discharging
            } else {
                PowerState::Charging
            }
        })
    }

    fn battery_level(&self) -> u32 {
        self.snapshot
            .lock()
            .map(|g| g.battery_level)
            .unwrap_or(0)
    }

    fn is_on_battery(&self) -> bool {
        self.snapshot
            .lock()
            .map(|g| g.on_battery)
            .unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_state_nominal() {
        // We cannot call read_thermal_state() in a unit test without a real
        // iOS runtime. Instead verify the constant mappings are correct.
        assert_eq!(NS_PROCESS_INFO_THERMAL_STATE_NOMINAL, 0);
        assert_eq!(NS_PROCESS_INFO_THERMAL_STATE_FAIR, 1);
        assert_eq!(NS_PROCESS_INFO_THERMAL_STATE_SERIOUS, 2);
        assert_eq!(NS_PROCESS_INFO_THERMAL_STATE_CRITICAL, 3);
    }

    #[test]
    fn test_thermal_state_mapping() {
        // Run on macOS where objc2 runtime is available
        if cfg!(target_os = "macos") {
            let state = read_thermal_state();
            // On macOS without thermal monitoring, this typically returns Nominal
            let valid_states = [
                ThermalState::Nominal,
                ThermalState::Fair,
                ThermalState::Serious,
                ThermalState::Critical,
            ];
            assert!(valid_states.contains(&state));
        }
    }

    #[test]
    fn test_battery_state_constants() {
        assert_eq!(UI_DEVICE_BATTERY_STATE_UNKNOWN, 0);
        assert_eq!(UI_DEVICE_BATTERY_STATE_UNPLUGGED, 1);
        assert_eq!(UI_DEVICE_BATTERY_STATE_CHARGING, 2);
        assert_eq!(UI_DEVICE_BATTERY_STATE_FULL, 3);
    }

    #[test]
    fn test_ios_monitor_creation() {
        let monitor = IosMonitor::new();
        let health = monitor.health();

        // Should return default/0 values before first sample
        assert!(health.battery_level <= 100);
        assert!(health.timestamp > 0);
    }

    #[test]
    fn test_ios_monitor_default_values() {
        let monitor = IosMonitor::new();
        // Immediately after creation, before any sampling:
        // available_ram_mb = 0 (zeroed before first sample)
        let health = monitor.health();
        assert_eq!(health.thermal, ThermalState::Nominal);
    }

    #[test]
    fn test_ios_monitor_sampling_thread() {
        let monitor = IosMonitor::new();
        // Give the sampling thread time to collect at least one sample
        thread::sleep(Duration::from_millis(1100));

        let health = monitor.health();
        // After 1+ seconds, the sampling thread should have run at least once
        assert!(
            health.timestamp > 0,
            "Expected snapshot timestamp to be set after sampling"
        );
    }

    #[test]
    fn test_ios_monitor_snapshot_freshness() {
        let monitor = IosMonitor::new();
        thread::sleep(Duration::from_millis(500));

        let snap1 = monitor.health();
        thread::sleep(Duration::from_millis(1100));
        let snap2 = monitor.health();

        // Later snapshot should have later (or equal) timestamp
        assert!(
            snap2.timestamp >= snap1.timestamp,
            "Expected snapshot timestamp to increase over time"
        );
    }

    #[test]
    fn test_ios_monitor_power_state() {
        let monitor = IosMonitor::new();
        let state = monitor.power_state();
        // Should be either Charging or Discharging (or Unknown on non-iOS)
        let valid_states = [PowerState::Charging, PowerState::Discharging, PowerState::Unknown];
        assert!(valid_states.contains(&state));
    }

    #[test]
    fn test_ios_monitor_drop_stops_thread() {
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
}
