//! Android hardware telemetry via JNI.
//!
//! Reads thermal, memory, and battery status from Android Java APIs using
//! the `jni` crate.  Requires the host app to call [`init_jni`] early in
//! startup with a `JavaVM` pointer and the application `Context`.

use crate::{HardwareMonitor, HealthStatus, MemoryStatus, PowerState, ThermalState};
use jni::objects::{GlobalRef, JObject, JValue};
use jni::JNIEnv;
use jni::JavaVM;
use std::sync::OnceLock;

// Android system service name constants
const ACTIVITY_SERVICE: &str = "activity";
const BATTERY_SERVICE: &str = "batterymanager";
const THERMAL_SERVICE: &str = "thermalservice";

// ---------------------------------------------------------------------------
// Global JNI state
// ---------------------------------------------------------------------------

static JVM: OnceLock<JavaVM> = OnceLock::new();
static APP_CONTEXT: OnceLock<GlobalRef> = OnceLock::new();

/// Initialise the JNI bridge with a JVM reference and Android Context.
///
/// Must be called from the application's main thread before any telemetry
/// sampling starts.  Typical call site is `Application.onCreate()`.
pub fn init_jni(jvm: JavaVM, context: &JObject<'_>) -> Result<(), JniInitError> {
    JVM.set(jvm).map_err(|_| JniInitError::AlreadyInitialized)?;
    let mut env = JVM
        .get()
        .expect("just set")
        .attach_current_thread()
        .map_err(|e| JniInitError::AttachFailed(e.to_string()))?;
    let global = env
        .new_global_ref(context)
        .map_err(|e| JniInitError::GlobalRefFailed(e.to_string()))?;
    APP_CONTEXT
        .set(global)
        .map_err(|_| JniInitError::AlreadyInitialized)
}

/// Errors from JNI initialisation.
#[derive(Debug)]
pub enum JniInitError {
    AlreadyInitialized,
    AttachFailed(String),
    GlobalRefFailed(String),
}

impl std::fmt::Display for JniInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JniInitError::AlreadyInitialized => write!(f, "JNI already initialized"),
            JniInitError::AttachFailed(e) => write!(f, "thread attach failed: {e}"),
            JniInitError::GlobalRefFailed(e) => write!(f, "global ref failed: {e}"),
        }
    }
}

/// Check whether JNI has been initialised.
pub fn is_initialized() -> bool {
    JVM.get().is_some() && APP_CONTEXT.get().is_some()
}

// ---------------------------------------------------------------------------
// JNI helper — run a closure with a JNIEnv attached to the current thread
// ---------------------------------------------------------------------------

fn with_env<F, T>(f: F) -> Option<T>
where
    F: FnOnce(&mut JNIEnv<'_>, &JObject<'_>) -> Option<T>,
{
    let jvm = JVM.get()?;
    let context_ref = APP_CONTEXT.get()?;
    let mut env = jvm.attach_current_thread().ok()?;
    f(&mut env, context_ref.as_obj())
}

// ---------------------------------------------------------------------------
// Thermal state via ThermalManager (API 30+)
// ---------------------------------------------------------------------------

/// Returns the thermal headroom in Celsius above the throttling threshold.
///
/// Calls `ThermalManager.getThermalHeadroom()` via JNI.
/// Returns `None` if JNI is not initialised or the API level is below 30.
pub fn thermal_headroom() -> Option<f32> {
    with_env(|env, context| {
        let service_name = env.new_string(THERMAL_SERVICE).ok()?;
        let thermal_manager = env
            .call_method(
                context,
                "getSystemService",
                "(Ljava/lang/String;)Ljava/lang/Object;",
                &[JValue::Object(&service_name.into())],
            )
            .ok()?;
        let tm_obj: JObject<'_> = thermal_manager.l()?;

        let headroom: f32 = env
            .call_method(&tm_obj, "getThermalHeadroom", "()F", &[])
            .ok()?
            .f()
            .ok()?;

        Some(headroom)
    })
}

/// Convert thermal headroom (Celsius above throttling) to ThermalState.
pub fn headroom_to_state(headroom_c: f32) -> ThermalState {
    if headroom_c <= 0.0 {
        ThermalState::Critical
    } else if headroom_c < 5.0 {
        ThermalState::Serious
    } else if headroom_c < 10.0 {
        ThermalState::Fair
    } else {
        ThermalState::Nominal
    }
}

// ---------------------------------------------------------------------------
// Memory via ActivityManager.MemoryInfo
// ---------------------------------------------------------------------------

/// Returns (available_mb, total_mb).
pub fn memory_mb() -> Option<(u64, u64)> {
    with_env(|env, context| {
        let service_name = env.new_string(ACTIVITY_SERVICE).ok()?;
        let activity_manager = env
            .call_method(
                context,
                "getSystemService",
                "(Ljava/lang/String;)Ljava/lang/Object;",
                &[JValue::Object(&service_name.into())],
            )
            .ok()?;
        let am_obj: JObject<'_> = activity_manager.l()?;

        let mem_info_class = env
            .find_class("android/app/ActivityManager$MemoryInfo")
            .ok()?;
        let mem_info = env.new_object(mem_info_class, "()V", &[]).ok()?;

        env.call_method(
            &am_obj,
            "getMemoryInfo",
            "(Landroid/app/ActivityManager$MemoryInfo;)V",
            &[JValue::Object(&mem_info)],
        )
        .ok()?;

        let avail_mem: i64 = env
            .get_field(&mem_info, "availMem", "J")
            .ok()?
            .i()
            .unwrap_or(0);
        let total_mem: i64 = env
            .get_field(&mem_info, "totalMem", "J")
            .ok()?
            .i()
            .unwrap_or(1);

        Some((
            (avail_mem as u64) / (1024 * 1024),
            (total_mem as u64) / (1024 * 1024),
        ))
    })
}

// ---------------------------------------------------------------------------
// Battery via BatteryManager
// ---------------------------------------------------------------------------

/// Returns (battery_level_percent, is_charging).
pub fn battery_info() -> Option<(u32, bool)> {
    with_env(|env, context| {
        let service_name = env.new_string(BATTERY_SERVICE).ok()?;
        let battery_manager = env
            .call_method(
                context,
                "getSystemService",
                "(Ljava/lang/String;)Ljava/lang/Object;",
                &[JValue::Object(&service_name.into())],
            )
            .ok()?;
        let bm_obj: JObject<'_> = battery_manager.l()?;

        // BatteryManager.getIntProperty(BATTERY_PROPERTY_CAPACITY = 4)
        let capacity: i32 = env
            .call_method(&bm_obj, "getIntProperty", "(I)I", &[JValue::Int(4)])
            .ok()?
            .i()
            .unwrap_or(100);

        // BatteryManager.getIntProperty(BATTERY_PROPERTY_IS_CHARGING = 5)
        let charging: i32 = env
            .call_method(&bm_obj, "getIntProperty", "(I)I", &[JValue::Int(5)])
            .ok()?
            .i()
            .unwrap_or(0);

        let level = capacity.max(0).min(100) as u32;
        Some((level, charging != 0))
    })
}

fn battery_to_power_state() -> PowerState {
    battery_info()
        .map(|(_, charging)| {
            if charging {
                PowerState::Charging
            } else {
                PowerState::Discharging
            }
        })
        .unwrap_or(PowerState::Unknown)
}

// ---------------------------------------------------------------------------
// AndroidMonitor — HardwareMonitor implementation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AndroidMonitor;

impl AndroidMonitor {
    pub fn new() -> Self {
        Self
    }

    /// Sample all hardware metrics in one JNI-attached call.
    pub fn sample(&self) -> Option<HealthStatus> {
        let thermal = thermal_headroom()
            .map(headroom_to_state)
            .unwrap_or(ThermalState::Nominal);
        let (avail_mb, total_mb) = memory_mb().unwrap_or((2048, 4096));
        let (level, charging) = battery_info().unwrap_or((100, true));

        Some(HealthStatus {
            thermal,
            available_ram_mb: avail_mb,
            total_ram_mb: total_mb,
            battery_level: level,
            on_battery: !charging,
            timestamp: chrono::Utc::now().timestamp(),
        })
    }
}

impl Default for AndroidMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareMonitor for AndroidMonitor {
    fn health(&self) -> HealthStatus {
        self.sample().unwrap_or(HealthStatus {
            thermal: ThermalState::Nominal,
            available_ram_mb: 2048,
            total_ram_mb: 4096,
            battery_level: 100,
            on_battery: false,
            timestamp: chrono::Utc::now().timestamp(),
        })
    }

    fn thermal_state(&self) -> ThermalState {
        thermal_headroom()
            .map(headroom_to_state)
            .unwrap_or(ThermalState::Nominal)
    }

    fn memory_status(&self) -> MemoryStatus {
        memory_mb()
            .map(|(avail, total)| MemoryStatus {
                available_mb: avail,
                total_mb: total,
                low_memory_threshold_mb: 800,
            })
            .unwrap_or(MemoryStatus {
                available_mb: 2048,
                total_mb: 4096,
                low_memory_threshold_mb: 800,
            })
    }

    fn power_state(&self) -> PowerState {
        battery_to_power_state()
    }

    fn battery_level(&self) -> u32 {
        battery_info().map(|(l, _)| l).unwrap_or(100)
    }

    fn is_on_battery(&self) -> bool {
        battery_info().map(|(_, c)| !c).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_headroom_to_state() {
        assert_eq!(headroom_to_state(15.0), ThermalState::Nominal);
        assert_eq!(headroom_to_state(7.5), ThermalState::Fair);
        assert_eq!(headroom_to_state(3.0), ThermalState::Serious);
        assert_eq!(headroom_to_state(0.0), ThermalState::Critical);
        assert_eq!(headroom_to_state(-1.0), ThermalState::Critical);
    }

    #[test]
    fn test_jni_not_initialized_by_default() {
        assert!(!is_initialized());
    }

    #[test]
    fn test_android_monitor_creation() {
        let monitor = AndroidMonitor::new();
        let health = monitor.health();
        assert!(health.available_ram_mb > 0);
        assert!(health.battery_level <= 100);
    }

    #[test]
    fn test_sample_without_jni() {
        let monitor = AndroidMonitor::new();
        assert!(monitor.sample().is_some());
    }

    #[test]
    fn test_memory_fallback() {
        let result = memory_mb();
        assert!(result.is_none());
    }

    #[test]
    fn test_battery_fallback() {
        let result = battery_info();
        assert!(result.is_none());
    }

    #[test]
    fn test_thermal_fallback() {
        let result = thermal_headroom();
        assert!(result.is_none());
    }

    #[test]
    fn test_power_state_without_jni() {
        let monitor = AndroidMonitor::new();
        assert_eq!(monitor.power_state(), PowerState::Unknown);
    }
}
