use serde::{Deserialize, Serialize};

/// Runtime privacy mode governing crash reporting, persistence, and logging.
///
/// - `Normal` — current behavior: crash reports to disk, model caching, full logging.
/// - `Ephemeral` — no crash reports, no disk writes, no logging beyond ring buffer.
/// - `Audited` — full logging of every decision, network call, file write for compliance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMode {
    Normal,
    Ephemeral,
    Audited,
}
