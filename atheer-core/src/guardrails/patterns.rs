use serde::{Deserialize, Serialize};

use crate::guardrails::normalizer::normalize_text;

/// A single injection detection pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEntry {
    pub pattern: String,
    pub score: f64,
    pub category: String,
}

/// A proximity-based detection pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityPair {
    pub word_a: String,
    pub word_b: String,
    pub max_distance: usize,
    pub score: f64,
    #[serde(default)]
    pub synonyms_a: Vec<String>,
    #[serde(default)]
    pub synonyms_b: Vec<String>,
}

/// Full pattern database for guardrail detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDatabase {
    #[serde(default)]
    pub high_severity: Vec<PatternEntry>,
    #[serde(default)]
    pub low_severity: Vec<PatternEntry>,
    #[serde(default)]
    pub proximity_pairs: Vec<ProximityPair>,
    #[serde(default)]
    pub system_prompt_leakage: Vec<String>,
    #[serde(default)]
    pub jailbreak_success_markers: Vec<String>,
}

impl PatternDatabase {
    /// Load the built-in pattern database (embedded at compile time).
    pub fn load_builtin() -> Self {
        let json = include_str!("builtin_patterns.json");
        serde_json::from_str(json).unwrap_or_else(|e| {
            tracing::error!(target: "atheer::guardrails", "Failed to parse builtin patterns: {e}");
            Self::empty()
        })
    }

    /// Load a pattern database from a sidecar JSON file.
    /// Falls back to builtin on any error.
    pub fn load_sidecar(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str::<PatternDatabase>(&contents) {
                Ok(db) => db,
                Err(e) => {
                    tracing::warn!(
                        target: "atheer::guardrails",
                        "Failed to parse sidecar patterns at {path}: {e}, using builtin"
                    );
                    Self::load_builtin()
                }
            },
            Err(e) => {
                tracing::debug!(
                    target: "atheer::guardrails",
                    "No sidecar patterns at {path}: {e}, using builtin"
                );
                Self::load_builtin()
            }
        }
    }

    /// Create an empty pattern database.
    pub fn empty() -> Self {
        Self {
            high_severity: Vec::new(),
            low_severity: Vec::new(),
            proximity_pairs: Vec::new(),
            system_prompt_leakage: Vec::new(),
            jailbreak_success_markers: Vec::new(),
        }
    }

    /// Merge another database into this one.
    ///
    /// The `other` database (e.g., sidecar) overrides `self` (e.g., builtin)
    /// at the individual pattern level: if `other` has a pattern with the same
    /// `pattern` string, the other entry replaces the builtin entry.
    /// New patterns in `other` that don't exist in `self` are appended.
    pub fn merge(&mut self, other: PatternDatabase) {
        // Merge high-severity
        merge_entries(&mut self.high_severity, other.high_severity);
        // Merge low-severity
        merge_entries(&mut self.low_severity, other.low_severity);
        // Proximity pairs: replace entirely if other has any
        if !other.proximity_pairs.is_empty() {
            self.proximity_pairs = other.proximity_pairs;
        }
        // System prompt leakage: replace entirely if other has any
        if !other.system_prompt_leakage.is_empty() {
            self.system_prompt_leakage = other.system_prompt_leakage;
        }
        // Jailbreak success markers: replace entirely if other has any
        if !other.jailbreak_success_markers.is_empty() {
            self.jailbreak_success_markers = other.jailbreak_success_markers;
        }
    }

    /// Append custom patterns as low-severity entries.
    pub fn append_custom(&mut self, patterns: Vec<String>) {
        for pattern in patterns {
            let normalized = normalize_text(&pattern);
            self.low_severity.push(PatternEntry {
                pattern: normalized,
                score: 0.5,
                category: "custom".to_string(),
            });
        }
    }
}

/// Merge `other` entries into `self` entries.
///
/// For each entry in `other`, if an entry with the same `pattern` string exists
/// in `self`, the `other` entry replaces it. Otherwise, the `other` entry is
/// appended to `self`.
fn merge_entries(self_entries: &mut Vec<PatternEntry>, other_entries: Vec<PatternEntry>) {
    for other_entry in other_entries {
        let other_normalized = normalize_text(&other_entry.pattern);
        if let Some(existing) = self_entries
            .iter_mut()
            .find(|e| normalize_text(&e.pattern) == other_normalized)
        {
            *existing = other_entry;
        } else {
            self_entries.push(other_entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_builtin() {
        let db = PatternDatabase::load_builtin();
        assert!(
            db.high_severity.len() >= 25,
            "Expected 25+ high-severity patterns, got {}",
            db.high_severity.len()
        );
        assert!(
            db.low_severity.len() >= 15,
            "Expected 15+ low-severity patterns, got {}",
            db.low_severity.len()
        );
        assert!(!db.proximity_pairs.is_empty());
        assert!(!db.system_prompt_leakage.is_empty());
        assert!(!db.jailbreak_success_markers.is_empty());
    }

    #[test]
    fn test_merge_overrides_same_pattern() {
        let mut builtin = PatternDatabase::load_builtin();

        let mut override_db = PatternDatabase::empty();
        override_db.high_severity.push(PatternEntry {
            pattern: builtin.high_severity[0].pattern.clone(),
            score: 0.5,
            category: "overridden".to_string(),
        });

        builtin.merge(override_db);

        let entry = builtin
            .high_severity
            .iter()
            .find(|e| e.category == "overridden")
            .expect("Override entry should exist");
        assert_eq!(entry.score, 0.5);
    }

    #[test]
    fn test_append_custom() {
        let mut db = PatternDatabase::empty();
        db.append_custom(vec!["test pattern".to_string()]);
        assert_eq!(db.low_severity.len(), 1);
        assert_eq!(db.low_severity[0].category, "custom");
        assert_eq!(db.low_severity[0].score, 0.5);
    }

    #[test]
    fn test_load_sidecar_missing_file() {
        let db = PatternDatabase::load_sidecar("/nonexistent/path.json");
        // Should fall back to builtin
        assert!(!db.high_severity.is_empty());
    }
}
