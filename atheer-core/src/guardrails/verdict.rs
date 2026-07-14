/// Outcome of a guardrail detection check.
///
/// Each verdict carries a score, category, and evidence string.
/// The `Pass` variant indicates no issues were detected.
#[derive(Debug, Clone)]
pub enum GuardrailVerdict {
    /// Generation must be aborted. The prompt was identified as a prompt injection.
    Block {
        score: f64,
        category: String,
        evidence: String,
    },
    /// Generation proceeds but a warning is surfaced to the app.
    Flag {
        score: f64,
        category: String,
        evidence: String,
    },
    /// No issues detected. Normal generation.
    Pass,
}

impl GuardrailVerdict {
    /// Returns `true` if this verdict requires aborting generation.
    pub fn is_blocked(&self) -> bool {
        matches!(self, GuardrailVerdict::Block { .. })
    }

    /// Returns `true` if this verdict is a warning (generation proceeds).
    pub fn is_flagged(&self) -> bool {
        matches!(self, GuardrailVerdict::Flag { .. })
    }

    /// Returns the detection score (0.0–1.0). `Pass` returns 0.0.
    pub fn score(&self) -> f64 {
        match self {
            GuardrailVerdict::Block { score, .. } | GuardrailVerdict::Flag { score, .. } => *score,
            GuardrailVerdict::Pass => 0.0,
        }
    }

    /// Returns the human-readable message for flagged/blocked verdicts.
    pub fn message(&self) -> Option<&str> {
        match self {
            GuardrailVerdict::Block { evidence, .. } | GuardrailVerdict::Flag { evidence, .. } => {
                Some(evidence.as_str())
            }
            GuardrailVerdict::Pass => None,
        }
    }

    /// Returns the category string for flagged/blocked verdicts.
    pub fn category(&self) -> Option<&str> {
        match self {
            GuardrailVerdict::Block { category, .. } | GuardrailVerdict::Flag { category, .. } => {
                Some(category.as_str())
            }
            GuardrailVerdict::Pass => None,
        }
    }
}
