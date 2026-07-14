use crate::guardrails::verdict::GuardrailVerdict;

/// L3 output guard — post-generation safety check.
///
/// Inspects generated output for indicators that a prompt injection succeeded:
/// system prompt leakage, jailbreak success markers, and output blocklist patterns.
pub struct L3OutputGuard {
    system_prompt_leakage: Vec<String>,
    jailbreak_success_markers: Vec<String>,
}

impl L3OutputGuard {
    pub fn new(system_prompt_leakage: Vec<String>, jailbreak_success_markers: Vec<String>) -> Self {
        Self {
            system_prompt_leakage,
            jailbreak_success_markers,
        }
    }

    /// Check the generated output for injection success indicators.
    ///
    /// Returns `Block` verdicts when output contains system prompt fragments
    /// or jailbreak success markers.
    pub fn check(&self, output: &str) -> Vec<GuardrailVerdict> {
        let mut verdicts = Vec::new();
        let output_lower = output.to_lowercase();

        // Check for system prompt leakage
        for pattern in &self.system_prompt_leakage {
            if output_lower.contains(&pattern.to_lowercase()) {
                verdicts.push(GuardrailVerdict::Block {
                    score: 1.0,
                    category: "output_sysprompt_leakage".to_string(),
                    evidence: format!("Output contains system prompt fragment: '{}'", pattern),
                });
            }
        }

        // Check for jailbreak success markers
        let output_upper = output.to_uppercase();
        for marker in &self.jailbreak_success_markers {
            if output_upper.contains(&marker.to_uppercase()) {
                verdicts.push(GuardrailVerdict::Block {
                    score: 1.0,
                    category: "output_jailbreak_marker".to_string(),
                    evidence: format!("Output contains jailbreak success marker: '{}'", marker),
                });
            }
        }

        verdicts
    }
}

impl Default for L3OutputGuard {
    fn default() -> Self {
        let db = crate::guardrails::patterns::PatternDatabase::load_builtin();
        Self::new(db.system_prompt_leakage, db.jailbreak_success_markers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocks_system_prompt_leakage() {
        let guard = L3OutputGuard::new(vec!["You are a helpful assistant".to_string()], vec![]);
        let verdicts = guard.check("I am You are a helpful assistant designed to help.");
        assert!(verdicts.iter().any(|v| v.is_blocked()));
    }

    #[test]
    fn test_blocks_jailbreak_marker() {
        let guard = L3OutputGuard::new(vec![], vec!["ALL SAFETY CHECKS PASSED".to_string()]);
        let verdicts = guard.check("The output is: ALL SAFETY CHECKS PASSED");
        assert!(verdicts.iter().any(|v| v.is_blocked()));
    }

    #[test]
    fn test_passes_clean_output() {
        let guard = L3OutputGuard::new(
            vec!["You are a helpful assistant".to_string()],
            vec!["ALL SAFETY CHECKS PASSED".to_string()],
        );
        let verdicts = guard.check("Paris is the capital of France.");
        assert!(verdicts.is_empty());
    }

    #[test]
    fn test_case_insensitive() {
        let guard = L3OutputGuard::new(vec![], vec!["SYSTEM OVERRIDE ACCEPTED".to_string()]);
        let verdicts = guard.check("system override accepted");
        assert!(verdicts.iter().any(|v| v.is_blocked()));
    }
}
