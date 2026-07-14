use std::collections::HashMap;

use crate::guardrails::verdict::GuardrailVerdict;

/// L2 token-level anomaly detector.
///
/// Analyzes tokenized input for statistical anomalies indicative of
/// prompt injection: token repetition, entropy anomalies, and adversarial suffixes.
pub struct L2Analyzer;

impl L2Analyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze token IDs for anomalies.
    ///
    /// Returns an empty vector (score 0.0) when token count < 10
    /// (insufficient data for reliable analysis).
    pub fn analyze(&self, token_ids: &[u32]) -> Vec<GuardrailVerdict> {
        // Early-exit: short prompts don't have enough data
        if token_ids.len() < 10 {
            return Vec::new();
        }

        let mut verdicts = Vec::new();

        // Check token repetition ratio
        let repetition_score = self.compute_repetition_ratio(token_ids);
        if repetition_score > 0.65 {
            verdicts.push(GuardrailVerdict::Flag {
                score: repetition_score,
                category: "token_repetition_anomaly".to_string(),
                evidence: format!(
                    "High token repetition ratio: {:.2} (threshold: 0.65)",
                    repetition_score
                ),
            });
        }

        // Check entropy anomaly
        let entropy_score = self.compute_entropy_anomaly(token_ids);
        if entropy_score > 0.7 {
            verdicts.push(GuardrailVerdict::Flag {
                score: entropy_score,
                category: "entropy_anomaly".to_string(),
                evidence: format!(
                    "Unusual token entropy distribution: {:.2} (threshold: 0.70)",
                    entropy_score
                ),
            });
        }

        // Check adversarial suffix
        let suffix_score = self.detect_adversarial_suffix(token_ids);
        if suffix_score > 0.7 {
            verdicts.push(GuardrailVerdict::Flag {
                score: suffix_score,
                category: "adversarial_suffix".to_string(),
                evidence: format!(
                    "Adversarial suffix detected: {:.2} of tokens are non-informative (threshold: 0.70)",
                    suffix_score
                ),
            });
        }

        verdicts
    }

    /// Compute the ratio of repeated n-grams to total tokens.
    ///
    /// Returns the MAX of bigram and trigram repetition ratios (not their sum),
    /// since n-gram repetition overlaps significantly between bigram and trigram
    /// analysis, and summing them would double-count closely related patterns.
    fn compute_repetition_ratio(&self, token_ids: &[u32]) -> f64 {
        if token_ids.len() < 4 {
            return 0.0;
        }

        let total = token_ids.len() as f64;

        // Bigram repetition
        let mut bigram_counts: HashMap<(u32, u32), usize> = HashMap::new();
        for window in token_ids.windows(2) {
            let bigram = (window[0], window[1]);
            *bigram_counts.entry(bigram).or_insert(0) += 1;
        }
        let bigram_repeated: usize = bigram_counts
            .values()
            .filter(|&&count| count > 1)
            .map(|&count| count - 1)
            .sum();
        let bigram_ratio = bigram_repeated as f64 / total;

        // Trigram repetition
        let mut trigram_counts: HashMap<(u32, u32, u32), usize> = HashMap::new();
        for window in token_ids.windows(3) {
            let trigram = (window[0], window[1], window[2]);
            *trigram_counts.entry(trigram).or_insert(0) += 1;
        }
        let trigram_repeated: usize = trigram_counts
            .values()
            .filter(|&&count| count > 1)
            .map(|&count| count - 1)
            .sum();
        let trigram_ratio = trigram_repeated as f64 / total;

        bigram_ratio.max(trigram_ratio).min(1.0)
    }

    /// Compute entropy-based anomaly score.
    ///
    /// Low entropy (many repeated tokens) is anomalous.
    /// High entropy (uniform distribution) is normal.
    /// Score is inverted: 1.0 - normalized_entropy.
    fn compute_entropy_anomaly(&self, token_ids: &[u32]) -> f64 {
        if token_ids.is_empty() {
            return 0.0;
        }

        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &id in token_ids {
            *counts.entry(id).or_insert(0) += 1;
        }

        let n = token_ids.len() as f64;
        let k = counts.len() as f64;

        // Shannon entropy
        let entropy: f64 = counts
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        // Normalize: max entropy is log2(k)
        let max_entropy = if k > 1.0 { k.log2() } else { 1.0 };
        let normalized = entropy / max_entropy;

        // Anomaly score: 1.0 - normalized entropy (low entropy = high anomaly)
        (1.0 - normalized).clamp(0.0, 1.0)
    }

    /// Detect adversarial suffixes: trailing runs of non-informative tokens.
    ///
    /// Counts the proportion of trailing tokens that are:
    /// - Single repeated token (e.g., "! ! ! !" or "AAAA AAAA")
    /// - Punctuation clusters
    fn detect_adversarial_suffix(&self, token_ids: &[u32]) -> f64 {
        if token_ids.len() < 5 {
            return 0.0;
        }

        // Scan from the end, count non-informative trailing tokens
        let mut trailing_count = 0usize;
        let mut i = token_ids.len();

        while i > 0 {
            i -= 1;
            let token = token_ids[i];

            // Heuristic: tokens < 128 that are ASCII punctuation or whitespace
            // are considered non-informative in adversarial suffixes
            if token < 128 {
                let ch = token as u8 as char;
                if ch.is_ascii_punctuation() || ch.is_ascii_whitespace() {
                    trailing_count += 1;
                    continue;
                }
            }

            // Check for repeated same-token runs
            if i + 2 < token_ids.len() {
                let next = token_ids[i + 1];
                let next2 = token_ids[i + 2];
                if token == next && token == next2 {
                    trailing_count += 1;
                    continue;
                }
            }

            // Non-non-informative token — stop scanning
            break;
        }

        // Score: proportion of trailing non-informative tokens
        (trailing_count as f64 / token_ids.len() as f64).min(1.0)
    }
}

impl Default for L2Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_prompt_skipped() {
        let analyzer = L2Analyzer::new();
        let token_ids: Vec<u32> = (0..5).collect();
        let verdicts = analyzer.analyze(&token_ids);
        assert!(verdicts.is_empty());
    }

    #[test]
    fn test_clean_prompt_passes() {
        let analyzer = L2Analyzer::new();
        // Simulate a normal diverse token sequence
        let token_ids: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let verdicts = analyzer.analyze(&token_ids);
        // Should have no high-severity flags
        assert!(verdicts.iter().all(|v| !v.is_blocked()));
    }

    #[test]
    fn test_repeated_tokens_detected() {
        let analyzer = L2Analyzer::new();
        // Highly repeated tokens
        let token_ids: Vec<u32> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3];
        let verdicts = analyzer.analyze(&token_ids);
        assert!(
            verdicts.iter().any(|v| v.is_flagged()),
            "Should detect token repetition"
        );
    }
}
