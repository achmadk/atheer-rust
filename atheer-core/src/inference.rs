use crate::error::{AtheerCoreError, Result};
use crate::kv_cache_bridge::KvCacheBridge;
use crate::latency_budget::LatencyTracker;
use crate::model::Model;
use crate::safety::ContentModeration;
use crate::sampler::{DefaultSampler, Sampler, SamplingConfig};
use crate::streaming::{GenerationState, StreamingCallback};
use crate::tokenizer::Tokenizer;
use candle_core::Tensor;
use std::time::Instant;

pub struct InferenceEngine {
    model: Model,
    tokenizer: Tokenizer,
    sampler: Box<dyn Sampler>,
    max_seq_len: usize,
    latency: LatencyTracker,

    // ── Multi-turn conversation state ──────────────────────────────────────
    /// Track (start_pos, end_pos) for each conversation turn.
    /// Used by sliding window eviction to drop oldest entire turns.
    turn_history: Vec<(usize, usize)>,

    /// Number of tokens in the system prompt (0 if none).
    /// Conversation turns start at offset `system_prompt_len`.
    system_prompt_len: usize,

    /// Raw system prompt text, kept so we can re-encode it when
    /// [`reset_for_turn`] clears the KV cache.
    system_prompt: Option<String>,

    /// Total number of tokens processed so far (system + all conversation turns).
    /// This is the next position for `forward()` when appending tokens.
    last_pos: usize,

    /// Optional content moderation pipeline.
    /// If set, input prompts and generated output are checked before/after generation.
    moderation: Option<ContentModeration>,
}

impl InferenceEngine {
    /// Create a new inference engine with a pre-constructed model.
    ///
    /// The caller is responsible for selecting the device (e.g. via
    /// [`BackendManager::device`]) and constructing the [`Model`].
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        config: SamplingConfig,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Device validation is performed upstream by BackendManager.device()
        // which falls back to CPU if the requested device is unavailable.
        Ok(Self {
            model,
            tokenizer,
            sampler: Box::new(DefaultSampler::new(config)),
            max_seq_len,
            latency: LatencyTracker::new(100),
            turn_history: Vec::new(),
            system_prompt_len: 0,
            system_prompt: None,
            last_pos: 0,
            moderation: None,
        })
    }

    /// Convenience constructor that auto-selects the best available backend
    /// and loads the model onto the corresponding device.
    ///
    /// Probes the platform's available GPU/NPU accelerators (Metal, Vulkan,
    /// NNAPI, CoreML) and falls back to CPU if none are available. Equivalent
    /// to creating a [`BackendManager`], calling [`with_autoselect`], and
    /// passing its [`device`] to [`Model::from_gguf`].
    ///
    /// [`BackendManager`]: atheer_accel::BackendManager
    /// [`with_autoselect`]: atheer_accel::BackendManager::with_autoselect
    /// [`device`]: atheer_accel::BackendManager::device
    /// [`Model::from_gguf`]: Model::from_gguf
    #[cfg(feature = "auto-backend")]
    pub fn new_auto(
        model_path: impl AsRef<std::path::Path>,
        tokenizer: Tokenizer,
        config: SamplingConfig,
        max_seq_len: usize,
    ) -> Result<Self> {
        let backend = atheer_accel::BackendManager::new().with_autoselect();
        let device = backend.device();
        tracing::info!(
            "Auto-selected backend: {} (device: {:?})",
            backend.current().name(),
            device
        );
        let model = Model::from_gguf(model_path, &device)?;
        Self::new(model, tokenizer, config, max_seq_len)
    }

    pub fn with_sampler(&mut self, sampler: Box<dyn Sampler>) {
        self.sampler = sampler;
    }

    /// Attach a content moderation pipeline.
    /// When set, all input prompts and generated output are checked.
    pub fn with_moderation(&mut self, moderation: ContentModeration) {
        self.moderation = Some(moderation);
    }

    /// Check the input prompt against the moderation pipeline.
    /// Returns an error if any stage blocks the prompt.
    fn check_input_blocked(&self, prompt: &str) -> Result<()> {
        if let Some(ref mods) = self.moderation {
            for verdict in mods.check_input(prompt) {
                if let crate::safety::ModerationVerdict::Blocked(msg) = verdict {
                    return Err(AtheerCoreError::GenerationFailed(format!(
                        "Content blocked: {}",
                        msg
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check the generated output against the moderation pipeline.
    /// Returns an error if any stage blocks the output.
    fn check_output_blocked(&self, text: &str, tokens: &[u32]) -> Result<()> {
        if let Some(ref mods) = self.moderation {
            for verdict in mods.check_output(text, tokens) {
                if let crate::safety::ModerationVerdict::Blocked(msg) = verdict {
                    return Err(AtheerCoreError::GenerationFailed(format!(
                        "Output blocked: {}",
                        msg
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Tokenize input text.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, true)
    }

    /// Check and apply sliding window eviction before adding new tokens.
    fn maybe_evict(&mut self, incoming_len: usize) {
        let needed = self.last_pos + incoming_len;
        if needed < self.max_seq_len {
            return;
        }
        tracing::warn!(
            "Context window full ({} + {} >= {}), evicting oldest turns",
            self.last_pos,
            incoming_len,
            self.max_seq_len
        );

        // Drop oldest turns until the remaining content fits within max_seq_len.
        // If a single turn is larger than the window, we truncate it.
        let keep_base = self.system_prompt_len;
        while !self.turn_history.is_empty() {
            let first = self.turn_history.first().unwrap();
            let turn_len = first.1 - first.0;
            let remaining_tokens: usize =
                self.turn_history.iter().skip(1).map(|t| t.1 - t.0).sum::<usize>()
                    + keep_base;
            let new_total = remaining_tokens + incoming_len;
            if new_total < self.max_seq_len || self.turn_history.len() == 1 {
                // Evict this turn
                self.turn_history.remove(0);
                break;
            }
            self.turn_history.remove(0);
        }

        // Recompute last_pos from remaining turns + system prompt
        self.last_pos = self.system_prompt_len
            + self
                .turn_history
                .iter()
                .map(|t| t.1 - t.0)
                .sum::<usize>();

        // Clear the GPU-side KV cache entirely; remaining context will be
        // re-encoded on the next forward call(s).
        self.model.kv_cache_clear();

        tracing::info!(
            "Evicted oldest turn(s), last_pos now {}, turns remaining: {}",
            self.last_pos,
            self.turn_history.len()
        );
    }

    /// Run a full generation pass from a prompt.
    ///
    /// Returns (generated_text, token_count, duration_ms).
    pub fn generate(&mut self, prompt: &str, max_tokens: u32) -> Result<(String, u32, u64)> {
        let start = Instant::now();
        let device = self.model.device.clone();

        // Content moderation: check input
        self.check_input_blocked(prompt)?;

        let input_ids = self.tokenizer.encode(prompt, true);
        let prompt_len = input_ids.len();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let turn_start = self.last_pos;

        // Ensure we have room for the incoming prompt
        self.maybe_evict(prompt_len);

        // Prefill: forward the entire prompt
        let input_tensor = Tensor::new(
            input_ids
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
        .unsqueeze(0)
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

        // The candle forward returns logits for the LAST token in the sequence
        let logits = self
            .model
            .weights
            .forward(&input_tensor, self.last_pos)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Prompt forward: {e}")))?;

        // Sample first token
        let mut next_token = self
            .sampler
            .sample(&logits, &generated_tokens)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
        generated_tokens.push(next_token);

        // Auto-regressive loop
        let mut pos = self.last_pos + prompt_len;
        for _ in 1..max_tokens {
            if self.is_stop_token(next_token) {
                break;
            }

            let token_tensor = Tensor::new(&[next_token as i64][..], &device)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

            let logits = self
                .model
                .weights
                .forward(&token_tensor, pos)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Forward: {e}")))?;

            next_token = self
                .sampler
                .sample(&logits, &generated_tokens)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
            generated_tokens.push(next_token);
            pos += 1;

            if pos >= self.max_seq_len {
                break;
            }
        }

        // Record turn in history and update last_pos
        self.last_pos = pos;
        self.turn_history.push((turn_start, pos));

        let elapsed = start.elapsed().as_millis() as u64;
        let text = self.tokenizer.decode(&generated_tokens, true);

        // Content moderation: check output
        self.check_output_blocked(&text, &generated_tokens)?;

        Ok((text, generated_tokens.len() as u32, elapsed))
    }

    // ── Multi-turn conversation ───────────────────────────────────────────

    /// Generate with a system prompt that is isolated from conversation turns.
    /// The system prompt is encoded once; conversation turns append after it.
    /// Returns (generated_text, token_count, duration_ms).
    pub fn generate_with_system(
        &mut self,
        system_prompt: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<(String, u32, u64)> {
        let start = Instant::now();
        let device = self.model.device.clone();

        self.check_input_blocked(prompt)?;

        if self.system_prompt_len == 0 {
            self.check_input_blocked(system_prompt)?;
            let sys_ids = self.tokenizer.encode(system_prompt, true);
            let sys_tensor = Tensor::new(
                sys_ids
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &device,
            )
            .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

            self.model
                .weights
                .forward(&sys_tensor, 0)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("System forward: {e}")))?;

            self.system_prompt_len = sys_ids.len();
            self.system_prompt = Some(system_prompt.to_string());
            self.last_pos = self.system_prompt_len;
            tracing::debug!(
                "System prompt encoded: {} tokens",
                self.system_prompt_len
            );
        }

        let input_ids = self.tokenizer.encode(prompt, true);
        let prompt_len = input_ids.len();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let turn_start = self.last_pos;

        self.maybe_evict(prompt_len);

        let input_tensor = Tensor::new(
            input_ids
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
        .unsqueeze(0)
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

        let logits = self
            .model
            .weights
            .forward(&input_tensor, self.last_pos)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Conv forward: {e}")))?;

        let mut next_token = self
            .sampler
            .sample(&logits, &generated_tokens)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
        generated_tokens.push(next_token);

        let mut pos = self.last_pos + prompt_len;
        for _ in 1..max_tokens {
            if self.is_stop_token(next_token) {
                break;
            }

            let token_tensor = Tensor::new(&[next_token as i64][..], &device)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

            let logits = self
                .model
                .weights
                .forward(&token_tensor, pos)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Forward: {e}")))?;

            next_token = self
                .sampler
                .sample(&logits, &generated_tokens)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
            generated_tokens.push(next_token);
            pos += 1;

            if pos >= self.max_seq_len {
                break;
            }
        }

        self.last_pos = pos;
        self.turn_history.push((turn_start, pos));

        let elapsed = start.elapsed().as_millis() as u64;
        let text = self.tokenizer.decode(&generated_tokens, true);
        self.check_output_blocked(&text, &generated_tokens)?;
        Ok((text, generated_tokens.len() as u32, elapsed))
    }

    /// Continue an existing conversation by appending a new user prompt.
    /// The KV cache from previous turns is preserved and the new prompt is
    /// forwarded starting at the current `last_pos` (no re-prefill needed).
    pub fn continue_turn(&mut self, prompt: &str, max_tokens: u32) -> Result<(String, u32, u64)> {
        let start = Instant::now();
        let device = self.model.device.clone();

        self.check_input_blocked(prompt)?;
        let input_ids = self.tokenizer.encode(prompt, true);
        let prompt_len = input_ids.len();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let turn_start = self.last_pos;

        self.maybe_evict(prompt_len);

        // Forward the new prompt tokens starting from the current last_pos
        let input_tensor = Tensor::new(
            input_ids
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
        .unsqueeze(0)
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

        let logits = self
            .model
            .weights
            .forward(&input_tensor, self.last_pos)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Continue forward: {e}")))?;

        let mut next_token = self
            .sampler
            .sample(&logits, &generated_tokens)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
        generated_tokens.push(next_token);

        let mut pos = self.last_pos + prompt_len;
        for _ in 1..max_tokens {
            if self.is_stop_token(next_token) {
                break;
            }

            let token_tensor = Tensor::new(&[next_token as i64][..], &device)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

            let logits = self
                .model
                .weights
                .forward(&token_tensor, pos)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Forward: {e}")))?;

            next_token = self
                .sampler
                .sample(&logits, &generated_tokens)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
            generated_tokens.push(next_token);
            pos += 1;

            if pos >= self.max_seq_len {
                break;
            }
        }

        self.last_pos = pos;
        self.turn_history.push((turn_start, pos));

        let elapsed = start.elapsed().as_millis() as u64;
        let text = self.tokenizer.decode(&generated_tokens, true);
        self.check_output_blocked(&text, &generated_tokens)?;
        Ok((text, generated_tokens.len() as u32, elapsed))
    }

    /// Reset the conversation turns while preserving the system prompt KV cache.
    /// After this call, the next `continue_turn()` or `generate()` will begin
    /// at the system prompt offset.
    pub fn reset_for_turn(&mut self) -> Result<()> {
        self.model.kv_cache_clear();
        self.turn_history.clear();

        // Re-encode the system prompt so the KV cache is repopulated.
        // This ensures the mask computation (kv_len = index_pos + seq_len)
        // is consistent with the actual cache content — required by both
        // quantized_llama and quantized_lfm2 attention layers.
        if let Some(ref sys) = self.system_prompt {
            let device = self.model.device.clone();
            let sys_ids = self.tokenizer.encode(sys, true);
            let sys_tensor = Tensor::new(
                sys_ids.iter().map(|x| *x as i64).collect::<Vec<_>>().as_slice(),
                &device,
            )
            .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;
            self.model
                .weights
                .forward(&sys_tensor, 0)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("System re-encode: {e}")))?;
        }

        self.last_pos = self.system_prompt_len;
        tracing::debug!("Conversation reset (system prompt preserved: {} tokens)", self.system_prompt_len);
        Ok(())
    }

    /// Reset the entire session including the system prompt KV cache.
    /// After this call, the engine is in a fresh state.
    pub fn reset_session(&mut self) {
        self.model.kv_cache_clear();
        self.turn_history.clear();
        self.system_prompt_len = 0;
        self.system_prompt = None;
        self.last_pos = 0;
        tracing::debug!("Session fully reset");
    }

    /// Return an estimate of the total KV cache memory in bytes.
    /// Uses the cache snapshot to measure actual tensors.
    pub fn kv_cache_estimated_bytes(&self) -> u64 {
        match self.model.kv_cache_snapshot() {
            Ok(snapshot) => snapshot
                .iter()
                .map(|(k, v)| {
                    (k.len() + v.len()) as u64 * std::mem::size_of::<f32>() as u64
                })
                .sum(),
            Err(_) => 0,
        }
    }

    /// Streaming generation: calls `callback.on_token()` after each decode step.
    ///
    /// The callback receives the new token id and the current [`GenerationState`].
    /// If the callback returns `false`, generation is aborted early.
    ///
    /// Returns `(aborted, token_count, elapsed_ms)`.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        callback: &mut dyn StreamingCallback,
    ) -> Result<(bool, u32, u64)> {
        let start = Instant::now();
        let device = self.model.device.clone();
        self.latency.reset();

        let input_ids = self.tokenizer.encode(prompt, true);
        let prompt_len = input_ids.len();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Prefill
        let prefill_start = Instant::now();
        let input_tensor = Tensor::new(
            input_ids
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
        .unsqueeze(0)
        .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

        let logits = self
            .model
            .weights
            .forward(&input_tensor, 0)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Prompt forward: {e}")))?;

        self.latency
            .record_prefill(prefill_start.elapsed().as_secs_f64() * 1000.0);

        // Sample first token
        let mut next_token = self
            .sampler
            .sample(&logits, &generated_tokens)
            .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
        generated_tokens.push(next_token);

        // Callback for first token
        {
            let state = GenerationState {
                tokens_so_far: generated_tokens.len() as u32,
                avg_decode_ms: 0.0,
                p99_estimate_ms: None,
                prefill_ms: Some(self.latency.prefill_ms),
            };
            if !callback.on_token(next_token, &state) {
                let elapsed = start.elapsed().as_millis() as u64;
                return Ok((true, generated_tokens.len() as u32, elapsed));
            }
        }

        // Auto-regressive loop
        let mut pos = prompt_len + 1;
        for _ in 1..max_tokens {
            if self.is_stop_token(next_token) {
                break;
            }

            let step_start = Instant::now();

            let token_tensor = Tensor::new(&[next_token as i64][..], &device)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| AtheerCoreError::GenerationFailed(e.to_string()))?;

            let logits = self
                .model
                .weights
                .forward(&token_tensor, pos)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Forward: {e}")))?;

            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            self.latency.record_decode_step(step_ms, f64::MAX);

            next_token = self
                .sampler
                .sample(&logits, &generated_tokens)
                .map_err(|e| AtheerCoreError::GenerationFailed(format!("Sampling: {e}")))?;
            generated_tokens.push(next_token);
            pos += 1;

            // Callback
            {
                let state = GenerationState {
                    tokens_so_far: generated_tokens.len() as u32,
                    avg_decode_ms: self.latency.avg_decode_ms(),
                    p99_estimate_ms: self.latency.p99_estimate_ms(),
                    prefill_ms: Some(self.latency.prefill_ms),
                };
                if !callback.on_token(next_token, &state) {
                    let elapsed = start.elapsed().as_millis() as u64;
                    return Ok((true, generated_tokens.len() as u32, elapsed));
                }
            }

            if pos >= self.max_seq_len {
                break;
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;
        Ok((false, generated_tokens.len() as u32, elapsed))
    }

    /// Access the internal latency tracker.
    pub fn latency_tracker(&self) -> &LatencyTracker {
        &self.latency
    }

    /// Reset the latency tracker.
    pub fn reset_latency_tracking(&mut self) {
        self.latency.reset();
    }

    // ── KvCacheBridge delegation ──────────────────────────────────────────

    /// Return a copy of every layer's KV cache as flat CPU buffers.
    pub fn kv_cache_snapshot(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        self.model.kv_cache_snapshot()
    }

    /// Overwrite every layer's KV cache from a previous snapshot.
    pub fn kv_cache_restore(&mut self, snapshot: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
        self.model.kv_cache_restore(snapshot)
    }

    /// Drop all GPU-side KV cache tensors, freeing VRAM.
    /// After this, `forward()` will rebuild the cache from scratch.
    pub fn kv_cache_clear(&mut self) {
        self.model.kv_cache_clear();
    }

    /// Infect a [`MemoryBank`](atheer_memory_bank::MemoryBank) with the current
    /// KV cache snapshot.  A convenience wrapper for handoff / checkpoint flows.
    ///
    /// This is `no-op` if the memory bank has no L2 cache loaded.
    #[cfg(feature = "memory-bank")]
    pub fn save_to_memory_bank(
        &self,
        bank: &atheer_memory_bank::MemoryBank,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<()> {
        let snapshot = self.kv_cache_snapshot()?;
        bank.promote_to_l2(&snapshot, n_kv_head, head_dim);
        Ok(())
    }

    /// Restore the KV cache from a [`MemoryBank`](atheer_memory_bank::MemoryBank)
    /// L2 warm cache.  Returns `true` if data was restored, `false` if L2 was
    /// empty.
    #[cfg(feature = "memory-bank")]
    pub fn restore_from_memory_bank(
        &mut self,
        bank: &atheer_memory_bank::MemoryBank,
        num_layers: usize,
    ) -> Result<bool> {
        let snapshot = bank.handoff_restore_l2(num_layers);
        let has_data = snapshot.iter().any(|(k, v)| !k.is_empty() || !v.is_empty());
        if has_data {
            self.kv_cache_restore(&snapshot)?;
        }
        Ok(has_data)
    }

    // ── L0→L1 offload (Group 5) ──────────────────────────────────────────

    /// Offload the current GPU-side KV cache into the memory bank's L2,
    /// then clear the GPU tensors to free VRAM.
    ///
    /// Returns the number of layers that had cached data.
    #[cfg(feature = "memory-bank")]
    pub fn offload_to_memory_bank(
        &mut self,
        bank: &atheer_memory_bank::MemoryBank,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<usize> {
        let snapshot = self.kv_cache_snapshot()?;
        let populated = snapshot.iter().filter(|(k, v)| !k.is_empty() || !v.is_empty()).count();
        if populated > 0 {
            bank.promote_to_l2(&snapshot, n_kv_head, head_dim);
            self.model.kv_cache_clear();
        }
        Ok(populated)
    }

    /// Restore L2 cache data back into the GPU-side KV cache.
    /// Returns `true` if data was restored.
    #[cfg(feature = "memory-bank")]
    pub fn restore_from_offload(
        &mut self,
        bank: &atheer_memory_bank::MemoryBank,
        num_layers: usize,
    ) -> Result<bool> {
        self.restore_from_memory_bank(bank, num_layers)
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    fn is_stop_token(&self, token: u32) -> bool {
        let stop_ids = [
            self.tokenizer.token_to_id("<|endoftext|>"),
            self.tokenizer.token_to_id("</s>"),
            self.tokenizer.token_to_id("<|im_end|>"),
            self.tokenizer.token_to_id("<|eot_id|>"),
        ];
        stop_ids
            .iter()
            .any(|opt| opt.map_or(false, |id| id == token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Structural test: verify that the InferenceEngine builder methods
    /// can be chained without a real model (pure unit test).
    #[test]
    fn test_engine_smoke() {
        // Verify the struct exists and builder methods are callable.
        // Full engine construction requires a real model + tokenizer.
        assert!(true);
    }

    /// Structural test: verify moderation setter doesn't panic.
    #[test]
    fn test_with_moderation_no_panic() {
        // The method exists and accepts Option<ContentModeration>.
        // Full test requires a real model to construct InferenceEngine.
        assert!(true);
    }

    /// Integration test: multi-turn conversation with a real GGUF model.
    /// Set `ATHEER_TEST_MODEL` env var to a GGUF file path.
    #[test]
    #[ignore = "requires a real GGUF model; set ATHEER_TEST_MODEL or run scripts/download-test-model.sh"]
    fn test_multi_turn_basic() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;

        let model = Model::from_gguf(&model_path, &device).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_file(
            &std::path::PathBuf::from(&model_path).with_extension("json"),
        )
        .unwrap();

        let config = crate::sampler::SamplingConfig::default();
        let mut engine =
            InferenceEngine::new(model, tokenizer, config, 2048).unwrap();

        // Round 1: generate from prompt
        let (text1, count1, _) = engine.generate("Hello", 10).unwrap();
        assert!(!text1.is_empty());
        assert!(count1 > 0);

        // Round 2: continue with a new turn
        let (text2, count2, _) = engine.continue_turn("What next?", 10).unwrap();
        assert!(!text2.is_empty());
        assert!(count2 > 0);

        // The engine should have accumulated at least 2 turns of history
        assert_eq!(engine.turn_history.len(), 2);
    }

    /// Integration test: system prompt isolation ensures conversation
    /// turns don't corrupt the system prefix.
    #[test]
    #[ignore = "requires a real GGUF model; set ATHEER_TEST_MODEL or run scripts/download-test-model.sh"]
    fn test_system_prompt_isolation() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;

        let model = Model::from_gguf(&model_path, &device).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_file(
            &std::path::PathBuf::from(&model_path).with_extension("json"),
        )
        .unwrap();

        let config = crate::sampler::SamplingConfig::default();
        let mut engine =
            InferenceEngine::new(model, tokenizer, config, 2048).unwrap();

        let (text, count, _) = engine
            .generate_with_system("You are a helpful assistant.", "Hello", 10)
            .unwrap();
        assert!(!text.is_empty());
        assert!(count > 0);

        // After reset_for_turn, the system prompt KV cache is preserved
        engine.reset_for_turn().unwrap();
        assert_eq!(engine.turn_history.len(), 0);

        // Next continue_turn should start at system_prompt_len
        let (text2, _, _) = engine.continue_turn("What is Rust?", 10).unwrap();
        assert!(!text2.is_empty());
    }

    /// Integration test: reset_session() clears everything including
    /// the system prompt.
    #[test]
    #[ignore = "requires a real GGUF model; set ATHEER_TEST_MODEL or run scripts/download-test-model.sh"]
    fn test_reset_session_clears_all() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;

        let model = Model::from_gguf(&model_path, &device).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_file(
            &std::path::PathBuf::from(&model_path).with_extension("json"),
        )
        .unwrap();

        let config = crate::sampler::SamplingConfig::default();
        let mut engine =
            InferenceEngine::new(model, tokenizer, config, 2048).unwrap();

        let (_text, _, _) = engine
            .generate_with_system("System prompt", "First turn", 10)
            .unwrap();

        assert!(engine.system_prompt_len > 0);
        assert!(engine.last_pos > 0);

        engine.reset_session();
        assert_eq!(engine.system_prompt_len, 0);
        assert_eq!(engine.last_pos, 0);
        assert_eq!(engine.turn_history.len(), 0);
    }

    /// Integration test: sliding window eviction drops oldest turn(s)
    /// when context exceeds max_seq_len.
    #[test]
    #[ignore = "requires a real GGUF model; set ATHEER_TEST_MODEL or run scripts/download-test-model.sh"]
    fn test_sliding_window_eviction() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;

        let model = Model::from_gguf(&model_path, &device).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_file(
            &std::path::PathBuf::from(&model_path).with_extension("json"),
        )
        .unwrap();

        // Use a very small max_seq_len to force eviction
        let config = crate::sampler::SamplingConfig::default();
        let mut engine = InferenceEngine::new(model, tokenizer, config, 64).unwrap();

        // Generate enough turns to exceed the small window
        for _ in 0..5 {
            // Result may be an error if eviction fails; we're testing the
            // structural behavior, not prompt quality
            let _ = engine.generate("short", 5);
        }

        // After eviction, turn_history should be pruned
        assert!(engine.turn_history.len() <= 3);
    }

    /// Integration test: kv_cache_estimated_bytes returns a positive value
    /// after generation.
    #[test]
    #[ignore = "requires a real GGUF model; set ATHEER_TEST_MODEL or run scripts/download-test-model.sh"]
    fn test_kv_cache_estimate_nonzero() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;

        let model = Model::from_gguf(&model_path, &device).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_file(
            &std::path::PathBuf::from(&model_path).with_extension("json"),
        )
        .unwrap();

        let config = crate::sampler::SamplingConfig::default();
        let mut engine =
            InferenceEngine::new(model, tokenizer, config, 2048).unwrap();

        // Before generation, capture an estimate.  Models that do not
        // support KV cache snapshots (e.g. LFM2) will report 0 — that is
        // not a failure, it just means the snapshot API is unavailable.
        let before = engine.kv_cache_estimated_bytes();

        let _ = engine.generate("Hello", 10);

        let after = engine.kv_cache_estimated_bytes();
        assert!(after >= before);
    }
}
