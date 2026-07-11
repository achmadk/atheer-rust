use crate::{
    AtheerConfig, AtheerError, AtheerInferenceMode, EngineStatus, GenerationRequest,
    GenerationResponse,
};
use atheer_accel::BackendManager;
use atheer_core::model_credential::ModelCredential;
use atheer_core::model_encryption::{aes256_gcm::Aes256GcmEncryption, ModelEncryption};
use atheer_core::{CrashReporter, InferenceEngine, SamplingConfig};
use atheer_hardware::{monitor::GenericMonitor, HardwareMonitor};
use atheer_memory_bank::{l3_compressed::L3CompressedStorage, MemoryBank};
use atheer_orchestrator::calibrator::CalibrationSample;
use atheer_orchestrator::{Orchestrator, OrchestratorConfig};
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[allow(dead_code)]
#[derive(uniffi::Object)]
pub struct AtheerEngine {
    config: AtheerConfig,
    backend_manager: BackendManager,
    inference_engine: Arc<Mutex<Option<InferenceEngine>>>,
    draft_engine: Arc<Mutex<Option<InferenceEngine>>>,
    orchestrator: Arc<Mutex<Orchestrator>>,
    memory_bank: Arc<Mutex<MemoryBank>>,
    monitor: Arc<dyn HardwareMonitor>,
    crash_reporter: CrashReporter,
    session_id: Arc<Mutex<Option<String>>>,
    // Encryption scheme registry for Custom credentials
    encryption_schemes: Mutex<HashMap<String, Box<dyn ModelEncryption>>>,
    // Device-derived key support
    device_uid: Mutex<Option<String>>,
    // Streaming state
    stream_tokens: Arc<Mutex<Vec<String>>>,
    stream_index: Arc<AtomicUsize>,
    stream_done: Arc<AtomicBool>,
    // Checkpoint persistence state
    last_checkpoint_uuid: Arc<Mutex<Option<String>>>,
    last_l3_snapshot_id: Arc<Mutex<Option<String>>>,
    l3_storage: Arc<Mutex<Option<L3CompressedStorage>>>,
}

#[uniffi::export]
impl AtheerEngine {
    #[uniffi::constructor]
    pub fn new(config: AtheerConfig) -> Self {
        let orch_config = OrchestratorConfig {
            adaptive: config.adaptive,
            ..Default::default()
        };

        // Probe backends — respect configured preference if set,
        // and wire CoreML model path if provided.
        let backend_manager = {
            let mut bm = match config.backend_type {
                Some(bt) => {
                    let mut b = BackendManager::new();
                    b.set_backend(bt.into());
                    b
                }
                None => BackendManager::new().with_autoselect(),
            };

            // If a CoreML .mlpackage path was provided, pass it through
            // so the ANE backend loads the real model instead of probing.
            if let Some(ref path) = config.coreml_model_path {
                // Extract architecture and param count from config.
                // Default to "llama" architecture and q4_k_m quantization.
                let architecture = config.model_id.as_deref().unwrap_or("llama");
                let quantization = &config.quantization;
                // Derive param count from model_id if possible, default to ~100M
                let param_count_m = parse_param_count(architecture);
                bm = bm.with_coreml_model(path, architecture, quantization, param_count_m);
            }

            bm
        };

        // Initialize L3-compressed storage if checkpoint_dir is configured
        let l3_storage = config
            .checkpoint_dir
            .as_ref()
            .map(|dir| PathBuf::from(dir).join("l3"))
            .and_then(|l3_dir| L3CompressedStorage::new(l3_dir).ok());

        Self {
            inference_engine: Arc::new(Mutex::new(None)),
            draft_engine: Arc::new(Mutex::new(None)),
            config: config.clone(),
            backend_manager,
            orchestrator: Arc::new(Mutex::new(Orchestrator::new(orch_config))),
            memory_bank: Arc::new(Mutex::new(MemoryBank::new(
                config.memory_bank_size_mb as usize,
            ))),
            monitor: Arc::new(GenericMonitor::new()),
            crash_reporter: CrashReporter::new(),
            session_id: Arc::new(Mutex::new(None)),
            encryption_schemes: Mutex::new(HashMap::new()),
            device_uid: Mutex::new(None),
            stream_tokens: Arc::new(Mutex::new(Vec::new())),
            stream_index: Arc::new(AtomicUsize::new(0)),
            stream_done: Arc::new(AtomicBool::new(false)),
            last_checkpoint_uuid: Arc::new(Mutex::new(None)),
            last_l3_snapshot_id: Arc::new(Mutex::new(None)),
            l3_storage: Arc::new(Mutex::new(l3_storage)),
        }
    }
}

/// Parse a parameter count from a model ID string.
///
/// Heuristic: looks for patterns like "700M", "1.5B", "7B" in the model ID.
/// Returns the count in millions (e.g., "1.5B" → 1500.0).
/// Defaults to ~100M if no pattern is found — a conservative default for
/// ANE compatibility heuristics.
fn parse_param_count(model_id: &str) -> f32 {
    let lower = model_id.to_lowercase();
    if let Some(end) = lower.rfind('b') {
        let prefix = &lower[..end].trim_end();
        if let Some(start) = prefix.rfind(|c: char| !c.is_ascii_digit() && c != '.') {
            if let Ok(v) = prefix[start + 1..].parse::<f32>() {
                return v * 1000.0;
            }
        } else if let Ok(v) = prefix.parse::<f32>() {
            return v * 1000.0;
        }
    }
    if let Some(end) = lower.rfind('m') {
        let prefix = &lower[..end].trim_end();
        if let Some(start) = prefix.rfind(|c: char| !c.is_ascii_digit() && c != '.') {
            if let Ok(v) = prefix[start + 1..].parse::<f32>() {
                return v;
            }
        } else if let Ok(v) = prefix.parse::<f32>() {
            return v;
        }
    }
    100.0 // fallback
}

#[uniffi::export]
impl AtheerEngine {
    pub fn initialize(&self) -> std::result::Result<(), AtheerError> {
        let model_path = self
            .config
            .model_path
            .as_ref()
            .ok_or(AtheerError::NotInitialized)?;
        let tokenizer_path = self
            .config
            .tokenizer_path
            .as_deref()
            .unwrap_or("tokenizer.json");

        let device = self.backend_manager.device();
        let cpu_device = candle_core::Device::Cpu;

        let model = {
            let try_load = |dev: &candle_core::Device| -> std::result::Result<atheer_core::Model, AtheerError> {
                if let Some(ref credential) = self.config.model_credential {
                    let bytes = self.decrypt_with_credential(credential, model_path)?;
                    let mut cursor = std::io::Cursor::new(bytes);
                    atheer_core::Model::from_gguf_reader(&mut cursor, dev).map_err(|e| {
                        AtheerError::ModelLoadFailed {
                            message: format!("{e}"),
                        }
                    })
                } else {
                    atheer_core::Model::from_gguf(model_path, dev).map_err(|e| {
                        AtheerError::ModelLoadFailed {
                            message: format!("{e}"),
                        }
                    })
                }
            };

            match try_load(&device) {
                Ok(m) => m,
                Err(first_err) => {
                    tracing::warn!(
                        target: "atheer::engine",
                        "Model load failed on {device:?}: {first_err}. Retrying on CPU."
                    );
                    match try_load(&cpu_device) {
                        Ok(m) => {
                            tracing::info!(
                                target: "atheer::engine",
                                "Model loaded on CPU (degraded mode — inference will be slower)"
                            );
                            m
                        }
                        Err(second_err) => {
                            return Err(AtheerError::ModelLoadFailed {
                                message: format!(
                                    "All device attempts failed. \
                                     Preferred ({device:?}): {first_err}. \
                                     CPU fallback: {second_err}"
                                ),
                            })
                        }
                    }
                }
            }
        };

        let tokenizer = atheer_core::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            AtheerError::TokenizerLoadFailed {
                message: format!("{e}"),
            }
        })?;

        let sampling_config = SamplingConfig {
            temperature: self.config.temperature as f64,
            ..Default::default()
        };

        let mut engine =
            InferenceEngine::new(model, tokenizer, sampling_config, 4096).map_err(|e| {
                AtheerError::ModelLoadFailed {
                    message: format!("Device validation: {e}"),
                }
            })?;

        // Wire checkpoint directory and model-id for lifecycle persistence
        if let Some(ref checkpoint_dir) = self.config.checkpoint_dir {
            engine.with_checkpoint_dir(std::path::PathBuf::from(checkpoint_dir));
            if let Some(ref model_id) = self.config.model_id {
                engine.with_model_id(model_id);
            }
        }

        {
            let mut guard = self
                .inference_engine
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            *guard = Some(engine);
        }

        // Auto-load draft model if standby_draft_path is configured
        if let Some(ref draft_path) = self.config.standby_draft_path {
            if !draft_path.is_empty() {
                tracing::info!(
                    target: "atheer::engine",
                    "Auto-loading draft model from standby_draft_path: {draft_path}"
                );
                // Ignore error — engine is usable without a draft model
                let _ = self.load_draft(draft_path);
            }
        }

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.inference_engine
            .lock()
            .ok()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    pub fn generate_sync(
        &self,
        request: &GenerationRequest,
    ) -> std::result::Result<GenerationResponse, AtheerError> {
        let mut guard = self
            .inference_engine
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        let engine = guard.as_mut().ok_or(AtheerError::NotInitialized)?;

        // Thaw L3 snapshot if one exists and KV cache is empty
        if let Ok(l3_id_guard) = self.last_l3_snapshot_id.lock() {
            if let Some(ref snapshot_id) = *l3_id_guard {
                // Check if KV cache is empty before attempting restore
                let snapshot = engine.kv_cache_snapshot().unwrap_or_default();
                let is_empty = snapshot.iter().all(|(k, v)| k.is_empty() && v.is_empty());
                if is_empty {
                    // Attempt L3 restore — best-effort, log failure and continue
                    match self.thaw_l3_snapshot(engine, snapshot_id) {
                        Ok(true) => {
                            tracing::info!(
                                target: "atheer::engine::lifecycle",
                                "generate_sync: L3 snapshot thawed {snapshot_id}"
                            );
                            // Clear the L3 snapshot ID so we don't thaw again
                            drop(l3_id_guard);
                            if let Ok(mut last) = self.last_l3_snapshot_id.lock() {
                                *last = None;
                            }
                        }
                        Ok(false) => {
                            tracing::info!(
                                target: "atheer::engine::lifecycle",
                                "generate_sync: no L3 data to thaw"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "atheer::engine::lifecycle",
                                "generate_sync: L3 thaw failed: {e}"
                            );
                            // Clear stale snapshot ID on failure
                            drop(l3_id_guard);
                            if let Ok(mut last) = self.last_l3_snapshot_id.lock() {
                                *last = None;
                            }
                        }
                    }
                }
            }
        }

        // Apply sampling configuration based on request
        let sampling_config = SamplingConfig {
            temperature: request.temperature as f64,
            ..Default::default()
        };

        let base_sampler = Box::new(atheer_core::sampler::DefaultSampler::new(sampling_config));

        // If JSON schema is requested, we apply the JSON grammar constraint
        if request.json_schema.is_some() {
            let grammar = atheer_orchestrator::JsonGrammar::new();
            let tokenizer_clone = engine.tokenizer().clone_inner();
            let grammar_sampler = Box::new(atheer_orchestrator::GrammarSampler::new(
                base_sampler,
                grammar,
                tokenizer_clone,
            ));
            engine.with_sampler(grammar_sampler);
        } else {
            engine.with_sampler(base_sampler);
        }

        let health = self.monitor.health();
        let mut orch = self
            .orchestrator
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;

        let mode = orch.select_mode(
            None, // thermal_c — would come from thermal headroom conversion
            health.available_ram_mb,
            Some(health.battery_level),
            health.on_battery,
        );

        // Check and relieve memory pressure before generation
        {
            let memory = self.memory_bank.lock().unwrap();
            if orch.check_memory_pressure(&memory) {
                orch.log_memory_pressure_if_needed(&memory);
                drop(memory);
                // Try to relieve pressure - demote L1 to L2
                let snapshot = engine.kv_cache_snapshot().unwrap_or_default();
                if !snapshot.is_empty() {
                    let mem = self.memory_bank.lock().unwrap();
                    mem.demote_l1_to_l2_on_pressure(&snapshot, 8, 128, 0.8);
                }
            }
        }

        let use_speculation =
            orch.is_draft_loaded() && orch.speculation_depth() > 0 && request.json_schema.is_none();
        // JSON-schema-constrained output currently uses grammar sampling which
        // is incompatible with draft-model speculation.

        if use_speculation {
            let mut draft_guard = self
                .draft_engine
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            let draft_engine = draft_guard.as_mut().ok_or(AtheerError::NotInitialized)?;
            let spec_depth = orch.speculation_depth();

            let accepted_tokens = std::sync::atomic::AtomicUsize::new(0);
            let total_draft = std::sync::atomic::AtomicUsize::new(0);

            let (text, tokens_gen, time_ms) = engine
                .generate_speculative(
                    &request.prompt,
                    request.max_tokens,
                    draft_engine,
                    spec_depth,
                    None,
                    |accepted: usize, total: usize| {
                        accepted_tokens.store(accepted, std::sync::atomic::Ordering::Relaxed);
                        total_draft.store(total, std::sync::atomic::Ordering::Relaxed);
                    },
                )
                .map_err(|e| AtheerError::GenerationFailed {
                    message: format!("{e}"),
                })?;

            let acc = accepted_tokens.load(std::sync::atomic::Ordering::Relaxed);
            let tot = total_draft.load(std::sync::atomic::Ordering::Relaxed);
            orch.record_speculative_result(acc, tot);

            // Feed generation metrics for calibration (task 4.2)
            let tok_s = compute_tok_s(tokens_gen, time_ms);
            let acceptance_rate = if tot > 0 {
                Some(acc as f32 / tot as f32)
            } else {
                None
            };
            orch.record_generation_metrics(CalibrationSample {
                tok_s,
                tokens_gen,
                mode,
                speculation_depth: spec_depth,
                acceptance_rate,
            });

            return Ok(GenerationResponse::new(
                text,
                tokens_gen,
                time_ms,
                mode.as_str(),
            ));
        }

        let (text, tokens_gen, time_ms) = engine
            .generate(&request.prompt, request.max_tokens, None)
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("{e}"),
            })?;

        // Feed generation metrics for calibration (task 4.1)
        let tok_s = compute_tok_s(tokens_gen, time_ms);
        orch.record_generation_metrics(CalibrationSample {
            tok_s,
            tokens_gen,
            mode,
            speculation_depth: 0,
            acceptance_rate: None,
        });

        Ok(GenerationResponse::new(
            text,
            tokens_gen,
            time_ms,
            mode.as_str(),
        ))
    }

    pub fn status(&self) -> EngineStatus {
        let health = self.monitor.health();
        let orch = self.orchestrator.lock().unwrap();
        let memory = self.memory_bank.lock().unwrap();

        EngineStatus {
            mode: orch.current_mode().as_str().to_string(),
            tokens_per_second: 0.0,
            draft_loaded: orch.is_draft_loaded(),
            hardware_health: crate::status::HardwareHealth {
                thermal: health.thermal.as_str().to_string(),
                available_ram_mb: health.available_ram_mb,
                battery_level: health.battery_level,
                on_battery: health.on_battery,
            },
            memory_bank: crate::status::MemoryBankStatus {
                l1_active: memory.l1_active(),
                l2_warm: memory.l2_warm(),
                alignment_score: memory.alignment_score(),
                is_handoff: memory.handoff_phase() != atheer_memory_bank::HandoffPhase::Idle,
                handoff_phase: format!("{:?}", memory.handoff_phase()).to_lowercase(),
            },
        }
    }

    pub fn set_mode(&self, mode: AtheerInferenceMode) -> std::result::Result<(), AtheerError> {
        let mut orch = self
            .orchestrator
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        orch.set_mode(mode.into());
        Ok(())
    }

    pub fn load_draft(&self, path: &str) -> std::result::Result<(), AtheerError> {
        if path.is_empty() {
            return Err(AtheerError::NotInitialized);
        }

        let device = self.backend_manager.device();
        let tokenizer_path = self
            .config
            .tokenizer_path
            .as_deref()
            .unwrap_or("tokenizer.json");

        let model = atheer_core::Model::from_gguf(path, &device).map_err(|e| {
            AtheerError::ModelLoadFailed {
                message: format!("Failed to load draft model: {e}"),
            }
        })?;

        let tokenizer = atheer_core::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            AtheerError::TokenizerLoadFailed {
                message: format!("{e}"),
            }
        })?;

        let sampling_config = SamplingConfig {
            temperature: self.config.temperature as f64,
            ..Default::default()
        };

        let engine =
            InferenceEngine::new(model, tokenizer, sampling_config, 4096).map_err(|e| {
                AtheerError::ModelLoadFailed {
                    message: format!("Draft engine init: {e}"),
                }
            })?;

        {
            let mut guard = self
                .draft_engine
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            *guard = Some(engine);
        }

        {
            let mut orch = self
                .orchestrator
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            orch.set_draft_model_loaded(true);
        }

        tracing::info!(target: "atheer::engine", "Draft model loaded from {path}");
        Ok(())
    }

    pub fn unload_draft(&self) -> std::result::Result<(), AtheerError> {
        {
            let mut guard = self
                .draft_engine
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            *guard = None;
        }

        {
            let mut orch = self
                .orchestrator
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            orch.set_draft_model_loaded(false);
        }

        tracing::info!(target: "atheer::engine", "Draft model unloaded");
        Ok(())
    }

    pub fn crash_log_path(&self) -> Option<String> {
        self.crash_reporter
            .crash_log_path()
            .map(|p| p.to_string_lossy().to_string())
    }

    pub fn generate_stream(&self, request: &GenerationRequest) -> Result<(), AtheerError> {
        if !self.is_initialized() {
            return Err(AtheerError::NotInitialized);
        }
        // Reset streaming state
        {
            let mut tokens = self
                .stream_tokens
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            tokens.clear();
        }
        self.stream_index.store(0, Ordering::Relaxed);
        self.stream_done.store(false, Ordering::Relaxed);

        let tokens_clone = self.stream_tokens.clone();
        let done_clone = self.stream_done.clone();
        let prompt = request.prompt.clone();
        let max_tokens = request.max_tokens;

        thread::spawn(move || {
            // Token generation (placeholder: split prompt)
            let tokens: Vec<String> = prompt
                .split_whitespace()
                .take(max_tokens as usize)
                .map(|w| format!(" {}", w))
                .collect();

            eprintln!("[stream] generated {} tokens", tokens.len());

            for token in tokens {
                // Simulate generation time per token
                thread::sleep(Duration::from_millis(50));
                let mut guard = tokens_clone.lock().unwrap();
                guard.push(token);
            }
            done_clone.store(true, Ordering::Relaxed);
            eprintln!("[stream] done set");
        });

        Ok(())
    }

    pub fn poll_stream_token(&self) -> Option<String> {
        let idx = self.stream_index.load(Ordering::Relaxed);
        let tokens = match self.stream_tokens.lock() {
            Ok(t) => t,
            Err(_) => return None,
        };
        if idx < tokens.len() {
            let token = tokens[idx].clone();
            self.stream_index.fetch_add(1, Ordering::Relaxed);
            Some(token)
        } else {
            None
        }
    }

    pub fn stream_done(&self) -> bool {
        self.stream_done.load(Ordering::Relaxed)
    }

    // ── Lifecycle hooks (called from Swift/Kotlin via UniFFI) ─────

    /// Save a full KV cache checkpoint to disk on app background.
    /// Best-effort: logs errors, never panics.
    pub fn on_background(&self) {
        tracing::info!(target: "atheer::engine::lifecycle", "on_background");
        if self.config.checkpoint_dir.is_none() {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_background: no checkpoint_dir, skipping");
            return;
        }
        if !self.config.checkpoint_on_background {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_background: disabled by config");
            return;
        }

        match self.save_checkpoint_inner() {
            Ok(uuid) => {
                tracing::info!(target: "atheer::engine::lifecycle", "on_background: checkpoint saved {uuid}");
                self.run_checkpoint_cleanup();
            }
            Err(e) => {
                tracing::error!(target: "atheer::engine::lifecycle", "on_background: checkpoint failed: {e}");
            }
        }
    }

    /// Restore KV cache from the latest checkpoint on app foreground.
    /// Returns `true` if a valid checkpoint was restored.
    pub fn on_foreground(&self) -> bool {
        tracing::info!(target: "atheer::engine::lifecycle", "on_foreground");
        if self.config.checkpoint_dir.is_none() {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_foreground: no checkpoint_dir");
            return false;
        }
        if !self.config.restore_on_foreground {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_foreground: disabled by config");
            return false;
        }

        match self.load_checkpoint_inner() {
            Ok(true) => {
                tracing::info!(target: "atheer::engine::lifecycle", "on_foreground: checkpoint restored");
                true
            }
            Ok(false) => {
                tracing::info!(target: "atheer::engine::lifecycle", "on_foreground: no checkpoint to restore");
                false
            }
            Err(e) => {
                tracing::warn!(target: "atheer::engine::lifecycle", "on_foreground: checkpoint restore failed: {e}");
                false
            }
        }
    }

    /// Respond to low-memory pressure by saving an LZ4-compressed L3 snapshot
    /// and optionally clearing GPU-side KV cache.
    /// Best-effort: never panics, never clears cache if save failed.
    pub fn on_low_memory(&self) {
        tracing::info!(target: "atheer::engine::lifecycle", "on_low_memory");
        if self.config.checkpoint_dir.is_none() {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_low_memory: no checkpoint_dir");
            return;
        }
        if !self.config.checkpoint_on_low_memory {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_low_memory: disabled by config");
            return;
        }

        let snapshot_id = match self.save_l3_snapshot_inner() {
            Ok(id) => id,
            Err(e) => {
                tracing::error!(target: "atheer::engine::lifecycle", "on_low_memory: L3 snapshot failed: {e}");
                return;
            }
        };

        tracing::info!(target: "atheer::engine::lifecycle", "on_low_memory: L3 snapshot saved {snapshot_id}");

        if self.config.clear_on_low_memory {
            if let Ok(mut guard) = self.inference_engine.lock() {
                if let Some(engine) = guard.as_mut() {
                    engine.kv_cache_clear();
                    tracing::info!(target: "atheer::engine::lifecycle", "on_low_memory: KV cache cleared");
                }
            }
        }
    }

    /// Flush any pending checkpoint on app termination.
    /// Best-effort: never panics.
    pub fn on_terminate(&self) {
        tracing::info!(target: "atheer::engine::lifecycle", "on_terminate");
        if self.config.checkpoint_dir.is_none() {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_terminate: no checkpoint_dir");
            return;
        }
        if !self.config.checkpoint_on_terminate {
            tracing::debug!(target: "atheer::engine::lifecycle", "on_terminate: disabled by config");
            return;
        }

        let has_saved = self
            .last_checkpoint_uuid
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .is_some();

        if !has_saved {
            match self.save_checkpoint_inner() {
                Ok(uuid) => {
                    tracing::info!(target: "atheer::engine::lifecycle", "on_terminate: final checkpoint saved {uuid}");
                }
                Err(e) => {
                    tracing::error!(target: "atheer::engine::lifecycle", "on_terminate: final checkpoint failed: {e}");
                }
            }
        }

        self.run_checkpoint_cleanup();
    }

    /// Check if a valid checkpoint exists and can be restored.
    /// Used by mobile apps to decide whether to show a cold-start screen.
    pub fn has_checkpoint(&self) -> bool {
        if self.config.checkpoint_dir.is_none() {
            return false;
        }
        // Check memory first
        if let Ok(guard) = self.last_checkpoint_uuid.lock() {
            if guard.is_some() {
                return true;
            }
        }
        // Fall back to sidecar
        self.read_latest_checkpoint_sidecar().is_some()
    }
}

// ── Private helpers ──────────────────────────────────────────────
impl AtheerEngine {
    // ── Checkpoint persistence helpers ────────────────────────────

    /// Save checkpoint via the inference engine and write sidecar.
    fn save_checkpoint_inner(&self) -> std::result::Result<String, AtheerError> {
        let mut guard = self
            .inference_engine
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        let engine = guard.as_mut().ok_or(AtheerError::NotInitialized)?;
        let uuid = engine
            .save_checkpoint()
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("checkpoint save failed: {e}"),
            })?;

        // Store in memory
        if let Ok(mut cp) = self.last_checkpoint_uuid.lock() {
            *cp = Some(uuid.clone());
        }

        // Write sidecar atomically
        self.write_latest_checkpoint_sidecar(&uuid);

        Ok(uuid)
    }

    /// Restore checkpoint — tries in-memory UUID first, then sidecar.
    fn load_checkpoint_inner(&self) -> std::result::Result<bool, AtheerError> {
        let uuid = {
            let guard = self
                .last_checkpoint_uuid
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            guard.clone()
        };
        let uuid = match uuid {
            Some(u) => u,
            None => match self.read_latest_checkpoint_sidecar() {
                Some(u) => u,
                None => return Ok(false),
            },
        };

        let mut guard = self
            .inference_engine
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        let engine = guard.as_mut().ok_or(AtheerError::NotInitialized)?;

        engine
            .load_checkpoint(&uuid)
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("checkpoint load failed: {e}"),
            })?;

        // Update in-memory UUID
        if let Ok(mut cp) = self.last_checkpoint_uuid.lock() {
            *cp = Some(uuid);
        }

        Ok(true)
    }

    /// Save an LZ4-compressed L3 snapshot from the current KV cache.
    fn save_l3_snapshot_inner(&self) -> std::result::Result<String, AtheerError> {
        let mut guard = self
            .inference_engine
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        let engine = guard.as_mut().ok_or(AtheerError::NotInitialized)?;

        let snapshot = engine
            .kv_cache_snapshot()
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("L3 snapshot: kv_cache_snapshot failed: {e}"),
            })?;

        // Serialize into the same binary format as a checkpoint (layer-prefixed)
        let mut buf = Vec::new();
        for (keys, values) in &snapshot {
            buf.extend_from_slice(&(keys.len() as u64).to_le_bytes());
            for v in keys {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        let model_id = self.config.model_id.as_deref().unwrap_or("unknown");
        let snapshot_id = {
            let mut storage_guard = self
                .l3_storage
                .lock()
                .map_err(|_| AtheerError::NotInitialized)?;
            let storage = storage_guard
                .as_mut()
                .ok_or_else(|| AtheerError::GenerationFailed {
                    message: "L3 storage not initialized".to_string(),
                })?;
            storage
                .snapshot(model_id, &buf)
                .map_err(|e| AtheerError::GenerationFailed {
                    message: format!("L3 snapshot failed: {e}"),
                })?
        };

        // Store in memory
        if let Ok(mut last) = self.last_l3_snapshot_id.lock() {
            *last = Some(snapshot_id.clone());
        }

        Ok(snapshot_id)
    }

    // ── Sidecar helpers ───────────────────────────────────────────

    /// Write `latest_checkpoint.txt` sidecar atomically (temp → rename).
    fn write_latest_checkpoint_sidecar(&self, uuid: &str) {
        let Some(ref dir) = self.config.checkpoint_dir else {
            return;
        };
        let sidecar_path = PathBuf::from(dir).join("latest_checkpoint.txt");
        let tmp_path = PathBuf::from(dir).join(".latest_checkpoint.tmp");
        if let Err(e) = (|| -> std::io::Result<()> {
            fs::write(&tmp_path, uuid)?;
            fs::rename(&tmp_path, &sidecar_path)?;
            Ok(())
        })() {
            tracing::warn!(
                target: "atheer::engine::lifecycle",
                "Failed to write sidecar: {e}"
            );
        }
    }

    /// Read the UUID from `latest_checkpoint.txt` sidecar.
    fn read_latest_checkpoint_sidecar(&self) -> Option<String> {
        let dir = self.config.checkpoint_dir.as_ref()?;
        let path = PathBuf::from(dir).join("latest_checkpoint.txt");
        let content = fs::read_to_string(path).ok()?;
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    // ── Generational cleanup ──────────────────────────────────────

    /// List checkpoint files for the current model, sort by creation time,
    /// keep at most `max_checkpoints`, delete the rest.
    fn run_checkpoint_cleanup(&self) {
        let Some(ref dir) = self.config.checkpoint_dir else {
            return;
        };
        let max_cp = self.config.max_checkpoints as usize;
        let ttl_hours = self.config.checkpoint_ttl_hours;
        let model_id = self.config.model_id.as_deref().unwrap_or("unknown");

        let entries = match fs::read_dir(dir) {
            Ok(e) => e.filter_map(|e| e.ok()).collect::<Vec<_>>(),
            Err(e) => {
                tracing::warn!(target: "atheer::engine::lifecycle", "cleanup: cannot read dir: {e}");
                return;
            }
        };

        // Collect (uuid, created_at_timestamp) pairs for this model
        let mut checkpoints: Vec<(String, i64)> = Vec::new();
        for entry in &entries {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(uuid) = name
                .strip_prefix("checkpoint_")
                .and_then(|n| n.strip_suffix(".meta"))
            {
                let meta_path = entry.path();
                let meta_content = match fs::read_to_string(&meta_path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let meta: serde_json::Value = match serde_json::from_str(&meta_content) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Filter by model_id
                let stored_model = meta.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
                if stored_model != model_id {
                    continue;
                }
                // Use file modification time as the sort key for cleanup ordering
                let mtime = meta_path
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                checkpoints.push((uuid.to_string(), mtime));
            }
        }

        // Sort by creation time (oldest first)
        checkpoints.sort_by_key(|(_, ts)| *ts);

        // TTL-based expiry
        if ttl_hours > 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            let cutoff = now - (ttl_hours as i64 * 3600);
            let expired: Vec<_> = checkpoints
                .iter()
                .filter(|(_, ts)| *ts < cutoff)
                .map(|(uuid, _)| uuid.clone())
                .collect();
            for uuid in &expired {
                delete_checkpoint_files(dir, uuid);
            }
            checkpoints.retain(|(uuid, _)| !expired.contains(uuid));
        }

        // Count-based pruning (keep max max_cp, sorted oldest-first → delete from front)
        if checkpoints.len() > max_cp {
            let to_delete = checkpoints.len() - max_cp;
            for (uuid, _) in checkpoints.iter().take(to_delete) {
                delete_checkpoint_files(dir, uuid);
            }
        }

        // Delete orphaned L3 snapshots that don't correspond to any retained checkpoint
        let retained_uuids: std::collections::HashSet<String> =
            checkpoints.iter().map(|(u, _)| u.clone()).collect();
        let l3_dir = PathBuf::from(dir).join("l3");
        if let Ok(l3_entries) = fs::read_dir(&l3_dir) {
            for entry in l3_entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&format!("{}_", model_id)) && name.ends_with(".snap") {
                    let prefix = format!("{}_", model_id);
                    let snap_id = name[prefix.len()..].trim_end_matches(".snap").to_string();
                    if !retained_uuids.contains(&snap_id) {
                        fs::remove_file(entry.path()).ok();
                    }
                }
            }
        }
    }

    /// Restore KV cache from an L3 compressed snapshot.
    /// Returns `Ok(true)` on restore, `Ok(false)` if nothing to restore,
    /// `Err` on failure (corrupt/expired snapshot).
    fn thaw_l3_snapshot(
        &self,
        engine: &mut InferenceEngine,
        snapshot_id: &str,
    ) -> std::result::Result<bool, AtheerError> {
        let storage_guard = self
            .l3_storage
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        let storage = storage_guard
            .as_ref()
            .ok_or_else(|| AtheerError::GenerationFailed {
                message: "L3 storage not initialized".to_string(),
            })?;

        let bytes = storage
            .restore(snapshot_id)
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("L3 restore failed: {e}"),
            })?;

        if bytes.is_empty() {
            return Ok(false);
        }

        // Deserialize the binary snapshot back into per-layer (Vec<f32>, Vec<f32>)
        let mut snapshot: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut offset = 0usize;
        while offset < bytes.len() {
            if offset + 8 > bytes.len() {
                break;
            }
            let key_len =
                u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            let key_end = offset + key_len * 4;
            if key_end > bytes.len() {
                break;
            }
            let keys: Vec<f32> = bytes[offset..key_end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            offset = key_end;

            if offset + 8 > bytes.len() {
                break;
            }
            let value_len =
                u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            let value_end = offset + value_len * 4;
            if value_end > bytes.len() {
                break;
            }
            let values: Vec<f32> = bytes[offset..value_end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            offset = value_end;

            snapshot.push((keys, values));
        }

        if snapshot.is_empty() {
            return Ok(false);
        }

        engine
            .kv_cache_restore(&snapshot)
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("L3 thaw: kv_cache_restore failed: {e}"),
            })?;

        Ok(true)
    }

    /// Select and run the decryption pipeline for the given credential.
    ///
    /// Handles key resolution (ServerDistributed via wrapped_key,
    /// DeviceDerived via HKDF, Custom via registered schemes),
    /// catch_unwind protection, and crash reporter scrubbing.
    fn decrypt_with_credential(
        &self,
        credential: &ModelCredential,
        model_path: &str,
    ) -> std::result::Result<Vec<u8>, AtheerError> {
        match credential {
            ModelCredential::ServerDistributed {
                key_id,
                nonce: _,
                wrapped_key,
            } => {
                let key_bytes =
                    wrapped_key
                        .as_ref()
                        .ok_or_else(|| AtheerError::ModelDecryptionFailed {
                            message: format!(
                                "ServerDistributed key '{key_id}': key not provided. \
                             Resolve from Keychain/Keystore and pass as wrapped_key."
                            ),
                        })?;
                if key_bytes.len() != 32 {
                    return Err(AtheerError::ModelDecryptionFailed {
                        message: format!(
                            "ServerDistributed key for {key_id}: expected 32 bytes, got {}",
                            key_bytes.len()
                        ),
                    });
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(key_bytes);
                let enc = Aes256GcmEncryption::new(arr);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    enc.decrypt_reader(model_path)
                }));
                match result {
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(e)) => {
                        enc.scrub();
                        self.record_scrubbed_crash("ModelDecryptFailed", model_path);
                        Err(AtheerError::ModelDecryptionFailed {
                            message: format!("{e}"),
                        })
                    }
                    Err(panic) => {
                        enc.scrub();
                        let msg = extract_panic_msg(&panic);
                        self.record_scrubbed_crash("ModelDecryptPanic", "");
                        Err(AtheerError::ModelDecryptionFailed {
                            message: format!("decrypt panicked: {msg}"),
                        })
                    }
                }
            }
            ModelCredential::DeviceDerived { salt, nonce: _ } => {
                let uid = self
                    .device_uid
                    .lock()
                    .map_err(|_| AtheerError::ModelDecryptionFailed {
                        message: "device_uid lock poisoned".into(),
                    })?
                    .clone()
                    .ok_or_else(|| AtheerError::ModelDecryptionFailed {
                        message: "DeviceDerived: device_uid not set. Call set_device_uid() first."
                            .into(),
                    })?;

                let mut hasher = Sha256::new();
                hasher.update(model_path.as_bytes());
                let model_hash = hasher.finalize();

                let prk = Hkdf::<Sha256>::new(None, uid.as_bytes());
                let mut key = [0u8; 32];
                let info = [model_hash.as_slice(), salt.as_slice()].concat();
                prk.expand(&info, &mut key)
                    .map_err(|e| AtheerError::ModelDecryptionFailed {
                        message: format!("DeviceDerived HKDF expand failed: {e}"),
                    })?;

                let enc = Aes256GcmEncryption::new(key);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    enc.decrypt_reader(model_path)
                }));
                match result {
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(e)) => {
                        enc.scrub();
                        Err(AtheerError::ModelDecryptionFailed {
                            message: format!("{e}"),
                        })
                    }
                    Err(panic) => {
                        enc.scrub();
                        let msg = extract_panic_msg(&panic);
                        Err(AtheerError::ModelDecryptionFailed {
                            message: format!("decrypt panicked: {msg}"),
                        })
                    }
                }
            }
            ModelCredential::Custom {
                scheme_name,
                config: _,
            } => {
                let schemes = self.encryption_schemes.lock().map_err(|_| {
                    AtheerError::ModelDecryptionFailed {
                        message: "encryption_schemes lock poisoned".into(),
                    }
                })?;
                let scheme =
                    schemes
                        .get(scheme_name)
                        .ok_or_else(|| AtheerError::ModelDecryptionFailed {
                            message: format!(
                                "Custom encryption scheme '{scheme_name}' not registered. \
                             Call register_encryption_scheme() before initialize()."
                            ),
                        })?;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    scheme.decrypt_reader(model_path)
                }));
                match result {
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(e)) => Err(AtheerError::ModelDecryptionFailed {
                        message: format!("{e}"),
                    }),
                    Err(panic) => {
                        let msg = extract_panic_msg(&panic);
                        Err(AtheerError::ModelDecryptionFailed {
                            message: format!("Custom scheme '{scheme_name}' panicked: {msg}"),
                        })
                    }
                }
            }
        }
    }

    /// Register a custom encryption scheme for `Custom` credentials.
    /// Rust-only (not UniFFI-exported because `Box<dyn ModelEncryption>` cannot
    /// cross the FFI boundary directly).
    pub fn register_encryption_scheme(
        &self,
        scheme_name: String,
        scheme: Box<dyn ModelEncryption>,
    ) {
        if let Ok(mut schemes) = self.encryption_schemes.lock() {
            schemes.insert(scheme_name, scheme);
        }
    }
} // end #[uniffi::export] impl AtheerEngine

// Non-UniFFI public methods (used from Rust)
impl AtheerEngine {
    /// Set the device UID used for `DeviceDerived` key derivation.
    pub fn set_device_uid(&self, uid: String) {
        if let Ok(mut guard) = self.device_uid.lock() {
            *guard = Some(uid);
        }
    }
}

impl AtheerEngine {
    fn record_scrubbed_crash(&self, error: &str, key_id_to_redact: &str) {
        self.crash_reporter
            .record_crash_scrubbed(error, "", key_id_to_redact);
    }
}

/// Delete checkpoint `.bin` and `.meta` files for a given UUID.
/// Best-effort: logs errors but does not fail the caller.
fn delete_checkpoint_files(dir: &str, uuid: &str) {
    for ext in &["bin", "meta"] {
        let path = PathBuf::from(dir).join(format!("checkpoint_{}.{}", uuid, ext));
        if path.exists() {
            if let Err(e) = fs::remove_file(&path) {
                tracing::warn!(
                    target: "atheer::engine::lifecycle",
                    "Failed to delete checkpoint file {:?}: {e}", path
                );
            }
        }
    }
}

fn extract_panic_msg(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Compute tokens-per-second from generation results.
/// Returns 0.0 when time_ms is zero (avoids division by zero).
fn compute_tok_s(tokens_gen: u32, time_ms: u64) -> f32 {
    if time_ms == 0 {
        return 0.0;
    }
    (tokens_gen as f32) / (time_ms as f32 / 1000.0)
}
