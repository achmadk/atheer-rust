use crate::{
    AtheerConfig, AtheerError, AtheerInferenceMode, EngineStatus, GenerationRequest,
    GenerationResponse,
};
use atheer_accel::BackendManager;
use atheer_core::{CrashReporter, InferenceEngine, SamplingConfig};
use atheer_hardware::{monitor::GenericMonitor, HardwareMonitor};
use atheer_memory_bank::MemoryBank;
use atheer_orchestrator::{Orchestrator, OrchestratorConfig};
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
    orchestrator: Arc<Mutex<Orchestrator>>,
    memory_bank: Arc<Mutex<MemoryBank>>,
    monitor: Arc<dyn HardwareMonitor>,
    crash_reporter: CrashReporter,
    session_id: Arc<Mutex<Option<String>>>,
    // Streaming state
    stream_tokens: Arc<Mutex<Vec<String>>>,
    stream_index: Arc<AtomicUsize>,
    stream_done: Arc<AtomicBool>,
}

#[uniffi::export]
impl AtheerEngine {
    #[uniffi::constructor]
    pub fn new(config: AtheerConfig) -> Self {
        let orch_config = OrchestratorConfig {
            adaptive: config.adaptive,
            ..Default::default()
        };

        // Probe backends — respect configured preference if set
        let backend_manager = match config.backend_type {
            Some(bt) => {
                let mut bm = BackendManager::new();
                bm.set_backend(bt.into());
                bm
            }
            None => BackendManager::new().with_autoselect(),
        };

        Self {
            inference_engine: Arc::new(Mutex::new(None)),
            config: config.clone(),
            backend_manager,
            orchestrator: Arc::new(Mutex::new(Orchestrator::new(orch_config))),
            memory_bank: Arc::new(Mutex::new(MemoryBank::new(
                config.memory_bank_size_mb as usize,
            ))),
            monitor: Arc::new(GenericMonitor::new()),
            crash_reporter: CrashReporter::new(),
            session_id: Arc::new(Mutex::new(None)),
            // Streaming state
            stream_tokens: Arc::new(Mutex::new(Vec::new())),
            stream_index: Arc::new(AtomicUsize::new(0)),
            stream_done: Arc::new(AtomicBool::new(false)),
        }
    }

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
        let model = atheer_core::Model::from_gguf(model_path, &device).map_err(|e| {
            AtheerError::ModelLoadFailed {
                message: format!("{e}"),
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

        let engine = InferenceEngine::new(model, tokenizer, sampling_config, 4096)
            .map_err(|e| AtheerError::ModelLoadFailed {
                message: format!("Device validation: {e}"),
            })?;

        let mut guard = self
            .inference_engine
            .lock()
            .map_err(|_| AtheerError::NotInitialized)?;
        *guard = Some(engine);

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
                    let mut mem = self.memory_bank.lock().unwrap();
                    mem.demote_l1_to_l2_on_pressure(&snapshot, 8, 128, 0.8);
                }
            }
        }

        let (text, tokens_gen, time_ms) = engine
            .generate(&request.prompt, request.max_tokens, None)
            .map_err(|e| AtheerError::GenerationFailed {
                message: format!("{e}"),
            })?;

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
            draft_loaded: false,
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

    pub fn unload_draft(&self) -> std::result::Result<(), AtheerError> {
        Ok(())
    }

    pub fn load_draft(&self, _path: &str) -> std::result::Result<(), AtheerError> {
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
}
