//! llm-worker-rbee - Candle-based Llama-2 inference worker daemon
//!
//! This worker implements Llama-2 inference using a hybrid approach:
//! - Pure ndarray for CPU (checkpoint validation)
//! - Candle kernels for CUDA acceleration (optional)
//!
//! Architecture follows `CANDLE_INTEGRATION_HANDOFF.md`:
//! - Use Candle's kernels, NOT the framework
//! - Keep checkpoint-driven validation
//! - Maintain educational value
//!
//! Created by: TEAM-000 (Foundation)
//! Modified by: TEAM-088 (added comprehensive error narration)

use clap::Parser;
use llm_worker_rbee::{
    backend::CandleInferenceBackend,
    create_router,
    narration::{
        ACTION_MODEL_LOAD, ACTION_STARTUP, ACTOR_LLM_WORKER_RBEE,
        ACTOR_MODEL_LOADER,
    },
    HttpServer,
};
use observability_narration_core::{narrate, NarrationFields};
use std::net::SocketAddr;
use std::sync::Arc;

/// CLI arguments for worker daemon
///
/// These are provided by pool-managerd when spawning the worker.
/// This is NOT a user-facing CLI - it's for orchestration.
#[derive(Parser, Debug)]
#[command(name = "llm-worker-rbee")]
#[command(about = "Candle-based Llama-2 worker daemon for llama-orch")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF format)
    #[arg(long)]
    model: String,

    /// Model reference (e.g., "hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    #[arg(long)]
    model_ref: String,

    /// Backend (e.g., "cpu", "cuda", "metal")
    #[arg(long)]
    backend: String,

    /// Device ID
    #[arg(long)]
    device: u32,

    /// HTTP server port - assigned by pool-managerd
    #[arg(long)]
    port: u16,

    /// Hive URL - where to send heartbeats
    #[arg(long)]
    hive_url: String,
    
    /// Local development mode (no auth, binds to 127.0.0.1 only)
    #[arg(long, default_value = "false")]
    local_mode: bool,
}

/// Main entry point
///
/// CRITICAL: Uses single-threaded tokio runtime for SPEED
/// - flavor = "`current_thread`" ensures NO thread pool
/// - All async operations run on ONE thread
/// - No context switching overhead
/// - Optimal for CPU-bound inference
///
/// Flow:
/// 1. Parse args (from pool-managerd)
/// 2. Load model to memory
/// 3. Start HTTP server
/// 4. Start heartbeat task
/// 5. Run forever (until killed by pool-managerd)
#[tokio::main(flavor = "current_thread")] // CRITICAL: Single-threaded!
async fn main() -> anyhow::Result<()> {
    // TEAM-088: Initialize tracing with human-friendly format for development
    // Use LLORCH_LOG_FORMAT=json for machine-readable output (production/SSH)
    let log_format = std::env::var("LLORCH_LOG_FORMAT").unwrap_or_else(|_| "pretty".to_string());

    if log_format == "json" {
        // JSON format for production/SSH (machine-readable)
        tracing_subscriber::fmt().with_target(false).json().init();
    } else {
        // Pretty format for development (human-readable)
        // TEAM-088: Use compact format with colors for better UX
        tracing_subscriber::fmt()
            .with_target(false)
            .with_level(true)
            .with_ansi(true)
            .with_timer(tracing_subscriber::fmt::time::uptime())
            .init();
    }

    // Parse CLI arguments (from pool-managerd)
    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        "Candle worker starting"
    );

    narrate(NarrationFields {
        actor: ACTOR_LLM_WORKER_RBEE,
        action: ACTION_STARTUP,
        target: args.worker_id.clone(),
        human: format!("Starting Candle worker on port {}", args.port),
        cute: Some(format!("Worker {} waking up to help with inference! 🌅", args.worker_id)),
        worker_id: Some(args.worker_id.clone()),
        ..Default::default()
    });

    // ============================================================
    // STEP 1: Load model to memory
    // ============================================================
    // TEAM-009: Initialize device and pass to backend
    use llm_worker_rbee::device::init_cpu_device;
    let device = init_cpu_device()?;

    tracing::info!(model = %args.model, "Loading Llama model...");

    narrate(NarrationFields {
        actor: ACTOR_MODEL_LOADER,
        action: ACTION_MODEL_LOAD,
        target: args.model.clone(),
        human: format!("Loading Llama model from {}", args.model),
        cute: Some("Fetching the sleepy Llama model from its cozy home! 📦".to_string()),
        worker_id: Some(args.worker_id.clone()),
        ..Default::default()
    });

    // TEAM-088: Wrap model loading with error narration
    let backend = match CandleInferenceBackend::load(&args.model, device) {
        Ok(backend) => {
            tracing::info!("Model loaded successfully");

            narrate(NarrationFields {
                actor: ACTOR_MODEL_LOADER,
                action: "model_load_success",
                target: args.model.clone(),
                human: "Model loaded successfully".to_string(),
                cute: Some("Model loaded and ready to help! ✅🎉".to_string()),
                worker_id: Some(args.worker_id.clone()),
                ..Default::default()
            });

            backend
        }
        Err(e) => {
            // TEAM-088: Narrate model loading failure with detailed error
            let error_msg = format!("{e:#}");

            narrate(NarrationFields {
                actor: ACTOR_MODEL_LOADER,
                action: "model_load_failed",
                target: args.model.clone(),
                human: format!(
                    "Model load failed: {}",
                    error_msg.lines().next().unwrap_or("unknown error")
                ),
                cute: Some("Oh no! Couldn't load the model! 😟💔".to_string()),
                worker_id: Some(args.worker_id.clone()),
                error_kind: Some("model_load_error".to_string()),
                ..Default::default()
            });

            tracing::error!(
                model = %args.model,
                error = %error_msg,
                "Model loading failed"
            );

            return Err(e);
        }
    };

    // ============================================================
    // STEP 2: Start heartbeat task
    // ============================================================
    // Send periodic heartbeats to rbee-hive to indicate worker is alive
    tracing::info!("Starting heartbeat task");
    
    let heartbeat_config = llm_worker_rbee::heartbeat::HeartbeatConfig::new(
        args.worker_id.clone(),
        args.hive_url.clone(),
    );
    let _heartbeat_handle = llm_worker_rbee::heartbeat::start_heartbeat_task(heartbeat_config);
    tracing::info!("Heartbeat task started (30s interval)");

    // ============================================================
    // STEP 3: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    // TEAM-017: Wrap backend in Mutex for stateful inference
    let backend = Arc::new(tokio::sync::Mutex::new(backend));

    // Security: Enforce proper auth based on mode
    let (addr, expected_token) = if args.local_mode {
        // LOCAL MODE: No auth, localhost only
        let token = std::env::var("LLORCH_API_TOKEN").unwrap_or_default();
        if !token.is_empty() {
            anyhow::bail!("--local-mode cannot use LLORCH_API_TOKEN (conflicting config)");
        }
        tracing::warn!("🏠 Local mode: No authentication, binding to 127.0.0.1 only");
        (SocketAddr::from(([127, 0, 0, 1], args.port)), String::new())
    } else {
        // NETWORK MODE: Auth required
        let token = std::env::var("LLORCH_API_TOKEN")
            .map_err(|_| anyhow::anyhow!("Network mode requires LLORCH_API_TOKEN (use --local-mode for dev)"))?;
        if token.is_empty() {
            anyhow::bail!("LLORCH_API_TOKEN cannot be empty in network mode");
        }
        tracing::info!("🌐 Network mode: Authentication enabled, binding to 0.0.0.0");
        (SocketAddr::from(([0, 0, 0, 0], args.port)), token)
    };

    // Create router with our backend (worker-http)
    // This wires up:
    // - GET /health -> backend.is_healthy()
    // - POST /execute -> backend.execute()
    // TEAM-102: Added expected_token for authentication
    let router = create_router(backend, expected_token);

    // Start HTTP server (worker-http)
    let server = HttpServer::new(addr, router).await?;

    // ============================================================
    // STEP 4: Run forever (until killed)
    // ============================================================
    // This blocks forever, processing HTTP requests
    // Pool-managerd will kill this process when:
    // - Worker needs to be shut down
    // - Model needs to be unloaded
    // - System is shutting down
    server.run().await?;

    Ok(())
}
