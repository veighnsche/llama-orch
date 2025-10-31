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

// TEAM-XXX: Build metadata via shadow-rs
use shadow_rs::shadow;
shadow!(build);

use clap::Parser;
use job_server::JobRegistry;
use llm_worker_rbee::{
    backend::{
        generation_engine::GenerationEngine,
        request_queue::{RequestQueue, TokenResponse},
        CandleInferenceBackend,
    },
    create_router,
    narration::{ACTION_MODEL_LOAD, ACTION_STARTUP},
    HttpServer,
};
use observability_narration_core::n;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

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

    /// Print build information and exit
    #[arg(long, hide = true)]
    build_info: bool,
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
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", build::BUILD_RUST_CHANNEL);
        std::process::exit(0);
    }

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

    n!(ACTION_STARTUP, "Starting Candle worker on port {}", args.port);

    // ============================================================
    // STEP 1: Load model to memory
    // ============================================================
    // TEAM-NARRATION-FIX: Device is compile-time (CPU), no runtime selection
    tracing::info!(model = %args.model, "Loading Llama model...");

    n!(ACTION_MODEL_LOAD, "Loading Llama model from {}", args.model);

    // TEAM-088: Wrap model loading with error narration
    let backend = match CandleInferenceBackend::load(&args.model) {
        Ok(backend) => {
            tracing::info!("Model loaded successfully");

            n!("model_load_success", "Model loaded successfully");

            backend
        }
        Err(e) => {
            // TEAM-088: Narrate model loading failure with detailed error
            let error_msg = format!("{e:#}");

            n!("model_load_failed", "Model load failed: {}", error_msg.lines().next().unwrap_or("unknown error"));

            tracing::error!(
                model = %args.model,
                error = %error_msg,
                "Model loading failed"
            );

            return Err(e);
        }
    };

    // ============================================================
    // STEP 2: Create request queue and start generation engine
    // ============================================================
    // TEAM-149: Real-time streaming architecture
    // - Request queue decouples HTTP from generation
    // - Generation engine runs in spawn_blocking
    // - Tokens flow through channels to SSE streams
    // TEAM-154: Added job registry for dual-call pattern
    tracing::info!("Creating request queue, job registry, and generation engine");

    // Wrap backend in Arc<Mutex> for sharing between engine and warmup
    let backend = Arc::new(Mutex::new(backend));

    // Create request queue
    let (request_queue, request_rx) = RequestQueue::new();
    let request_queue = Arc::new(request_queue);

    // TEAM-154: Create job registry for dual-call pattern
    // Generic over TokenResponse type from request_queue
    let job_server: Arc<JobRegistry<TokenResponse>> = Arc::new(JobRegistry::new());

    // Start generation engine in background
    let generation_engine = GenerationEngine::new(Arc::clone(&backend), request_rx);
    generation_engine.start();
    tracing::info!("Generation engine started");

    // ============================================================
    // STEP 3: Start heartbeat task
    // ============================================================
    // Send periodic heartbeats to queen to indicate worker is alive
    // TEAM-285: Updated to use WorkerInfo (TEAM-284 contract changes)
    tracing::info!("Starting heartbeat task");

    let worker_info = worker_contract::WorkerInfo {
        id: args.worker_id.clone(),
        model_id: args.model_ref.clone(),
        device: format!("{}:{}", args.backend, args.device),
        port: args.port,
        status: worker_contract::WorkerStatus::Ready,
        implementation: "llm-worker-rbee".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let _heartbeat_handle = llm_worker_rbee::heartbeat::start_heartbeat_task(
        worker_info,
        args.hive_url.clone(), // Actually queen URL (TEAM-261)
    );
    tracing::info!("Heartbeat task started (30s interval)");

    // ============================================================
    // STEP 4: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

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
        let token = std::env::var("LLORCH_API_TOKEN").map_err(|_| {
            anyhow::anyhow!("Network mode requires LLORCH_API_TOKEN (use --local-mode for dev)")
        })?;
        if token.is_empty() {
            anyhow::bail!("LLORCH_API_TOKEN cannot be empty in network mode");
        }
        tracing::info!("🌐 Network mode: Authentication enabled, binding to 0.0.0.0");
        (SocketAddr::from(([0, 0, 0, 0], args.port)), token)
    };

    // TEAM-149: Create router with request queue (not backend directly)
    // HTTP handlers add requests to queue and return immediately
    // Generation happens in spawn_blocking, tokens stream in real-time
    // TEAM-102: Added expected_token for authentication
    // TEAM-154: Added job_server for dual-call pattern
    let router = create_router(request_queue, job_server, expected_token);

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
