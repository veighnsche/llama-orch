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
    backend::CandleInferenceBackend, callback_ready, create_router, narration::{ACTOR_LLM_WORKER_RBEE, ACTION_STARTUP, ACTOR_MODEL_LOADER, ACTION_MODEL_LOAD, ACTION_CALLBACK_READY}, HttpServer,
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

    /// Pool manager callback URL - where to report ready status
    #[arg(long)]
    callback_url: String,
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
/// 3. Call back to pool-managerd (worker ready)
/// 4. Start HTTP server
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
            let error_msg = format!("{:#}", e);
            
            narrate(NarrationFields {
                actor: ACTOR_MODEL_LOADER,
                action: "model_load_failed",
                target: args.model.clone(),
                human: format!("Model load failed: {}", error_msg.lines().next().unwrap_or("unknown error")),
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
            
            // TEAM-088: Try to callback to rbee-hive with error state
            if !args.callback_url.contains("localhost:9999") {
                narrate(NarrationFields {
                    actor: ACTOR_LLM_WORKER_RBEE,
                    action: "callback_error",
                    target: args.callback_url.clone(),
                    human: "Reporting error to pool-managerd".to_string(),
                    cute: Some("Telling pool-managerd we couldn't start 😢".to_string()),
                    worker_id: Some(args.worker_id.clone()),
                    ..Default::default()
                });
                
                // TODO: Implement error callback to rbee-hive
                // For now, just log the intent
                tracing::warn!("Worker failed to load model, should callback error to rbee-hive");
            }
            
            return Err(e);
        }
    };

    // ============================================================
    // STEP 2: Call back to pool-managerd (worker ready)
    // ============================================================
    // This tells pool-managerd:
    // - Worker is ready to accept requests
    // - Worker is listening on args.port
    // - Worker has loaded X bytes of memory
    // - Worker type and capabilities
    if args.callback_url.contains("localhost:9999") {
        tracing::info!("Test mode: skipping pool manager callback");
        
        narrate(NarrationFields {
            actor: ACTOR_LLM_WORKER_RBEE,
            action: "test_mode",
            target: "callback".to_string(),
            human: "Test mode: skipping callback to pool-managerd".to_string(),
            cute: Some("Running in test mode! No callback needed! 🧪".to_string()),
            worker_id: Some(args.worker_id.clone()),
            ..Default::default()
        });
    } else {
        narrate(NarrationFields {
            actor: ACTOR_LLM_WORKER_RBEE,
            action: ACTION_CALLBACK_READY,
            target: args.callback_url.clone(),
            human: format!("Reporting ready to pool-managerd at {}", args.callback_url),
            cute: Some("Waving hello to pool-managerd: 'I'm ready to work!' 👋".to_string()),
            story: Some(format!(
                "\"I'm ready!\" announced worker-{}. \"Great!\" replied pool-managerd.",
                args.worker_id
            )),
            worker_id: Some(args.worker_id.clone()),
            ..Default::default()
        });

        callback_ready(
            &args.callback_url,
            &args.worker_id,
            &args.model_ref,
            &args.backend,
            args.device,
            backend.memory_bytes(),
            args.port,
        )
        .await?;

        tracing::info!("Callback sent to pool-managerd");
    }

    // ============================================================
    // STEP 3: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    // TEAM-017: Wrap backend in Mutex for stateful inference
    let backend = Arc::new(tokio::sync::Mutex::new(backend));

    // TEAM-102: Load API token for authentication
    // TODO: Replace with secrets-management file-based loading
    let expected_token = std::env::var("LLORCH_API_TOKEN")
        .unwrap_or_else(|_| {
            tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
            String::new()
        });
    
    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

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
