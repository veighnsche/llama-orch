//! llorch-candled - Candle-based Llama-2 inference worker daemon
//!
//! This worker implements Llama-2 inference using a hybrid approach:
//! - Pure ndarray for CPU (checkpoint validation)
//! - Candle kernels for CUDA acceleration (optional)
//!
//! Architecture follows CANDLE_INTEGRATION_HANDOFF.md:
//! - Use Candle's kernels, NOT the framework
//! - Keep checkpoint-driven validation
//! - Maintain educational value
//!
//! Created by: TEAM-000 (Foundation)

use clap::Parser;
use llorch_candled::{backend::CandleInferenceBackend, callback_ready, create_router, HttpServer};
use std::net::SocketAddr;
use std::sync::Arc;

/// CLI arguments for worker daemon
///
/// These are provided by pool-managerd when spawning the worker.
/// This is NOT a user-facing CLI - it's for orchestration.
#[derive(Parser, Debug)]
#[command(name = "llorch-candled")]
#[command(about = "Candle-based Llama-2 worker daemon for llama-orch")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF format)
    #[arg(long)]
    model: String,

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
/// - flavor = "current_thread" ensures NO thread pool
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
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt().with_target(false).json().init();

    // Parse CLI arguments (from pool-managerd)
    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        "Candle worker starting"
    );

    // ============================================================
    // STEP 1: Load model to memory
    // ============================================================
    // TEAM-009: Initialize device and pass to backend
    use llorch_candled::device::init_cpu_device;
    let device = init_cpu_device()?;

    tracing::info!(model = %args.model, "Loading Llama model...");
    let backend = CandleInferenceBackend::load(&args.model, device)?;
    tracing::info!("Model loaded successfully");

    // ============================================================
    // STEP 2: Call back to pool-managerd (worker ready)
    // ============================================================
    // This tells pool-managerd:
    // - Worker is ready to accept requests
    // - Worker is listening on args.port
    // - Worker has loaded X bytes of memory
    // - Worker type and capabilities
    if !args.callback_url.contains("localhost:9999") {
        callback_ready(&args.callback_url, &args.worker_id, backend.memory_bytes(), args.port)
            .await?;

        tracing::info!("Callback sent to pool-managerd");
    } else {
        tracing::info!("Test mode: skipping pool manager callback");
    }

    // ============================================================
    // STEP 3: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let backend = Arc::new(backend);

    // Create router with our backend (worker-http)
    // This wires up:
    // - GET /health -> backend.is_healthy()
    // - POST /execute -> backend.execute()
    let router = create_router(backend);

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
