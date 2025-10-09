//! CPU-only worker binary
//!
//! Uses MKL (Linux/Windows) or Accelerate (macOS) for CPU inference.
//! This binary is feature-gated to CPU backend only.
//!
//! Created by: TEAM-007
//! Modified by: TEAM-017 (updated for multi-model support)

use anyhow::Result;
use clap::Parser;
use llorch_candled::device::{init_cpu_device, verify_device};
use llorch_candled::{backend::CandleInferenceBackend, callback_ready, create_router, HttpServer};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

/// CLI arguments for CPU worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-cpu-candled")]
#[command(about = "CPU-only Candle-based multi-model worker daemon")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF or `SafeTensors` format)
    #[arg(long)]
    model: String,

    /// HTTP server port - assigned by pool-managerd
    #[arg(long)]
    port: u16,

    /// Pool manager callback URL - where to report ready status
    #[arg(long)]
    callback_url: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt().with_target(false).json().init();

    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        backend = "cpu",
        "Starting llorch-cpu-candled"
    );

    // ============================================================
    // STEP 1: Initialize CPU device
    // ============================================================
    tracing::info!("Initializing CPU device");
    let device = init_cpu_device()?;
    verify_device(&device)?;
    tracing::info!("CPU device initialized and verified");

    // ============================================================
    // STEP 2: Load model to memory
    // ============================================================
    // TEAM-009: Pass device to backend
    // TEAM-017: Load model with auto-detected architecture
    tracing::info!(model = %args.model, "Loading model...");
    let backend = CandleInferenceBackend::load(&args.model, device)?;
    tracing::info!("Model loaded successfully");

    // ============================================================
    // STEP 3: Call back to pool-managerd (worker ready)
    // ============================================================
    if args.callback_url.contains("localhost:9999") {
        tracing::info!("Test mode: skipping pool manager callback");
    } else {
        callback_ready(&args.callback_url, &args.worker_id, backend.memory_bytes(), args.port)
            .await?;

        tracing::info!("Callback sent to pool-managerd");
    }

    // ============================================================
    // STEP 4: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    // TEAM-017: Wrap backend in Mutex for stateful inference
    let backend = Arc::new(Mutex::new(backend));

    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-cpu-candled ready on port {}", args.port);

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
