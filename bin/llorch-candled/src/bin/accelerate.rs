//! Apple Accelerate worker binary
//!
//! Uses Apple Accelerate framework for optimized CPU inference on macOS.
//! Note: This is CPU Accelerate, NOT Metal (GPU).
//! This binary is feature-gated to Accelerate backend only.
//!
//! Created by: TEAM-007

use anyhow::Result;
use clap::Parser;
use llorch_candled::device::{init_accelerate_device, verify_device};
use llorch_candled::{backend::CandleInferenceBackend, callback_ready, create_router, HttpServer};
use std::net::SocketAddr;
use std::sync::Arc;

/// CLI arguments for Accelerate worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-accelerate-candled")]
#[command(about = "Apple Accelerate Candle-based Llama-2 worker daemon")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF or SafeTensors format)
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
        backend = "accelerate",
        "Starting llorch-accelerate-candled"
    );

    // ============================================================
    // STEP 1: Initialize Accelerate device
    // ============================================================
    tracing::info!("Initializing Apple Accelerate device (CPU-optimized)");
    let device = init_accelerate_device()?;
    verify_device(&device)?;
    tracing::info!("Accelerate device initialized and verified");

    // ============================================================
    // STEP 2: Load model to memory
    // ============================================================
    // TEAM-009: Pass device to backend
    tracing::info!(model = %args.model, "Loading Llama model...");
    let backend = CandleInferenceBackend::load(&args.model, device)?;
    tracing::info!("Model loaded successfully");

    // ============================================================
    // STEP 3: Call back to pool-managerd (worker ready)
    // ============================================================
    if !args.callback_url.contains("localhost:9999") {
        callback_ready(&args.callback_url, &args.worker_id, backend.memory_bytes(), args.port)
            .await?;

        tracing::info!("Callback sent to pool-managerd");
    } else {
        tracing::info!("Test mode: skipping pool manager callback");
    }

    // ============================================================
    // STEP 4: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let backend = Arc::new(backend);

    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-accelerate-candled ready on port {}", args.port);

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
