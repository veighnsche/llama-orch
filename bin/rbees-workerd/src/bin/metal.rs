//! Apple Metal GPU worker binary
//!
//! Uses Apple Metal for GPU inference on macOS with Apple Silicon.
//! This binary is feature-gated to Metal backend only.
//!
//! Created by: TEAM-018

use anyhow::Result;
use clap::Parser;
use llorch_candled::device::{init_metal_device, verify_device};
use llorch_candled::{backend::CandleInferenceBackend, callback_ready, create_router, HttpServer};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

/// CLI arguments for Metal worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-metal-candled")]
#[command(about = "Apple Metal GPU Candle-based multi-model worker daemon (pre-release)")]
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

    /// Metal device ID (default: 0)
    #[arg(long, default_value = "0")]
    metal_device: usize,
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
        metal_device = args.metal_device,
        backend = "metal",
        status = "pre-release",
        "Starting llorch-metal-candled"
    );

    // ============================================================
    // STEP 1: Initialize Metal device
    // ============================================================
    tracing::info!(metal_device = args.metal_device, "Initializing Apple Metal device (GPU)");
    let device = init_metal_device(args.metal_device)?;
    verify_device(&device)?;
    tracing::info!("Metal device {} initialized and verified", args.metal_device);

    // ============================================================
    // STEP 2: Load model to memory (on Metal device)
    // ============================================================
    // TEAM-018: Load model with auto-detected architecture
    tracing::info!(model = %args.model, "Loading model to Metal GPU...");
    let mut backend = CandleInferenceBackend::load(&args.model, device)?;
    tracing::info!("Model loaded successfully on Metal GPU");

    // ============================================================
    // STEP 2.5: GPU Warmup
    // ============================================================
    // TEAM-018: Warmup Metal GPU to eliminate cold start overhead
    backend.warmup()?;
    tracing::info!("Metal GPU warmup complete - ready for inference");

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
    // TEAM-018: Wrap backend in Mutex for stateful inference
    let backend = Arc::new(Mutex::new(backend));

    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!(
        "llorch-metal-candled ready on port {} (Metal GPU {})",
        args.port,
        args.metal_device
    );

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
