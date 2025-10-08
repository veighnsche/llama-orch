//! CUDA GPU worker binary
//!
//! Uses NVIDIA CUDA for GPU inference with strict device residency.
//! This binary is feature-gated to CUDA backend only.
//!
//! Created by: TEAM-007
//! Modified by: TEAM-014 (Added GPU warmup)

use anyhow::Result;
use clap::Parser;
use llorch_candled::device::{init_cuda_device, verify_device};
use llorch_candled::{backend::CandleInferenceBackend, callback_ready, create_router, HttpServer};
use std::net::SocketAddr;
use std::sync::Arc;

/// CLI arguments for CUDA worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-cuda-candled")]
#[command(about = "CUDA GPU Candle-based Llama-2 worker daemon")]
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

    /// CUDA device ID (default: 0)
    #[arg(long, default_value = "0")]
    cuda_device: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt()
        .with_target(false)
        .json()
        .init();

    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        cuda_device = args.cuda_device,
        backend = "cuda",
        "Starting llorch-cuda-candled"
    );

    // ============================================================
    // STEP 1: Initialize CUDA device
    // ============================================================
    tracing::info!(cuda_device = args.cuda_device, "Initializing CUDA device");
    let device = init_cuda_device(args.cuda_device)?;
    verify_device(&device)?;
    tracing::info!("CUDA device {} initialized and verified", args.cuda_device);

    // ============================================================
    // STEP 2: Load model to memory (on CUDA device)
    // ============================================================
    // TEAM-009: Pass device to backend
    tracing::info!(model = %args.model, "Loading Llama model to GPU...");
    let backend = CandleInferenceBackend::load(&args.model, device)?;
    tracing::info!("Model loaded successfully on GPU");

    // ============================================================
    // STEP 2.5: GPU Warmup
    // ============================================================
    // TEAM-014: Warmup GPU to eliminate cold start overhead
    backend.warmup()?;
    tracing::info!("GPU warmup complete - ready for inference");

    // ============================================================
    // STEP 3: Call back to pool-managerd (worker ready)
    // ============================================================
    if !args.callback_url.contains("localhost:9999") {
        callback_ready(
            &args.callback_url,
            &args.worker_id,
            backend.memory_bytes(),
            args.port,
        )
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

    tracing::info!(
        "llorch-cuda-candled ready on port {} (GPU {})",
        args.port,
        args.cuda_device
    );

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
