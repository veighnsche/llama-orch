// TEAM-109: Audited 2025-10-18 - ⚠️ KNOWN ISSUE - Secrets in env vars (line 106)

//! CUDA GPU worker binary
//!
//! Uses NVIDIA CUDA for GPU inference with strict device residency.
//! This binary is feature-gated to CUDA backend only.
//!
//! Created by: TEAM-007
//! Modified by: TEAM-014 (Added GPU warmup)
//! Modified by: TEAM-017 (updated for multi-model support)

use anyhow::Result;
use clap::Parser;
use llm_worker_rbee::{backend::CandleInferenceBackend, setup_worker_with_backend, HttpServer};
use std::net::SocketAddr;

/// CLI arguments for CUDA worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-cuda-candled")]
#[command(about = "CUDA GPU Candle-based multi-model worker daemon")]
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

    /// Hive URL - where to send heartbeats
    #[arg(long)]
    hive_url: String,

    /// CUDA device ID (default: 0)
    #[arg(long, default_value = "0")]
    cuda_device: usize,
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
        cuda_device = args.cuda_device,
        backend = "cuda",
        "Starting llorch-cuda-candled"
    );

    // ============================================================
    // STEP 1: Load model to GPU
    // ============================================================
    // TEAM-018: Load model with auto-detected architecture
    // TEAM-NARRATION-FIX: Device is compile-time (CUDA), GPU ID passed at runtime
    tracing::info!(model = %args.model, cuda_device = args.cuda_device, "Loading model to GPU...");
    let mut backend = CandleInferenceBackend::load(&args.model, args.cuda_device)?;
    tracing::info!("Model loaded successfully on GPU {}", args.cuda_device);

    // ============================================================
    // STEP 2.5: GPU Warmup
    // ============================================================
    // TEAM-014: Warmup GPU to eliminate cold start overhead
    backend.warmup()?;
    tracing::info!("GPU warmup complete - ready for inference");

    // ============================================================
    // STEP 3: Start heartbeat task
    // ============================================================
    // TEAM-285: Updated to use WorkerInfo (TEAM-284 contract changes)
    tracing::info!("Starting heartbeat task");

    let worker_info = worker_contract::WorkerInfo {
        id: args.worker_id.clone(),
        model_id: args.model_ref.clone(),
        device: format!("cuda:{}", args.cuda_device),
        port: args.port,
        status: worker_contract::WorkerStatus::Ready,
        implementation: "llm-worker-rbee-cuda".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let _heartbeat_handle =
        llm_worker_rbee::heartbeat::start_heartbeat_task(worker_info, args.hive_url.clone());
    tracing::info!("Heartbeat task started (30s interval)");

    // ============================================================
    // STEP 4: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

    // TEAM-102: Load API token for authentication
    let expected_token = std::env::var("LLORCH_API_TOKEN").unwrap_or_else(|_| {
        tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
        String::new()
    });

    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

    // TEAM-NARRATION-FIX: Use helper to setup job-based architecture
    let router = setup_worker_with_backend(backend, expected_token);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-cuda-candled ready on port {} (GPU {})", args.port, args.cuda_device);

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
