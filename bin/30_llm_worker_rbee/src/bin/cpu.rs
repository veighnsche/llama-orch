// TEAM-109: Audited 2025-10-18 - ⚠️ KNOWN ISSUE - Secrets in env vars (line 101)

//! CPU-only worker binary
//!
//! Uses MKL (Linux/Windows) or Accelerate (macOS) for CPU inference.
//! This binary is feature-gated to CPU backend only.
//!
//! Created by: TEAM-007
//! Modified by: TEAM-017 (updated for multi-model support)

use anyhow::Result;
use clap::Parser;
use llm_worker_rbee::device::{init_cpu_device, verify_device};
use llm_worker_rbee::{backend::CandleInferenceBackend, create_router, HttpServer};
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

    /// Model reference (e.g., "hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    #[arg(long)]
    model_ref: String,

    /// HTTP server port - assigned by pool-managerd
    #[arg(long)]
    port: u16,

    /// Hive URL - where to send heartbeats
    #[arg(long)]
    hive_url: String,
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
    // STEP 3: Start heartbeat task
    // ============================================================
    // TEAM-285: Updated to use WorkerInfo (TEAM-284 contract changes)
    tracing::info!("Starting heartbeat task");

    let worker_info = worker_contract::WorkerInfo {
        id: args.worker_id.clone(),
        model_id: args.model_ref.clone(),
        device: "cpu:0".to_string(),
        port: args.port,
        status: worker_contract::WorkerStatus::Ready,
        implementation: "llm-worker-rbee-cpu".to_string(),
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
    // TEAM-017: Wrap backend in Mutex for stateful inference
    let backend = Arc::new(Mutex::new(backend));

    // TEAM-102: Load API token for authentication
    let expected_token = std::env::var("LLORCH_API_TOKEN").unwrap_or_else(|_| {
        tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
        String::new()
    });

    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

    let router = create_router(backend, expected_token);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-cpu-candled ready on port {}", args.port);

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
