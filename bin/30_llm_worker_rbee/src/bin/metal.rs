// TEAM-109: Audited 2025-10-18 - ⚠️ KNOWN ISSUE - Secrets in env vars (line 104)

//! Apple Metal GPU worker binary
//!
//! Uses Apple Metal for GPU inference on macOS with Apple Silicon.
//! This binary is feature-gated to Metal backend only.
//!
//! Created by: TEAM-018

use anyhow::Result;
use clap::Parser;
use llm_worker_rbee::{backend::CandleInferenceBackend, setup_worker_with_backend, HttpServer};
use std::net::SocketAddr;

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

    /// Hive URL - where to send heartbeats
    #[arg(long)]
    hive_url: String,

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
    // STEP 1: Load model to Metal GPU
    // ============================================================
    // TEAM-018: Load model with auto-detected architecture
    // TEAM-NARRATION-FIX: Device is compile-time (Metal), GPU ID passed at runtime
    tracing::info!(model = %args.model, metal_device = args.metal_device, "Loading model to Metal GPU...");
    let mut backend = CandleInferenceBackend::load(&args.model, args.metal_device)?;
    tracing::info!("Model loaded successfully on Metal GPU");

    // ============================================================
    // STEP 2.5: GPU Warmup
    // ============================================================
    // TEAM-018: Warmup Metal GPU to eliminate cold start overhead
    backend.warmup()?;
    tracing::info!("Metal GPU warmup complete - ready for inference");

    // ============================================================
    // STEP 3: Start heartbeat task
    // ============================================================
    // TEAM-285: Updated to use WorkerInfo (TEAM-284 contract changes)
    tracing::info!("Starting heartbeat task");

    let worker_info = worker_contract::WorkerInfo {
        id: args.worker_id.clone(),
        model_id: args.model_ref.clone(),
        device: format!("metal:{}", args.metal_device),
        port: args.port,
        status: worker_contract::WorkerStatus::Ready,
        implementation: "llm-worker-rbee-metal".to_string(),
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

    tracing::info!(
        "llorch-metal-candled ready on port {} (Metal GPU {})",
        args.port,
        args.metal_device
    );

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
