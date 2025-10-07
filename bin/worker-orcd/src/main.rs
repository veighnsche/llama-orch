//! worker-orcd — GPU worker daemon
//!
//! Self-contained inference executor that loads ONE model to VRAM and executes
//! inference requests from orchestratord.
//!
//! # Architecture
//!
//! ```
//! ┌─────────────────────────────────────────┐
//! │ Rust Layer (src/*.rs)                   │
//! │ • HTTP server (axum)                    │
//! │ • CLI parsing                           │
//! │ • SSE streaming                         │
//! │ • Error handling                        │
//! └────────────┬────────────────────────────┘
//!              │ FFI (unsafe extern "C")
//! ┌────────────▼────────────────────────────┐
//! │ C++/CUDA Layer (cuda/*.cpp, *.cu)       │
//! │ • CUDA context management               │
//! │ • Model loading (disk → VRAM)           │
//! │ • Inference execution                   │
//! │ • VRAM health checks                    │
//! └─────────────────────────────────────────┘
//! ```
//!
//! See: `.specs/01_cuda_ffi_boundary.md`

use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use worker_common::startup;
use worker_http::{create_router, HttpServer};
use worker_orcd::cuda;
use worker_orcd::inference::cuda_backend::CudaInferenceBackend;

#[derive(Parser, Debug)]
#[command(name = "worker-orcd")]
#[command(about = "GPU worker daemon for llama-orch")]
struct Args {
    /// Worker ID (UUID)
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF format)
    #[arg(long)]
    model: String,

    /// CUDA device ID (0, 1, ...)
    #[arg(long)]
    gpu_device: i32,

    /// HTTP server port
    #[arg(long)]
    port: u16,

    /// Pool manager callback URL
    #[arg(long)]
    callback_url: String,
}

// [TEAM PICASSO 2025-10-07T17:47Z] BUG FIX: Use single-threaded runtime per M0-W-1301
// SPEC VIOLATION: M0-W-1301 requires "Worker-orcd MUST process inference requests sequentially"
// ROOT CAUSE: Multi-threaded tokio runtime was used by default, violating spec and adding complexity
// IMPACT: Caused logging issues, thread contention, unnecessary overhead
// FIX: Explicitly use current_thread flavor for single-threaded execution
// REFERENCE: bin/.specs/01_M0_worker_orcd.md line 901-905
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_target(false).json().init();

    // Parse CLI arguments
    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        gpu_device = args.gpu_device,
        port = args.port,
        "Worker starting"
    );

    // Initialize CUDA context
    let cuda_ctx = cuda::Context::new(args.gpu_device)?;
    tracing::info!(gpu_device = args.gpu_device, "CUDA context initialized");

    // Load model to VRAM
    tracing::info!(model = %args.model, "Loading model to VRAM...");
    let cuda_model = cuda_ctx.load_model(&args.model)?;
    tracing::info!(vram_bytes = cuda_model.vram_bytes(), "Model loaded to VRAM");

    // Call back to pool manager (skip for test mode)
    if !args.callback_url.contains("localhost:9999") {
        startup::callback_ready(
            &args.callback_url,
            &args.worker_id,
            cuda_model.vram_bytes(),
            args.port,
        )
        .await?;
    } else {
        tracing::info!("Test mode: skipping pool manager callback");
    }

    tracing::info!("Worker ready, starting HTTP server");

    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let backend = Arc::new(
        CudaInferenceBackend::new(cuda_model, &args.model)
            .map_err(|e| anyhow::anyhow!("Failed to create inference backend: {}", e))?,
    );
    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;

    server.run().await?;

    Ok(())
}
