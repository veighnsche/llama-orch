//! llorch-cpud: CPU-based GPT-2 inference worker daemon
//!
//! This is the main entry point for the llorch-cpud worker.
//! It uses worker-http for HTTP server infrastructure.
//!
//! IMPORTS: worker-http, worker-common
//! ARCHITECTURE: Single-threaded (tokio::main(flavor = "current_thread"))
//! CHECKPOINT: 0 (Foundation - HTTP Server)

use anyhow::Result;
use std::sync::Arc;
use worker_common::startup;
use worker_http::{create_router, HttpServer};

mod backend;
mod cache;
mod error;
mod layers;
mod model;
mod tensor;

use backend::CpuInferenceBackend;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    tracing::info!("Starting llorch-cpud...");

    // Parse CLI arguments
    let args = parse_args();

    // Load model and create backend
    tracing::info!("Loading model from: {}", args.model_path);
    let backend = CpuInferenceBackend::load(&args.model_path)?;
    let backend = Arc::new(backend);

    // Callback to pool manager (worker-common)
    tracing::info!("Notifying pool manager at: {}", args.callback_url);
    startup::callback_ready(
        &args.callback_url,
        &args.worker_id,
        backend.memory_bytes(),
        args.port,
    )
    .await?;

    // Start HTTP server (worker-http)
    tracing::info!("Starting HTTP server on port {}", args.port);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", args.port).parse()?;
    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-cpud ready!");
    server.run().await?;

    Ok(())
}

#[derive(Debug)]
struct Args {
    model_path: String,
    callback_url: String,
    worker_id: String,
    port: u16,
}

fn parse_args() -> Args {
    // TODO: Use clap for proper CLI parsing
    // For now, use environment variables
    Args {
        model_path: std::env::var("MODEL_PATH").unwrap_or_else(|_| "./models/gpt2-medium".to_string()),
        callback_url: std::env::var("CALLBACK_URL").unwrap_or_else(|_| "http://localhost:8080".to_string()),
        worker_id: std::env::var("WORKER_ID").unwrap_or_else(|_| "llorch-cpud-0".to_string()),
        port: std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000),
    }
}
