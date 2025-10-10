//! Mock Worker Binary for BDD Tests
//!
//! Created by: TEAM-059
//! Modified by: TEAM-061 (added ready callback timeout to prevent hangs)
//!
//! This is a REAL binary that runs as a separate process to simulate a worker.
//! It provides HTTP endpoints for inference and sends ready callbacks to queen-rbee.

use axum::{
    routing::{get, post},
    Router, Json,
    http::header,
    response::IntoResponse,
};
use clap::Parser;
use std::net::SocketAddr;
use tokio::time::{sleep, Duration};

#[derive(Parser, Debug)]
#[command(name = "mock-worker")]
#[command(about = "Mock worker process for BDD testing")]
struct Args {
    /// Port to listen on
    #[arg(long)]
    port: u16,

    /// Worker ID
    #[arg(long)]
    worker_id: String,

    /// Queen-rbee URL for ready callback
    #[arg(long)]
    queen_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .init();

    let args = Args::parse();

    tracing::info!("🤖 Mock worker {} starting on port {}", args.worker_id, args.port);

    // Build HTTP server
    let app = Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/ready", get(handle_ready))
        .route("/v1/inference", post(handle_inference));

    let addr: SocketAddr = format!("127.0.0.1:{}", args.port).parse()?;
    
    // Start server in background
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("✅ Mock worker HTTP server listening on {}", addr);

    // TEAM-059: Server-first, then callback (per memory spec)
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("Server failed");
    });

    // Wait for server to be ready
    sleep(Duration::from_millis(100)).await;

    // Send ready callback to queen-rbee
    send_ready_callback(&args.queen_url, &args.worker_id, &format!("http://127.0.0.1:{}", args.port)).await?;

    tracing::info!("✅ Mock worker {} ready and registered", args.worker_id);

    // Keep running until killed
    tokio::signal::ctrl_c().await?;
    tracing::info!("🛑 Mock worker {} shutting down", args.worker_id);

    Ok(())
}

async fn send_ready_callback(queen_url: &str, worker_id: &str, worker_url: &str) -> anyhow::Result<()> {
    // TEAM-061: Create HTTP client with timeouts to prevent hangs
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .connect_timeout(Duration::from_secs(3))
        .build()?;
    
    let callback_url = format!("{}/v2/workers/ready", queen_url);

    let payload = serde_json::json!({
        "worker_id": worker_id,
        "url": worker_url,
        "model_ref": "mock-model",  // Per memory: must include model_ref
        "state": "idle",
        "backend": "cpu",
        "device": 0,
        "slots_total": 4,
        "slots_available": 4,
    });

    // TEAM-061: Wrap callback in timeout with retry logic
    for attempt in 1..=3 {
        match tokio::time::timeout(
            Duration::from_secs(10),
            client.post(&callback_url).json(&payload).send()
        )
        .await
        {
            Ok(Ok(resp)) => {
                if resp.status().is_success() {
                    tracing::info!("✅ Ready callback sent to {} (attempt {})", callback_url, attempt);
                    return Ok(());
                } else {
                    tracing::warn!("⚠️  Ready callback returned status: {} (attempt {})", resp.status(), attempt);
                }
            }
            Ok(Err(e)) => {
                tracing::warn!("⚠️  Ready callback failed: {} (attempt {})", e, attempt);
            }
            Err(_) => {
                tracing::warn!("⚠️  Ready callback timeout after 10s (attempt {})", attempt);
            }
        }
        
        if attempt < 3 {
            sleep(Duration::from_millis(500 * attempt as u64)).await;
        }
    }

    tracing::error!("❌ Ready callback failed after 3 attempts");
    Ok(())
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "state": "idle"
    }))
}

async fn handle_ready() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "ready": true,
        "state": "idle"
    }))
}

async fn handle_inference(Json(req): Json<serde_json::Value>) -> impl IntoResponse {
    tracing::info!("🔮 Mock worker: inference request: {:?}", req);

    // TEAM-059: Return REAL SSE stream with mock tokens
    let sse_response = "data: {\"t\":\"Once\"}\n\ndata: {\"t\":\" upon\"}\n\ndata: {\"t\":\" a\"}\n\ndata: {\"t\":\" time\"}\n\ndata: {\"t\":\"...\"}\n\ndata: [DONE]\n\n";

    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        sse_response
    )
}
