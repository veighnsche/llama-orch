// TEAM-303: Fake worker binary for E2E testing
//!
//! Simulates llama-worker-rbee behavior for integration tests:
//! - Emits narration to stdout (captured by ProcessNarrationCapture)
//! - Simulates worker lifecycle (startup, load model, ready, inference)
//! - Propagates correlation ID

use observability_narration_core::{n, with_narration_context, NarrationContext};

#[tokio::main]
async fn main() {
    // Get job_id from environment
    let job_id = std::env::var("JOB_ID")
        .expect("JOB_ID environment variable required");
    
    // Get correlation ID if provided
    let correlation_id = std::env::var("CORRELATION_ID").ok();
    
    // Create narration context
    let mut ctx = NarrationContext::new().with_job_id(&job_id);
    if let Some(corr_id) = correlation_id.as_ref() {
        ctx = ctx.with_correlation_id(corr_id);
    }
    
    with_narration_context(ctx, async {
        // Worker startup sequence
        n!("worker_startup", "Worker process starting");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("worker_init", "Initializing worker environment");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("worker_load_model", "Loading model into memory");
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        n!("worker_model_loaded", "Model loaded successfully");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("worker_ready", "Worker ready to serve requests");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Simulate some work
        n!("worker_inference_start", "Starting inference");
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        n!("worker_inference_progress", "Inference 50% complete");
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        n!("worker_inference_complete", "Inference completed successfully");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("worker_shutdown", "Worker shutting down gracefully");
    }).await;
    
    // Exit successfully
    std::process::exit(0);
}
