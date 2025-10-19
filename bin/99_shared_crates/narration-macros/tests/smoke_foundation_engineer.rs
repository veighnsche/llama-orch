//! Smoke test: Foundation engineers can use macros out-of-the-box.
//!
//! This test verifies that a foundation engineer can:
//! - Import and use #[narrate(...)] macro
//! - Import and use #[trace_fn] macro
//! - Have actor auto-inferred from module path
//! - Use template interpolation
//! - Support async functions
//! - Support Result types

use observability_narration_macros::{narrate, trace_fn};

// Simulate orchestratord module for actor inference
mod orchestratord {
    use super::*;

    /// Foundation engineer writes this - basic narration
    #[narrate(action = "enqueue", human = "Enqueued job {job_id}")]
    pub fn enqueue_job(job_id: &str) -> Result<(), String> {
        Ok(())
    }

    /// Foundation engineer writes this - with cute mode
    #[narrate(
        action = "dispatch",
        human = "Dispatched job {job_id} to worker {worker_id}",
        cute = "Sent job {job_id} off to its new friend {worker_id}! 🎫"
    )]
    pub fn dispatch_job(job_id: &str, worker_id: &str) -> Result<(), String> {
        Ok(())
    }

    /// Foundation engineer writes this - async function
    #[narrate(action = "complete", human = "Completed job {job_id}")]
    pub async fn complete_job(job_id: &str) -> Result<(), String> {
        Ok(())
    }

    /// Foundation engineer writes this - function tracing
    #[trace_fn]
    pub fn process_request(request_id: &str) -> Result<String, String> {
        Ok(format!("Processed {}", request_id))
    }

    /// Foundation engineer writes this - async tracing
    #[trace_fn]
    pub async fn async_process(data: &str) -> Result<String, String> {
        Ok(format!("Processed {}", data))
    }
}

// Simulate pool_managerd module
mod pool_managerd {
    use super::*;

    #[narrate(action = "spawn", human = "Spawning worker on {device}")]
    pub fn spawn_worker(device: &str) -> Result<(), String> {
        Ok(())
    }

    #[trace_fn]
    pub fn register_worker(_worker_id: &str) -> Result<(), String> {
        Ok(())
    }
}

// Simulate worker_orcd module
mod worker_orcd {
    use super::*;

    #[narrate(action = "inference_start", human = "Starting inference for job {job_id}")]
    pub async fn start_inference(job_id: &str) -> Result<(), String> {
        Ok(())
    }

    #[trace_fn]
    pub async fn execute_inference(job_id: &str) -> Result<String, String> {
        Ok(format!("Result for {}", job_id))
    }
}

#[test]
fn smoke_macros_compile() {
    // Foundation engineer: Just verify the macros compile and generate valid code
    // The macros wrap the functions, so we test by calling them

    // Sync functions
    let _ = orchestratord::enqueue_job("job-123");
    let _ = orchestratord::dispatch_job("job-123", "worker-1");
    let _ = orchestratord::process_request("req-abc");
    let _ = pool_managerd::spawn_worker("GPU0");
    let _ = pool_managerd::register_worker("worker-1");

    // Test passes if all functions compile and are callable
}
