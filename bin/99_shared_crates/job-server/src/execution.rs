//! Job execution and streaming
//!
//! TEAM-186: Execute and stream helper
//! TEAM-305: Added timeout and cancellation support
//! TEAM-312: Extracted to separate module, migrated to n!() macro
//! TEAM-312: DELETED execute_and_stream - use execute_and_stream instead with timeout: None

use std::sync::Arc;
use std::time::Duration;
use futures::stream::{self, Stream};
use observability_narration_core::n;

use crate::{JobError, JobRegistry, JobState};

// ============================================================================
// ⚠️  CRITICAL: WHY WE DON'T KEEP BACKWARDS COMPATIBILITY FUNCTIONS
// ============================================================================
//
// TEAM-312: This module previously had TWO functions:
//   1. execute_and_stream() - original without timeout
//   2. execute_and_stream_with_timeout() - enhanced with timeout
//
// This is a BACKWARDS COMPATIBILITY TRAP that creates PERMANENT TECHNICAL DEBT:
//
// ❌ PROBLEMS WITH KEEPING BOTH:
//   - Bugs must be fixed in TWO places (doubled maintenance)
//   - New features must be added to TWO functions (doubled work)
//   - Developers don't know which one to use (confusion)
//   - Code reviews must check BOTH functions (doubled review time)
//   - Tests must cover BOTH functions (doubled test surface)
//   - Documentation must explain BOTH (doubled docs)
//   - The old function becomes a PERMANENT ZOMBIE (never dies)
//
// ✅ SOLUTION: BREAKING CHANGES ARE TEMPORARY, ENTROPY IS FOREVER
//   - We DELETED the old function
//   - Compiler finds all call sites (30 seconds)
//   - We fix them by adding `timeout: None` (5 minutes)
//   - Done. No permanent debt.
//
// 📊 COST COMPARISON:
//   - Breaking change: 5 minutes of pain (ONE TIME)
//   - Backwards compat: 2x maintenance cost (FOREVER)
//
// Pre-1.0 software is ALLOWED to break. Use that freedom to avoid entropy.
//
// See: .windsurf/rules/engineering-rules.md (RULE ZERO)
// ============================================================================

/// Execute a job and stream its results with optional timeout and cancellation
///
/// TEAM-186: Reusable helper for deferred execution pattern
/// TEAM-305: Added timeout and cancellation support
/// TEAM-312: Renamed from execute_and_stream_with_timeout, deleted old execute_and_stream
///
/// # Type Parameters
/// - `T`: Token type for streaming (must implement ToString)
/// - `F`: Future that executes the job
/// - `Exec`: Function that creates the execution future
///
/// # Arguments
/// - `timeout`: Optional timeout duration. Pass `None` for no timeout.
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
    timeout: Option<Duration>,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
{
    // Retrieve payload and cancellation token
    let payload = registry.take_payload(&job_id);
    let cancellation_token = registry.get_cancellation_token(&job_id);

    if let Some(payload) = payload {
        let job_id_clone = job_id.clone();
        let registry_clone = registry.clone();
        
        tokio::spawn(async move {
            n!("execute", "Executing job {}", job_id_clone);

            // Execute with timeout and cancellation support using JobError
            let execution_future = executor(job_id_clone.clone(), payload);
            
            let result: Result<(), JobError> = if let Some(cancellation_token) = cancellation_token {
                // With cancellation support
                if let Some(timeout_duration) = timeout {
                    // With both timeout and cancellation
                    tokio::select! {
                        result = execution_future => result.map_err(JobError::from),
                        _ = cancellation_token.cancelled() => {
                            Err(JobError::Cancelled)
                        }
                        _ = tokio::time::sleep(timeout_duration) => {
                            Err(JobError::Timeout(timeout_duration))
                        }
                    }
                } else {
                    // With cancellation only
                    tokio::select! {
                        result = execution_future => result.map_err(JobError::from),
                        _ = cancellation_token.cancelled() => {
                            Err(JobError::Cancelled)
                        }
                    }
                }
            } else if let Some(timeout_duration) = timeout {
                // With timeout only
                match tokio::time::timeout(timeout_duration, execution_future).await {
                    Ok(result) => result.map_err(JobError::from),
                    Err(_) => Err(JobError::Timeout(timeout_duration)),
                }
            } else {
                // No timeout or cancellation
                execution_future.await.map_err(JobError::from)
            };
            
            // Update state based on JobError type
            match result {
                Ok(_) => {
                    registry_clone.update_state(&job_id_clone, JobState::Completed);
                }
                Err(JobError::Cancelled) => {
                    registry_clone.update_state(&job_id_clone, JobState::Cancelled);
                    n!("cancelled", "Job {} cancelled", job_id_clone);
                }
                Err(JobError::Timeout(duration)) => {
                    let error_msg = format!("Timeout after {:?}", duration);
                    registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
                    n!("timeout", "Job {} timed out: {}", job_id_clone, error_msg);
                }
                Err(JobError::ExecutionFailed(error_msg)) => {
                    registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
                    n!("failed", "Job {} failed: {}", job_id_clone, error_msg);
                }
            }
        });
    } else {
        n!("no_payload", "Warning: No payload found for job {}", job_id);
    }

    // Stream results and send [DONE], [ERROR], or [CANCELLED]
    let receiver = registry.take_token_receiver(&job_id);
    let registry_clone = registry.clone();
    let job_id_clone = job_id.clone();

    stream::unfold((receiver, false, job_id_clone, registry_clone), 
        |(rx_opt, done_sent, job_id, registry)| async move {
            if done_sent {
                return None;
            }

            match rx_opt {
                Some(mut rx) => match rx.recv().await {
                    Some(token) => {
                        let data = token.to_string();
                        Some((data, (Some(rx), false, job_id, registry)))
                    }
                    None => {
                        // Channel closed - check job state and send appropriate signal
                        let state = registry.get_job_state(&job_id);
                        let signal = match state {
                            Some(JobState::Failed(err)) => format!("[ERROR] {}", err),
                            Some(JobState::Cancelled) => "[CANCELLED]".to_string(),
                            _ => "[DONE]".to_string(),
                        };
                        Some((signal, (None, true, job_id, registry)))
                    }
                },
                None => {
                    // No receiver - check state and send appropriate signal
                    let state = registry.get_job_state(&job_id);
                    let signal = match state {
                        Some(JobState::Cancelled) => "[CANCELLED]".to_string(),
                        _ => "[DONE]".to_string(),
                    };
                    Some((signal, (None, true, job_id, registry)))
                }
            }
        }
    )
}
