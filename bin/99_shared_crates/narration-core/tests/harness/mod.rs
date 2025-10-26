// TEAM-302: Test harness for multi-service narration testing
//!
//! Test harness infrastructure for E2E testing of narration flows across service boundaries.
//!
//! # Purpose
//!
//! Provides reusable infrastructure for testing:
//! - Job creation and SSE channel management
//! - Narration event streaming
//! - Multi-service integration
//! - Concurrent job isolation
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::harness::NarrationTestHarness;
//!
//! let harness = NarrationTestHarness::start().await;
//! let job_id = harness.submit_job(operation).await;
//! let mut stream = harness.get_sse_stream(&job_id);
//! stream.assert_next("action", "message").await;
//! ```

use std::sync::Arc;
use job_server::JobRegistry;
use observability_narration_core::output::sse_sink::NarrationEvent;

pub mod sse_utils;

/// Test harness for multi-service narration testing
///
/// TEAM-302: Provides infrastructure for E2E testing of narration flows
pub struct NarrationTestHarness {
    registry: Arc<JobRegistry<String>>,
    base_url: String,
}

impl NarrationTestHarness {
    /// Start test harness with in-memory job registry
    ///
    /// TEAM-302: Creates isolated test environment
    pub async fn start() -> Self {
        let registry = Arc::new(JobRegistry::new());
        
        Self {
            registry,
            base_url: "http://localhost:8765".to_string(),
        }
    }
    
    /// Submit operation and create job
    ///
    /// TEAM-302: Creates job and SSE channel for testing
    pub async fn submit_job(&self, operation: serde_json::Value) -> String {
        // Create job via registry
        let job_id = self.registry.create_job();
        
        // Store operation payload
        self.registry.set_payload(&job_id, operation);
        
        // Create SSE channel for this job (1000 event capacity)
        observability_narration_core::output::sse_sink::create_job_channel(
            job_id.clone(),
            1000
        );
        
        job_id
    }
    
    /// Get SSE stream tester for job
    ///
    /// TEAM-302: Returns helper for testing SSE streams
    pub fn get_sse_stream(&self, job_id: &str) -> SSEStreamTester {
        let rx = observability_narration_core::output::sse_sink::take_job_receiver(job_id)
            .expect("Failed to get job receiver - was channel already taken?");
        
        SSEStreamTester::new(rx)
    }
    
    /// Get base URL for HTTP requests
    ///
    /// TEAM-302: For tests that need to make HTTP calls
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    
    /// Get job registry reference
    ///
    /// TEAM-302: For tests that need direct registry access
    pub fn registry(&self) -> &Arc<JobRegistry<String>> {
        &self.registry
    }
}

/// Helper for testing SSE streams
///
/// TEAM-302: Provides assertion helpers for SSE event testing
pub struct SSEStreamTester {
    receiver: tokio::sync::mpsc::Receiver<NarrationEvent>,
}

impl SSEStreamTester {
    /// Create new SSE stream tester
    ///
    /// TEAM-302: Wraps receiver with test helpers
    pub fn new(receiver: tokio::sync::mpsc::Receiver<NarrationEvent>) -> Self {
        Self { receiver }
    }
    
    /// Wait for next event (with timeout)
    ///
    /// TEAM-302: Returns None if timeout (5 seconds) or channel closed
    pub async fn next_event(&mut self) -> Option<NarrationEvent> {
        tokio::time::timeout(
            tokio::time::Duration::from_secs(5),
            self.receiver.recv()
        )
        .await
        .ok()
        .flatten()
    }
    
    /// Assert next event matches criteria
    ///
    /// TEAM-302: Convenience method for common assertion pattern
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - No event received (timeout or channel closed)
    /// - Action doesn't match
    /// - Message doesn't contain expected substring
    pub async fn assert_next(&mut self, action: &str, message_contains: &str) {
        let event = self.next_event().await
            .expect("Expected narration event but stream ended or timed out");
        
        assert_eq!(
            event.action, action,
            "Action mismatch: expected '{}', got '{}'",
            action, event.action
        );
        
        assert!(
            event.human.contains(message_contains),
            "Message '{}' doesn't contain '{}'",
            event.human,
            message_contains
        );
    }
    
    /// Collect all events until [DONE] or timeout
    ///
    /// TEAM-302: Useful for tests that need to verify event sequences
    pub async fn collect_until_done(&mut self) -> Vec<NarrationEvent> {
        let mut events = Vec::new();
        
        while let Some(event) = self.next_event().await {
            if event.human.contains("[DONE]") {
                break;
            }
            events.push(event);
        }
        
        events
    }
    
    /// Assert no more events (with short timeout)
    ///
    /// TEAM-302: Useful for verifying job isolation (no cross-contamination)
    ///
    /// # Panics
    ///
    /// Panics if an event is received within 100ms
    pub async fn assert_no_more_events(&mut self) {
        let result = tokio::time::timeout(
            tokio::time::Duration::from_millis(100),
            self.receiver.recv()
        ).await;
        
        assert!(
            result.is_err(),
            "Expected no more events, but received: {:?}",
            result.unwrap()
        );
    }
}
