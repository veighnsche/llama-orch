//! World state for worker-http BDD tests

use cucumber::World;
use worker_http::validation::ExecuteRequest;

#[derive(Debug, Default, World)]
pub struct HttpWorld {
    // Request validation
    pub request: Option<ExecuteRequest>,
    pub validation_error: Option<String>,
    pub validation_passed: bool,
    
    // SSE streaming
    pub event_count: usize,
    pub has_terminal_event: bool,
    pub event_types: Vec<String>,
    
    // Error handling
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}
