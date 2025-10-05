//! World state for worker-http BDD tests

use cucumber::World;

#[derive(Debug, Default, World)]
pub struct HttpWorld {
    // Server lifecycle
    pub port: Option<u16>,
    pub server_running: bool,
    pub shutdown_sent: bool,
    
    // SSE streaming
    pub has_sse_stream: bool,
    pub events_sent: usize,
    pub stream_closed: bool,
    
    // Request validation
    pub request_headers: Vec<String>,
    pub validation_passed: bool,
}
