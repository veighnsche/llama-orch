//! Shutdown HTTP endpoint
//!
//! TEAM-339: Added POST /v1/shutdown endpoint for graceful shutdown

use axum::http::StatusCode;

/// POST /v1/shutdown - Gracefully shutdown queen-rbee
///
/// Returns 200 OK and then exits the process.
/// This allows clients (like rbee-keeper) to cleanly shut down the queen.
pub async fn handle_shutdown() -> StatusCode {
    // TEAM-339: Spawn shutdown in background to allow response to be sent
    tokio::spawn(async {
        // Give the HTTP response time to be sent (increased from 100ms to 500ms)
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        eprintln!("🛑 Shutdown requested via /v1/shutdown endpoint");
        eprintln!("🛑 Exiting process now...");
        
        // Use std::process::exit which immediately terminates
        std::process::exit(0);
    });
    
    StatusCode::OK
}
