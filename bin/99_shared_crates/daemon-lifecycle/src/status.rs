//! Check daemon status on remote machine
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - None (uses reqwest directly for HTTP health checks)
//! - Provides check_daemon_health() for utils::poll::poll_daemon_health()
//!
//! # Requirements
//!
//! ## Input
//! - `health_url`: Health endpoint URL (e.g., "http://192.168.1.100:7835/health")
//!
//! ## Process
//! 1. HTTP GET to health endpoint (NO SSH)
//!    - Use: `reqwest::get(health_url)`
//!    - Timeout: 2 seconds
//!    - Return: true if 200 OK, false otherwise
//!
//! ## SSH Calls
//! - Total: 0 SSH calls (HTTP only)
//!
//! ## Error Handling
//! - Connection timeout → return false (daemon not running)
//! - Connection refused → return false (daemon not running)
//! - HTTP error → return false (daemon unhealthy)
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::check_daemon_status;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let is_running = check_daemon_status("http://192.168.1.100:7835/health").await?;
//! if is_running {
//!     println!("Daemon is running");
//! } else {
//!     println!("Daemon is not running");
//! }
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

/// Check daemon health
///
/// TEAM-330: RULE ZERO - One function, not two!
/// Used by utils/poll.rs and everywhere else
pub async fn check_daemon_health(health_url: &str) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    
    match client.get(health_url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}
