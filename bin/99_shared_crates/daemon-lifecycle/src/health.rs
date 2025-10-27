//! Daemon health checking
//!
//! TEAM-259: Extracted from lib.rs for better organization
//! TEAM-328: Consolidated status.rs and get.rs into health.rs (RULE ZERO)
//!
//! Provides HTTP-based health checking for daemons.

use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;

// TEAM-328: Re-export status types from daemon-contract
pub use daemon_contract::{StatusRequest, StatusResponse};

/// Check if daemon is healthy by querying its health endpoint
///
/// # Arguments
/// * `base_url` - Base URL of daemon (e.g., "http://localhost:8500")
/// * `health_endpoint` - Optional health endpoint path (default: "/health")
/// * `timeout` - Optional timeout duration (default: 2 seconds)
///
/// # Returns
/// * `true` if daemon responds with 2xx status
/// * `false` if daemon is unreachable or returns error status
pub async fn is_daemon_healthy(
    base_url: &str,
    health_endpoint: Option<&str>,
    timeout: Option<Duration>,
) -> bool {
    let endpoint = health_endpoint.unwrap_or("/health");
    let timeout = timeout.unwrap_or(Duration::from_secs(2));

    let client = match Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(_) => return false,
    };

    let url = format!("{}{}", base_url, endpoint);

    match client.get(&url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Configuration for health polling with exponential backoff
///
/// TEAM-276: Added for daemon startup synchronization
pub struct HealthPollConfig {
    /// Base URL of daemon (e.g., "http://localhost:8500")
    pub base_url: String,

    /// Optional health endpoint path (default: "/health")
    pub health_endpoint: Option<String>,

    /// Maximum number of polling attempts (default: 10)
    pub max_attempts: usize,

    /// Initial delay in milliseconds (default: 200ms)
    pub initial_delay_ms: u64,

    /// Backoff multiplier for exponential backoff (default: 1.5)
    pub backoff_multiplier: f64,

    /// Optional job_id for narration routing
    pub job_id: Option<String>,

    /// Optional daemon name for narration (default: "daemon")
    pub daemon_name: Option<String>,
}

impl Default for HealthPollConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            health_endpoint: None,
            max_attempts: 10,
            initial_delay_ms: 200,
            backoff_multiplier: 1.5,
            job_id: None,
            daemon_name: None,
        }
    }
}

impl HealthPollConfig {
    /// Create a new config with just the base URL
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), ..Default::default() }
    }

    /// Set the health endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.health_endpoint = Some(endpoint.into());
        self
    }

    /// Set the maximum attempts
    pub fn with_max_attempts(mut self, attempts: usize) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Set the job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set the daemon name for narration
    pub fn with_daemon_name(mut self, name: impl Into<String>) -> Self {
        self.daemon_name = Some(name.into());
        self
    }
}

/// Poll daemon health endpoint until healthy or max attempts reached
///
/// TEAM-276: Added for daemon startup synchronization with exponential backoff
///
/// Uses exponential backoff:
/// - Attempt 1: 200ms
/// - Attempt 2: 300ms
/// - Attempt 3: 450ms
/// - Attempt 4: 675ms
/// - ...
///
/// # Arguments
/// * `config` - Health polling configuration
///
/// # Returns
/// * `Ok(())` - Daemon is healthy
/// * `Err` - Daemon failed to become healthy after max attempts
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{poll_until_healthy, HealthPollConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HealthPollConfig::new("http://localhost:8500")
///     .with_max_attempts(10)
///     .with_job_id("job-123");
///
/// poll_until_healthy(config).await?;
/// # Ok(())
/// # }
/// ```
#[with_job_id] // TEAM-328: Eliminates job_id context boilerplate
pub async fn poll_until_healthy(config: HealthPollConfig) -> anyhow::Result<()> {
    let daemon_name = config.daemon_name.as_deref().unwrap_or("daemon");

    // Emit start narration
    n!(
        "daemon_health_poll",
        "⏳ Waiting for {} to become healthy at {}",
        daemon_name,
        config.base_url
    );

    for attempt in 1..=config.max_attempts {
        // Check health
        if is_daemon_healthy(
            &config.base_url,
            config.health_endpoint.as_deref(),
            Some(Duration::from_secs(2)),
        )
        .await
        {
            // Success!
            n!("daemon_healthy", "✅ {} is healthy (attempt {})", daemon_name, attempt);
            return Ok(());
        }

        // Not healthy yet, calculate delay with exponential backoff
        if attempt < config.max_attempts {
            let delay_ms = (config.initial_delay_ms as f64
                * config.backoff_multiplier.powi((attempt - 1) as i32))
                as u64;
            let delay = Duration::from_millis(delay_ms);

            // Emit progress narration
            n!(
                "daemon_poll_retry",
                "🔄 Attempt {}/{}, retrying in {}ms...",
                attempt,
                config.max_attempts,
                delay_ms
            );

            sleep(delay).await;
        }
    }

    // Failed after max attempts
    n!(
        "daemon_not_healthy",
        "❌ {} failed to be come healthy after {} attempts",
        daemon_name,
        config.max_attempts
    );

    anyhow::bail!(
        "{} at {} failed to become healthy after {} attempts",
        daemon_name,
        config.base_url,
        config.max_attempts
    )
}

// TEAM-328: Renamed exports for consistent naming
/// Alias for is_daemon_healthy with consistent naming
pub use is_daemon_healthy as check_daemon_health;

/// Alias for poll_until_healthy with consistent naming  
pub use poll_until_healthy as poll_daemon_health;
