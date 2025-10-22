//! Timeout Enforcer - Hard timeout enforcement with visual countdown
//!
//! Created by: TEAM-163
//! Updated by: TEAM-197 (narration-core v0.5.0 migration)
//!
//! # Purpose
//! Prevents hanging operations by enforcing hard timeouts with visual feedback.
//! Every operation that could hang MUST use this crate.
//!
//! # Features
//! - Hard timeout enforcement (operation WILL fail after timeout)
//! - Visual countdown in terminal (shows remaining time)
//! - Clear error messages when timeout occurs
//! - Zero tolerance for hanging operations
//!
//! # Usage
//! ```no_run
//! use timeout_enforcer::TimeoutEnforcer;
//! use std::time::Duration;
//!
//! async fn my_operation() -> anyhow::Result<String> {
//!     // Your operation here
//!     Ok("done".to_string())
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let result = TimeoutEnforcer::new(Duration::from_secs(30))
//!         .with_label("Starting queen-rbee")
//!         .enforce(my_operation())
//!         .await?;
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use observability_narration_core::NarrationFactory;
use std::future::Future;
use std::time::Duration;
use tokio::time::{interval, timeout};

// TEAM-197: Migrated to narration-core v0.5.0 pattern
// Actor: "timeout" (7 chars, ≤10 limit)
const NARRATE: NarrationFactory = NarrationFactory::new("timeout");

/// Timeout enforcer with visual countdown feedback
///
/// # Example
/// ```no_run
/// use timeout_enforcer::TimeoutEnforcer;
/// use std::time::Duration;
///
/// async fn slow_operation() -> anyhow::Result<()> {
///     tokio::time::sleep(Duration::from_secs(5)).await;
///     Ok(())
/// }
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     TimeoutEnforcer::new(Duration::from_secs(10))
///         .with_label("Slow operation")
///         .enforce(slow_operation())
///         .await?;
///     Ok(())
/// }
/// ```
pub struct TimeoutEnforcer {
    duration: Duration,
    label: Option<String>,
    show_countdown: bool,
    job_id: Option<String>, // TEAM-207: For SSE routing
}

impl TimeoutEnforcer {
    /// Create a new timeout enforcer
    ///
    /// # Arguments
    /// * `duration` - Maximum time to wait before failing
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30));
    /// ```
    pub fn new(duration: Duration) -> Self {
        // TEAM-197: Countdown disabled by default to avoid interfering with narration output
        // The narration start/end messages provide sufficient feedback
        Self { duration, label: None, show_countdown: false, job_id: None }
    }

    /// Set a label for the operation (shown in countdown)
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .with_label("Starting queen-rbee");
    /// ```
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set job_id for SSE routing (server-side operations)
    ///
    /// TEAM-207: Required for timeout narration to flow through SSE channels.
    /// Without this, timeout events go to stdout and never reach the client.
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .with_label("Fetching data")
    ///     .with_job_id("job-123");
    /// ```
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Enable countdown display (visual feedback)
    ///
    /// Note: Countdown is disabled by default to avoid interfering with narration output.
    /// Only enable if you need the visual countdown feedback.
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .with_countdown();
    /// ```
    pub fn with_countdown(mut self) -> Self {
        self.show_countdown = true;
        self
    }

    /// Disable countdown display (silent mode)
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .silent();
    /// ```
    pub fn silent(mut self) -> Self {
        self.show_countdown = false;
        self
    }

    /// Enforce timeout on a future
    ///
    /// # Arguments
    /// * `future` - The async operation to enforce timeout on
    ///
    /// # Returns
    /// * `Ok(T)` - Operation completed successfully
    /// * `Err` - Operation timed out or failed
    ///
    /// # Example
    /// ```no_run
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// async fn my_op() -> anyhow::Result<String> {
    ///     Ok("done".to_string())
    /// }
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let result = TimeoutEnforcer::new(Duration::from_secs(30))
    ///         .enforce(my_op())
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn enforce<F, T>(self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        // TEAM-164: Auto-disable countdown when stderr is not a TTY
        // This fixes hangs when running via Command::output() which captures stderr to a pipe
        let is_tty = atty::is(atty::Stream::Stderr);
        let should_show_countdown = self.show_countdown && is_tty;

        if should_show_countdown {
            self.enforce_with_countdown(future).await
        } else {
            self.enforce_silent(future).await
        }
    }

    /// Enforce timeout silently (no countdown)
    ///
    /// TEAM-207: Still emits narration (start/timeout events), just no progress bar.
    /// This ensures timeout enforcement is visible in SSE streams even when running as daemon.
    async fn enforce_silent<F, T>(self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let label = self.label.clone().unwrap_or_else(|| "Operation".to_string());
        let total_secs = self.duration.as_secs();

        // TEAM-207: Emit start narration (no progress bar, just notification)
        let mut narration = NARRATE
            .action("start")
            .context(label.clone())
            .context(total_secs.to_string())
            .human("⏱️  {0} (timeout: {1}s)");

        if let Some(ref job_id) = self.job_id {
            narration = narration.job_id(job_id);
        }

        narration.emit();

        match timeout(self.duration, future).await {
            Ok(result) => result,
            Err(_) => {
                // TEAM-207: Emit timeout error narration
                let mut narration = NARRATE
                    .action("timeout")
                    .context(label.clone())
                    .context(total_secs.to_string())
                    .human("❌ {0} TIMED OUT after {1}s")
                    .error_kind("operation_timeout");

                if let Some(ref job_id) = self.job_id {
                    narration = narration.job_id(job_id);
                }

                narration.emit_error();

                anyhow::bail!("{} timed out after {} seconds", label, total_secs)
            }
        }
    }

    /// Enforce timeout with visual progress bar
    async fn enforce_with_countdown<F, T>(self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let label = self.label.clone().unwrap_or_else(|| "Operation".to_string());
        let total_secs = self.duration.as_secs();

        // TEAM-197: Use narration for start message
        // TEAM-207: Include job_id for SSE routing
        let mut narration = NARRATE
            .action("start")
            .context(label.clone())
            .context(total_secs.to_string())
            .human("⏱️  {0} (timeout: {1}s)");

        if let Some(ref job_id) = self.job_id {
            narration = narration.job_id(job_id);
        }

        narration.emit();

        // TEAM-197: Create progress bar that fills up over time
        let pb = ProgressBar::new(total_secs);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}s - {msg}")
                .expect("Invalid progress bar template")
                .progress_chars("█▓▒░ "),
        );
        pb.set_message(label.clone());

        // Spawn progress bar update task
        let pb_clone = pb.clone();
        let progress_handle = tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(1));
            let mut elapsed = 0u64;

            loop {
                ticker.tick().await;
                elapsed += 1;
                pb_clone.set_position(elapsed);

                if elapsed >= total_secs {
                    break;
                }
            }
        });

        // Run the operation with timeout
        let result = match timeout(self.duration, future).await {
            Ok(result) => {
                // Operation completed - stop progress bar
                progress_handle.abort();
                pb.finish_and_clear();
                result
            }
            Err(_) => {
                // Timeout occurred
                progress_handle.abort();
                pb.finish_and_clear();

                // TEAM-197: Use narration for timeout error
                // TEAM-207: Include job_id for SSE routing
                let mut narration = NARRATE
                    .action("timeout")
                    .context(label.clone())
                    .context(total_secs.to_string())
                    .human("❌ {0} TIMED OUT after {1}s")
                    .error_kind("operation_timeout");

                if let Some(ref job_id) = self.job_id {
                    narration = narration.job_id(job_id);
                }

                narration.emit_error();

                anyhow::bail!(
                    "{} timed out after {} seconds - operation was hanging",
                    label,
                    total_secs
                )
            }
        };

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_successful_operation() {
        async fn quick_op() -> Result<String> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok("success".to_string())
        }

        let result = TimeoutEnforcer::new(Duration::from_secs(1))
            .with_label("Quick operation")
            .silent()
            .enforce(quick_op())
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_timeout_occurs() {
        async fn slow_op() -> Result<String> {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok("should not reach here".to_string())
        }

        let result = TimeoutEnforcer::new(Duration::from_secs(1))
            .with_label("Slow operation")
            .silent()
            .enforce(slow_op())
            .await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("timed out"));
        assert!(err_msg.contains("1 second"));
    }

    #[tokio::test]
    async fn test_operation_failure() {
        async fn failing_op() -> Result<String> {
            anyhow::bail!("operation failed")
        }

        let result =
            TimeoutEnforcer::new(Duration::from_secs(1)).silent().enforce(failing_op()).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("operation failed"));
    }
}
