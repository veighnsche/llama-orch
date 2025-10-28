//! Timeout enforcement implementation (silent and with countdown)
//!
//! Created by: TEAM-163
//! Updated by: TEAM-330 (Universal context propagation)
//! Updated by: TEAM-330 (Countdown via narration events for SSE support)

use crate::enforcer::TimeoutEnforcer;
use anyhow::Result;
use observability_narration_core::n;
use std::future::Future;
use std::time::Duration;
use tokio::time::{interval, timeout};

impl TimeoutEnforcer {
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
    /// TEAM-330: Context propagates automatically - no manual job_id needed!
    /// Narration includes job_id from NarrationContext if available.
    pub(crate) async fn enforce_silent<F, T>(self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let label = self.label.clone().unwrap_or_else(|| "Operation".to_string());
        let total_secs = self.duration.as_secs();

        // TEAM-330: Simplified - n!() automatically includes context (job_id, correlation_id)
        // No need for manual with_narration_context() - it's already set by caller
        n!("start", "⏱️  {} (timeout: {}s)", label, total_secs);

        match timeout(self.duration, future).await {
            Ok(result) => result,
            Err(_) => {
                // TEAM-330: Timeout narration automatically includes context
                n!("timeout", "❌ {} TIMED OUT after {}s", label, total_secs);

                anyhow::bail!("{} timed out after {} seconds", label, total_secs)
            }
        }
    }

    /// Enforce timeout with countdown narration
    ///
    /// TEAM-330: Context propagates automatically!
    /// TEAM-330: Countdown goes through SSE as narration events (not stderr!)
    pub(crate) async fn enforce_with_countdown<F, T>(self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let label = self.label.clone().unwrap_or_else(|| "Operation".to_string());
        let total_secs = self.duration.as_secs();

        // TEAM-330: Simplified - context propagates automatically
        n!("start", "⏱️  {} (timeout: {}s)", label, total_secs);

        // TEAM-330: Spawn countdown narration task (goes through SSE!)
        // Instead of stderr progress bar, emit narration events every second
        let label_clone = label.clone();
        let progress_handle = tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(1));
            let mut elapsed = 0u64;

            loop {
                ticker.tick().await;
                elapsed += 1;

                // TEAM-330: Emit progress as narration (goes through SSE if job_id set!)
                n!("progress", "⏱️  {} - {}s / {}s elapsed", label_clone, elapsed, total_secs);

                if elapsed >= total_secs {
                    break;
                }
            }
        });

        // Run the operation with timeout
        let result = match timeout(self.duration, future).await {
            Ok(result) => {
                // Operation completed - stop countdown narration
                progress_handle.abort();
                result
            }
            Err(_) => {
                // Timeout occurred - stop countdown narration
                progress_handle.abort();

                // TEAM-330: Simplified - context propagates automatically
                n!("timeout", "❌ {} TIMED OUT after {}s", label, total_secs);

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
