//! Handoff watcher for GPU nodes
//!
//! Watches `.runtime/engines/*.json` for handoff files written by engine-provisioner.
//! When detected, notifies callback to update pool registry.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// Handoff file payload written by engine-provisioner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffPayload {
    pub pool_id: String,
    pub replica_id: String,
    pub engine: String,
    pub engine_version: String,
    pub url: String,
    pub device_mask: Option<String>,
    pub slots: Option<u32>,
    pub pid: Option<u32>,
}

/// Callback invoked when handoff file is detected
pub type HandoffCallback = Box<dyn Fn(HandoffPayload) -> Result<()> + Send + Sync>;

/// Configuration for handoff watcher
#[derive(Debug, Clone)]
pub struct HandoffWatcherConfig {
    pub runtime_dir: PathBuf,
    pub poll_interval_ms: u64,
}

impl Default for HandoffWatcherConfig {
    fn default() -> Self {
        Self { runtime_dir: PathBuf::from(".runtime/engines"), poll_interval_ms: 1000 }
    }
}

/// Handoff watcher service
pub struct HandoffWatcher {
    config: HandoffWatcherConfig,
    callback: HandoffCallback,
    seen_files: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<PathBuf>>>,
}

impl HandoffWatcher {
    /// Create a new handoff watcher
    pub fn new(config: HandoffWatcherConfig, callback: HandoffCallback) -> Self {
        Self {
            config,
            callback,
            seen_files: std::sync::Arc::new(
                std::sync::Mutex::new(std::collections::HashSet::new()),
            ),
        }
    }

    /// Spawn the watcher task
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.run().await;
        })
    }

    /// Run the watcher loop
    async fn run(&self) {
        let mut ticker = interval(Duration::from_millis(self.config.poll_interval_ms));

        info!(
            runtime_dir = %self.config.runtime_dir.display(),
            poll_interval_ms = self.config.poll_interval_ms,
            "Starting handoff watcher"
        );

        loop {
            ticker.tick().await;

            if let Err(e) = self.check_for_handoffs().await {
                warn!(error = %e, "Error checking for handoff files");
            }
        }
    }

    /// Check for new handoff files
    async fn check_for_handoffs(&self) -> Result<()> {
        // Ensure directory exists
        if !self.config.runtime_dir.exists() {
            debug!("Runtime directory does not exist yet");
            return Ok(());
        }

        // Read directory
        let mut entries = tokio::fs::read_dir(&self.config.runtime_dir)
            .await
            .context("Failed to read runtime directory")?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only process .json files
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            // Skip if already seen
            {
                let seen = self.seen_files.lock().unwrap();
                if seen.contains(&path) {
                    continue;
                }
            }

            // Process handoff file
            if let Err(e) = self.process_handoff_file(&path).await {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "Failed to process handoff file"
                );
            } else {
                // Mark as seen
                let mut seen = self.seen_files.lock().unwrap();
                seen.insert(path);
            }
        }

        Ok(())
    }

    /// Process a single handoff file
    async fn process_handoff_file(&self, path: &Path) -> Result<()> {
        info!(path = %path.display(), "Processing handoff file");

        // Read file
        let content =
            tokio::fs::read_to_string(path).await.context("Failed to read handoff file")?;

        // Parse JSON
        let payload: HandoffPayload =
            serde_json::from_str(&content).context("Failed to parse handoff JSON")?;

        debug!(
            pool_id = %payload.pool_id,
            replica_id = %payload.replica_id,
            engine = %payload.engine,
            url = %payload.url,
            "Parsed handoff payload"
        );

        // Invoke callback
        (self.callback)(payload).context("Handoff callback failed")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn test_handoff_detection() {
        let temp_dir = tempfile::tempdir().unwrap();
        let runtime_dir = temp_dir.path().join("engines");
        tokio::fs::create_dir_all(&runtime_dir).await.unwrap();

        let detected = Arc::new(Mutex::new(Vec::new()));
        let detected_clone = detected.clone();

        let callback: HandoffCallback = Box::new(move |payload| {
            detected_clone.lock().unwrap().push(payload);
            Ok(())
        });

        let config =
            HandoffWatcherConfig { runtime_dir: runtime_dir.clone(), poll_interval_ms: 100 };

        let watcher = HandoffWatcher::new(config, callback);
        let handle = watcher.spawn();

        // Write handoff file
        let handoff = HandoffPayload {
            pool_id: "pool-0".to_string(),
            replica_id: "r0".to_string(),
            engine: "llamacpp".to_string(),
            engine_version: "b1234".to_string(),
            url: "http://localhost:8080".to_string(),
            device_mask: Some("0".to_string()),
            slots: Some(4),
            pid: Some(12345),
        };

        let handoff_path = runtime_dir.join("pool-0-r0.json");
        tokio::fs::write(&handoff_path, serde_json::to_string(&handoff).unwrap()).await.unwrap();

        // Wait for detection
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Check callback was invoked
        let detected_payloads = detected.lock().unwrap();
        assert_eq!(detected_payloads.len(), 1);
        assert_eq!(detected_payloads[0].pool_id, "pool-0");

        handle.abort();
    }
}
