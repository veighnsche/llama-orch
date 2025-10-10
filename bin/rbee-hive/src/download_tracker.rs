//! Download progress tracking with SSE streaming
//!
//! Per test-001-mvp.md Phase 3: Model Download Progress
//! Industry standard pattern from mistral.rs streaming.rs
//!
//! Created by: TEAM-034

use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Download event states (per test-001-mvp.md)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
pub enum DownloadEvent {
    #[serde(rename = "downloading")]
    Downloading {
        bytes_downloaded: u64,
        bytes_total: u64,
        speed_mbps: f64,
    },
    #[serde(rename = "complete")]
    Complete { local_path: String },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Download stream state (mistral.rs pattern)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    Running,
    SendingDone,
    Done,
}

/// Tracks download progress for multiple concurrent downloads
///
/// Uses broadcast channels for fan-out to multiple SSE subscribers.
/// Industry standard: 100 buffer size (mistral.rs pattern)
pub struct DownloadTracker {
    downloads: Arc<RwLock<HashMap<String, broadcast::Sender<DownloadEvent>>>>,
}

impl DownloadTracker {
    pub fn new() -> Self {
        Self {
            downloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start tracking a new download
    ///
    /// Returns a unique download ID for progress tracking
    pub async fn start_download(&self) -> String {
        let download_id = Uuid::new_v4().to_string();
        // Industry standard: 100 buffer size (mistral.rs uses this)
        let (tx, _rx) = broadcast::channel(100);

        self.downloads.write().await.insert(download_id.clone(), tx);
        download_id
    }

    /// Send progress update to all subscribers
    ///
    /// Ignores send errors (no active subscribers)
    pub async fn send_progress(&self, download_id: &str, event: DownloadEvent) -> Result<()> {
        let downloads = self.downloads.read().await;
        if let Some(tx) = downloads.get(download_id) {
            // Ignore send errors (no active subscribers)
            let _ = tx.send(event);
        }
        Ok(())
    }

    /// Subscribe to download progress
    ///
    /// Returns None if download ID not found
    pub async fn subscribe(&self, download_id: &str) -> Option<broadcast::Receiver<DownloadEvent>> {
        let downloads = self.downloads.read().await;
        downloads.get(download_id).map(|tx| tx.subscribe())
    }

    /// Complete and cleanup download
    pub async fn complete_download(&self, download_id: &str) {
        self.downloads.write().await.remove(download_id);
    }
}

impl Default for DownloadTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_download_tracker_start() {
        let tracker = DownloadTracker::new();
        let download_id = tracker.start_download().await;

        // Should be a valid UUID
        assert!(!download_id.is_empty());
        assert!(Uuid::parse_str(&download_id).is_ok());
    }

    #[tokio::test]
    async fn test_download_tracker_subscribe() {
        let tracker = DownloadTracker::new();
        let download_id = tracker.start_download().await;

        // Should be able to subscribe
        let rx = tracker.subscribe(&download_id).await;
        assert!(rx.is_some());

        // Non-existent ID should return None
        let rx = tracker.subscribe("non-existent").await;
        assert!(rx.is_none());
    }

    #[tokio::test]
    async fn test_download_tracker_send_progress() {
        let tracker = DownloadTracker::new();
        let download_id = tracker.start_download().await;

        let mut rx = tracker.subscribe(&download_id).await.unwrap();

        // Send progress event
        tracker
            .send_progress(
                &download_id,
                DownloadEvent::Downloading {
                    bytes_downloaded: 1024,
                    bytes_total: 2048,
                    speed_mbps: 10.0,
                },
            )
            .await
            .unwrap();

        // Receive event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, DownloadEvent::Downloading { .. }));
    }

    #[tokio::test]
    async fn test_download_tracker_multiple_subscribers() {
        let tracker = DownloadTracker::new();
        let download_id = tracker.start_download().await;

        let mut rx1 = tracker.subscribe(&download_id).await.unwrap();
        let mut rx2 = tracker.subscribe(&download_id).await.unwrap();

        // Send event
        tracker
            .send_progress(
                &download_id,
                DownloadEvent::Downloading {
                    bytes_downloaded: 1024,
                    bytes_total: 2048,
                    speed_mbps: 10.0,
                },
            )
            .await
            .unwrap();

        // Both subscribers should receive
        let event1 = rx1.recv().await.unwrap();
        let event2 = rx2.recv().await.unwrap();

        assert!(matches!(event1, DownloadEvent::Downloading { .. }));
        assert!(matches!(event2, DownloadEvent::Downloading { .. }));
    }

    #[tokio::test]
    async fn test_download_tracker_complete() {
        let tracker = DownloadTracker::new();
        let download_id = tracker.start_download().await;

        // Complete download
        tracker.complete_download(&download_id).await;

        // Should no longer be able to subscribe
        let rx = tracker.subscribe(&download_id).await;
        assert!(rx.is_none());
    }

    #[tokio::test]
    async fn test_download_event_serialization() {
        let event = DownloadEvent::Downloading {
            bytes_downloaded: 1024,
            bytes_total: 2048,
            speed_mbps: 10.5,
        };

        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["stage"], "downloading");
        assert_eq!(json["bytes_downloaded"], 1024);
        assert_eq!(json["bytes_total"], 2048);
        assert_eq!(json["speed_mbps"], 10.5);
    }

    #[tokio::test]
    async fn test_download_event_complete_serialization() {
        let event = DownloadEvent::Complete {
            local_path: "/models/test.gguf".to_string(),
        };

        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["stage"], "complete");
        assert_eq!(json["local_path"], "/models/test.gguf");
    }

    #[tokio::test]
    async fn test_download_event_error_serialization() {
        let event = DownloadEvent::Error {
            message: "Download failed".to_string(),
        };

        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["stage"], "error");
        assert_eq!(json["message"], "Download failed");
    }
}
