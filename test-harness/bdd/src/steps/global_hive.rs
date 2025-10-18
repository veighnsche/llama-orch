//! Global rbee-hive instance management for BDD tests
//!
//! TEAM-085: Created to support localhost inference tests
//!
//! This module manages a single rbee-hive process that is started on-demand
//! when queen-rbee needs to route localhost inference requests.
//!
//! Architecture:
//! - queen-rbee (port 8080) - Global instance, always running
//! - rbee-hive (port 9200) - Started on-demand for localhost tests
//! - workers (port 8001+) - Spawned by rbee-hive as needed

use std::sync::OnceLock;
use std::time::Duration;
use tokio::process::Child;
use tokio::time::sleep;

static GLOBAL_HIVE: OnceLock<GlobalRbeeHive> = OnceLock::new();

pub struct GlobalRbeeHive {
    process: std::sync::Mutex<Option<Child>>,
    url: String,
}

impl GlobalRbeeHive {
    fn new(process: Child, url: String) -> Self {
        Self { process: std::sync::Mutex::new(Some(process)), url }
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for GlobalRbeeHive {
    fn drop(&mut self) {
        if let Some(mut proc) = self.process.lock().unwrap().take() {
            let _ = proc.start_kill();
            tracing::info!("🛑 Killed global rbee-hive process");
        }
    }
}

/// Start the global rbee-hive instance
/// This is called on-demand when localhost inference tests need it
pub async fn start_global_rbee_hive() {
    // Check if already initialized
    if GLOBAL_HIVE.get().is_some() {
        tracing::info!("✅ Global rbee-hive already initialized");
        return;
    }

    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));

    let binary_path = workspace_dir.join("target/debug/rbee-hive");

    tracing::info!("🐝 Starting GLOBAL rbee-hive process at {:?}...", binary_path);

    let child = {
        let mut child = tokio::process::Command::new(&binary_path)
            .args(["daemon", "--addr", "127.0.0.1:9200"])
            .env("RBEE_WORKER_HOST", "127.0.0.1")
            .current_dir(&workspace_dir)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("Failed to start global rbee-hive");

        // Wait for server to be ready with 30s timeout
        let client = crate::steps::world::create_http_client();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(30);

        for i in 0..300 {
            if start.elapsed() > timeout {
                let _ = child.start_kill();
                panic!("❌ Global rbee-hive failed to start within 30s - timeout exceeded");
            }

            // Check if process is still alive
            match child.try_wait() {
                Ok(Some(status)) => {
                    panic!("Global rbee-hive exited during startup with status: {} (likely port 9200 already in use)", status);
                }
                Ok(None) => {
                    // Process still running - good
                }
                Err(e) => {
                    tracing::warn!("Failed to check process status: {}", e);
                }
            }

            if let Ok(resp) = client.get("http://127.0.0.1:9200/v1/health").send().await {
                if resp.status().is_success() {
                    tracing::info!("✅ Global rbee-hive is ready (took {:?})", start.elapsed());
                    break;
                }
            }

            if i % 10 == 0 && i > 0 {
                tracing::info!("⏳ Waiting for global rbee-hive... ({:?})", start.elapsed());
            }
            sleep(Duration::from_millis(100)).await;
        }

        child
    };

    let hive = GlobalRbeeHive::new(child, "http://127.0.0.1:9200".to_string());

    let _ = GLOBAL_HIVE.set(hive);

    tracing::info!("✅ Global rbee-hive available at: http://127.0.0.1:9200");
}

/// Get the URL of the global rbee-hive instance
pub fn get_global_hive_url() -> Option<String> {
    GLOBAL_HIVE.get().map(|h| h.url().to_string())
}

/// Cleanup global rbee-hive before exit
pub fn cleanup_global_hive() {
    if let Some(hive) = GLOBAL_HIVE.get() {
        if let Some(mut proc) = hive.process.lock().unwrap().take() {
            let _ = proc.start_kill();
            tracing::info!("🛑 Force-killed global rbee-hive before exit");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
