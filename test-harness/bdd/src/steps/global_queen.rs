// Global queen-rbee instance management
// Created by: TEAM-051
// Modified by: TEAM-061 (added startup timeout to prevent hangs)
// Modified by: TEAM-064 (added explicit warning preservation notice)
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// This module manages a single queen-rbee process that is started once
// before all tests and cleaned up at the end. This prevents port conflicts
// and improves test performance.
use std::sync::OnceLock;
use std::time::Duration;
use tokio::process::Child;
use tokio::time::sleep;

static GLOBAL_QUEEN: OnceLock<GlobalQueenRbee> = OnceLock::new();

pub struct GlobalQueenRbee {
    process: std::sync::Mutex<Option<Child>>,
    url: String,
}

impl GlobalQueenRbee {
    fn new(process: Child, url: String) -> Self {
        Self { process: std::sync::Mutex::new(Some(process)), url }
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for GlobalQueenRbee {
    fn drop(&mut self) {
        // TEAM-074: Simplified drop - just kill without waiting for port
        // Port waiting causes hang on exit - process cleanup handled by OS
        if let Some(mut proc) = self.process.lock().unwrap().take() {
            let _ = proc.start_kill();
            tracing::info!("🛑 Killed global queen-rbee process");
        }
    }
}

/// Start the global queen-rbee instance
/// This should be called once before running any tests
pub async fn start_global_queen_rbee() {
    // TEAM-051: Initialize the global instance if not already done
    if GLOBAL_QUEEN.get().is_some() {
        tracing::info!("✅ Global queen-rbee already initialized");
        return;
    }

    // Create temp directory for test database
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("global_test_beehives.db");

    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));

    let binary_path = workspace_dir.join("target/debug/queen-rbee");

    tracing::info!("🐝 Starting GLOBAL queen-rbee process at {:?}...", binary_path);

    let child = {
        // TEAM-058: Changed to inherit stdio to see panic messages
        let mut child = tokio::process::Command::new(&binary_path)
            .args(["--port", "8080", "--database"])
            .arg(&db_path)
            .env("MOCK_SSH", "true")
            .current_dir(&workspace_dir)
            .stdout(std::process::Stdio::inherit()) // Print to console
            .stderr(std::process::Stdio::inherit()) // Print to console
            .spawn()
            .expect("Failed to start global queen-rbee");

        // TEAM-061: Wait for server to be ready with 30s timeout
        let client = crate::steps::world::create_http_client();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(30);

        for i in 0..300 {
            // TEAM-061: Check timeout first
            if start.elapsed() > timeout {
                let _ = child.start_kill();
                panic!("❌ Global queen-rbee failed to start within 30s - timeout exceeded");
            }

            // Check if process is still alive
            match child.try_wait() {
                Ok(Some(status)) => {
                    panic!("Global queen-rbee exited during startup with status: {} (likely port 8080 already in use)", status);
                }
                Ok(None) => {
                    // Process still running - good
                }
                Err(e) => {
                    tracing::warn!("Failed to check process status: {}", e);
                }
            }

            if let Ok(resp) = client.get("http://localhost:8080/health").send().await {
                if resp.status().is_success() {
                    tracing::info!("✅ Global queen-rbee is ready (took {:?})", start.elapsed());
                    break;
                }
            }

            if i % 10 == 0 && i > 0 {
                tracing::info!("⏳ Waiting for global queen-rbee... ({:?})", start.elapsed());
            }
            sleep(Duration::from_millis(100)).await;
        }

        child
    };

    // Keep temp_dir alive for the duration of the tests
    std::mem::forget(temp_dir);

    let queen = GlobalQueenRbee::new(child, "http://localhost:8080".to_string());

    let _ = GLOBAL_QUEEN.set(queen);

    tracing::info!("✅ Global queen-rbee available at: http://localhost:8080");
}

/// Get the URL of the global queen-rbee instance
pub fn get_global_queen_url() -> Option<String> {
    GLOBAL_QUEEN.get().map(|q| q.url().to_string())
}

/// TEAM-074: Explicit cleanup before exit to prevent Drop hang
/// Forcefully kills queen-rbee without waiting for port release
pub fn cleanup_global_queen() {
    if let Some(queen) = GLOBAL_QUEEN.get() {
        if let Some(mut proc) = queen.process.lock().unwrap().take() {
            // SIGKILL (-9) for immediate termination
            let _ = proc.start_kill();
            tracing::info!("🛑 Force-killed global queen-rbee before exit");
            // Brief sleep to allow kill signal to send
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
