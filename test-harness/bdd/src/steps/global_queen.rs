// Global queen-rbee instance shared across all BDD scenarios
// Created by: TEAM-051
// Modified by: TEAM-061 (added startup timeout to prevent hangs)
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
        Self {
            process: std::sync::Mutex::new(Some(process)),
            url,
        }
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for GlobalQueenRbee {
    fn drop(&mut self) {
        // TEAM-051: Kill global queen-rbee when tests complete
        if let Some(mut proc) = self.process.lock().unwrap().take() {
            let _ = proc.start_kill();
            tracing::info!("🛑 Killed global queen-rbee process");
            
            // Wait for port to be released
            for i in 0..50 {
                std::thread::sleep(Duration::from_millis(100));
                if std::net::TcpStream::connect_timeout(
                    &"127.0.0.1:8080".parse().unwrap(),
                    Duration::from_millis(100)
                ).is_err() {
                    tracing::info!("✅ Port 8080 released after {}ms", (i + 1) * 100);
                    break;
                }
            }
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
                .stdout(std::process::Stdio::inherit())  // Print to console
                .stderr(std::process::Stdio::inherit())  // Print to console
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
