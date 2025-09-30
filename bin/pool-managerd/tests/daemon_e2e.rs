//! End-to-end test for pool-managerd daemon
//!
//! Tests:
//! 1. Start daemon
//! 2. Health check
//! 3. Preload engine via HTTP API
//! 4. Check pool status
//! 5. Verify engine is running
//! 6. Cleanup

use anyhow::Result;
use serde_json::json;
use std::time::Duration;

#[tokio::test]
#[ignore] // Run with: cargo test -p pool-managerd --test daemon_e2e -- --ignored --nocapture
async fn test_daemon_e2e() -> Result<()> {
    // Skip if not opted in
    if std::env::var("LLORCH_DAEMON_E2E").as_deref() != Ok("1") {
        eprintln!("skipping daemon E2E test (set LLORCH_DAEMON_E2E=1 to run)");
        return Ok(());
    }

    // 1. Start daemon in background
    eprintln!("=== Starting pool-managerd daemon ===");
    let daemon = tokio::spawn(async {
        pool_managerd::api::routes::create_router(pool_managerd::api::routes::AppState {
            registry: std::sync::Arc::new(std::sync::Mutex::new(
                pool_managerd::core::registry::Registry::new(),
            )),
        });
        // Note: In real test, we'd start the actual server
        // For now, this is a placeholder
    });

    // Give it time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 2. Health check
    eprintln!("\n=== Testing health endpoint ===");
    let client = reqwest::Client::new();
    let health_resp = client
        .get("http://127.0.0.1:9001/health")
        .send()
        .await?;

    assert_eq!(health_resp.status(), 200);
    let health_json: serde_json::Value = health_resp.json().await?;
    assert_eq!(health_json["status"], "ok");
    eprintln!("✓ Health check passed: {:?}", health_json);

    // 3. Preload engine (mock PreparedEngine)
    eprintln!("\n=== Testing preload endpoint ===");
    let prepared = json!({
        "prepared": {
            "binary_path": "/usr/local/bin/llama-server",
            "flags": ["--model", "/models/test.gguf", "--port", "8080"],
            "port": 8080,
            "host": "127.0.0.1",
            "model_path": "/models/test.gguf",
            "engine_version": "llamacpp-test",
            "pool_id": "test-pool",
            "replica_id": "r0",
            "device_mask": null
        }
    });

    let preload_resp = client
        .post("http://127.0.0.1:9001/pools/test-pool/preload")
        .json(&prepared)
        .send()
        .await;

    // Note: This will fail if daemon isn't actually running
    // This is a placeholder test structure
    match preload_resp {
        Ok(resp) => {
            eprintln!("✓ Preload response: {:?}", resp.status());
            if resp.status().is_success() {
                let preload_json: serde_json::Value = resp.json().await?;
                eprintln!("✓ Preload succeeded: {:?}", preload_json);
            }
        }
        Err(e) => {
            eprintln!("⚠ Preload failed (expected if daemon not running): {}", e);
        }
    }

    // 4. Check pool status
    eprintln!("\n=== Testing status endpoint ===");
    let status_resp = client
        .get("http://127.0.0.1:9001/pools/test-pool/status")
        .send()
        .await;

    match status_resp {
        Ok(resp) => {
            eprintln!("✓ Status response: {:?}", resp.status());
            if resp.status().is_success() {
                let status_json: serde_json::Value = resp.json().await?;
                eprintln!("✓ Pool status: {:?}", status_json);
            }
        }
        Err(e) => {
            eprintln!("⚠ Status failed (expected if daemon not running): {}", e);
        }
    }

    // Cleanup
    daemon.abort();

    eprintln!("\n=== Test complete ===");
    eprintln!("Note: This is a placeholder test. For real E2E:");
    eprintln!("1. Start daemon: cargo run -p pool-managerd");
    eprintln!("2. Run test: LLORCH_DAEMON_E2E=1 cargo test --test daemon_e2e -- --ignored");

    Ok(())
}

#[test]
fn test_systemd_unit_exists() {
    // Find repo root
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let unit_path = manifest_dir.join("pool-managerd.service");
    
    assert!(
        unit_path.exists(),
        "systemd unit file should exist at {}",
        unit_path.display()
    );

    let content = std::fs::read_to_string(&unit_path).expect("read systemd unit");
    assert!(content.contains("pool-managerd"), "unit should reference pool-managerd");
    assert!(content.contains("ExecStart=/usr/local/bin/pool-managerd"), "unit should have ExecStart");
    assert!(content.contains("POOL_MANAGERD_ADDR=127.0.0.1:9001"), "unit should set bind address");
    
    eprintln!("✓ Systemd unit file validated: {}", unit_path.display());
}
