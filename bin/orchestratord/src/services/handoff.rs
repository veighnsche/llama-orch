//! Handoff autobind watcher — monitors engine-provisioner handoff files and auto-binds adapters.
//!
//! TODO[ORCHD-HANDOFF-AUTOBIND-0002]: File watcher for `.runtime/engines/*.json` (handoffs written by engine-provisioner).
//! For each handoff: bind `llamacpp-http` adapter using `url`, associate `pool_id`/`replica_id`.
//! Update `state.pool_manager` via Owner E API to mark Ready and set engine meta.

use crate::state::AppState;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Spawn a background task that watches for engine handoff files and auto-binds adapters.
pub fn spawn_handoff_autobind_watcher(state: AppState) {
    tokio::spawn(async move {
        let runtime_dir = std::env::var("ORCHD_RUNTIME_DIR")
            .unwrap_or_else(|_| ".runtime/engines".to_string());
        let watch_interval_ms: u64 = std::env::var("ORCHD_HANDOFF_WATCH_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        info!(
            target: "orchestratord::handoff",
            runtime_dir=%runtime_dir,
            interval_ms=%watch_interval_ms,
            "handoff autobind watcher started"
        );

        loop {
            sleep(Duration::from_millis(watch_interval_ms)).await;
            
            let path = PathBuf::from(&runtime_dir);
            if !path.exists() {
                debug!(target: "orchestratord::handoff", "runtime dir does not exist yet: {}", runtime_dir);
                continue;
            }

            let entries = match std::fs::read_dir(&path) {
                Ok(e) => e,
                Err(err) => {
                    warn!(target: "orchestratord::handoff", "failed to read runtime dir: {}", err);
                    continue;
                }
            };

            for entry in entries.flatten() {
                let file_path = entry.path();
                if file_path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }

                if let Err(e) = process_handoff_file(&state, &file_path).await {
                    warn!(
                        target: "orchestratord::handoff",
                        file=?file_path,
                        error=%e,
                        "failed to process handoff"
                    );
                }
            }
        }
    });
}

async fn process_handoff_file(state: &AppState, path: &PathBuf) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let handoff: serde_json::Value = serde_json::from_str(&content)?;

    // Extract required fields
    let _url = handoff
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing 'url' in handoff"))?;
    let pool_id = handoff
        .get("pool_id")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let replica_id = handoff
        .get("replica_id")
        .and_then(|v| v.as_str())
        .unwrap_or("r0");

    // Check if already bound (idempotency)
    {
        let bound = state.bound_pools.lock().map_err(|_| anyhow::anyhow!("lock"))?;
        let key = format!("{}:{}", pool_id, replica_id);
        if bound.contains(&key) {
            debug!(
                target: "orchestratord::handoff",
                pool_id=%pool_id,
                replica_id=%replica_id,
                "already bound, skipping"
            );
            return Ok(());
        }
    }

    // Bind adapter
    #[cfg(feature = "llamacpp-adapter")]
    {
        let adapter = worker_adapters_llamacpp_http::LlamaCppHttpAdapter::new(url);
        state.adapter_host.bind(pool_id.to_string(), replica_id.to_string(), Arc::new(adapter));
        info!(
            target: "orchestratord::handoff",
            pool_id=%pool_id,
            replica_id=%replica_id,
            url=%url,
            "adapter bound"
        );
    }

    #[cfg(not(feature = "llamacpp-adapter"))]
    {
        debug!(
            target: "orchestratord::handoff",
            "llamacpp-adapter feature not enabled, skipping bind"
        );
    }

    // pool-managerd daemon now manages its own registry
    // No need to update from orchestratord side
    info!(
        target: "orchestratord::handoff",
        pool_id=%pool_id,
        replica_id=%replica_id,
        "handoff processed (daemon manages registry)"
    );

    // Mark as bound
    {
        let mut bound = state.bound_pools.lock().map_err(|_| anyhow::anyhow!("lock"))?;
        let key = format!("{}:{}", pool_id, replica_id);
        bound.insert(key);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use std::fs;
    use tempfile::TempDir;

    // ORCHD-AUTOBIND-UT-1001: given a handoff JSON on disk, the watcher binds an adapter and updates registry.
    #[tokio::test]
    async fn test_orchd_autobind_ut_1001_handoff_binding() {
        let state = AppState::new();
        let temp = TempDir::new().unwrap();
        let handoff_path = temp.path().join("test-handoff.json");

        let handoff = serde_json::json!({
            "url": "http://127.0.0.1:9999",
            "pool_id": "test-pool",
            "replica_id": "r1",
            "engine_version": "llamacpp-test-v1",
            "device_mask": "GPU0",
            "slots_total": 4,
            "slots_free": 4
        });
        fs::write(&handoff_path, serde_json::to_string(&handoff).unwrap()).unwrap();

        // Process handoff
        let result = process_handoff_file(&state, &handoff_path).await;
        assert!(result.is_ok(), "handoff processing failed: {:?}", result);

        // Verify registry updated
        {
            let reg = state.pool_manager.lock().unwrap();
            let health = reg.get_health("test-pool");
            assert!(health.is_some(), "pool not registered");
            let h = health.unwrap();
            assert!(h.live, "pool not live");
            assert!(h.ready, "pool not ready");
            assert_eq!(reg.get_engine_version("test-pool").as_deref(), Some("llamacpp-test-v1"));
            assert_eq!(reg.get_slots_total("test-pool"), Some(4));
            assert_eq!(reg.get_slots_free("test-pool"), Some(4));
        }

        // Verify bound_pools tracking
        {
            let bound = state.bound_pools.lock().unwrap();
            assert!(bound.contains("test-pool:r1"), "pool not marked as bound");
        }

        // Idempotency: process again should not error
        let result2 = process_handoff_file(&state, &handoff_path).await;
        assert!(result2.is_ok(), "idempotent call failed");
    }

    // ORCHD-AUTOBIND-UT-1002: partial handoff fields should not panic
    #[tokio::test]
    async fn test_orchd_autobind_ut_1002_partial_handoff() {
        let state = AppState::new();
        let temp = TempDir::new().unwrap();
        let handoff_path = temp.path().join("partial.json");

        // Minimal handoff with only url
        let handoff = serde_json::json!({
            "url": "http://127.0.0.1:8888"
        });
        fs::write(&handoff_path, serde_json::to_string(&handoff).unwrap()).unwrap();

        let result = process_handoff_file(&state, &handoff_path).await;
        assert!(result.is_ok(), "partial handoff should not panic");

        // Verify defaults applied
        {
            let reg = state.pool_manager.lock().unwrap();
            let health = reg.get_health("default");
            assert!(health.is_some());
            assert!(health.unwrap().ready);
        }
    }

    // ORCHD-AUTOBIND-UT-1003: missing url should error
    #[tokio::test]
    async fn test_orchd_autobind_ut_1003_missing_url_errors() {
        let state = AppState::new();
        let temp = TempDir::new().unwrap();
        let handoff_path = temp.path().join("no-url.json");

        let handoff = serde_json::json!({
            "pool_id": "test"
        });
        fs::write(&handoff_path, serde_json::to_string(&handoff).unwrap()).unwrap();

        let result = process_handoff_file(&state, &handoff_path).await;
        assert!(result.is_err(), "should error on missing url");
    }
}
