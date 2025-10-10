// BDD runner for llama-orch system tests
// Created by: TEAM-040
// Modified by: TEAM-051 (added global queen-rbee instance)
// Modified by: TEAM-054 (added mock rbee-hive on port 9200)

mod steps;
mod mock_rbee_hive;

use cucumber::World as _;
use std::path::PathBuf;
use steps::world::World;

#[tokio::main]
async fn main() {
    // Enable BDD test mode
    std::env::set_var("LLORCH_BDD_MODE", "1");

    // Initialize tracing for debugging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,test_harness_bdd=debug".to_string()),
        )
        .init();

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features = match std::env::var("LLORCH_BDD_FEATURE_PATH").ok() {
        Some(p) => {
            let pb = PathBuf::from(&p);
            if pb.is_absolute() {
                pb
            } else {
                // If relative, resolve from workspace root, not crate root
                let workspace_root = root.parent().unwrap().parent().unwrap();
                workspace_root.join(pb)
            }
        }
        None => root.join("tests/features"),
    };

    tracing::info!("Running BDD tests from: {}", features.display());

    // TEAM-051: Start global queen-rbee instance before running tests
    steps::global_queen::start_global_queen_rbee().await;

    // TEAM-054: Start mock rbee-hive on port 9200 (NOT 8080 or 8090!)
    tokio::spawn(async {
        if let Err(e) = mock_rbee_hive::start_mock_rbee_hive().await {
            tracing::error!("Mock rbee-hive failed: {}", e);
        }
    });
    
    // Wait for mock servers to start
    // TEAM-058: Increased from 500ms to 1000ms per TEAM-057 recommendation for better reliability
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
    
    tracing::info!("✅ Mock servers ready:");
    tracing::info!("   - queen-rbee: http://127.0.0.1:8080");
    tracing::info!("   - rbee-hive:  http://127.0.0.1:9200");

    World::cucumber().fail_on_skipped().run_and_exit(features).await;
    
    // TEAM-051: Cleanup happens automatically via Drop in global_queen module
}
