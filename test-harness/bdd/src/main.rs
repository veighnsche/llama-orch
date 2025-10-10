// BDD runner for llama-orch system tests
// Created by: TEAM-040

mod steps;

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

    World::cucumber().fail_on_skipped().run_and_exit(features).await;
}
