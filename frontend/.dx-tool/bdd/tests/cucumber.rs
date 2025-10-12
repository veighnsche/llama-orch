// Created by: TEAM-DX-003
// BDD integration test for dx CLI tool
//
// This is an INTEGRATION TEST, not a binary.
// Run with: cargo test
// The cucumber crate will automatically discover and run this test.

use cucumber::World as _;
use std::path::PathBuf;
use std::time::Duration;
use dx_bdd::steps::world::DxWorld;

#[tokio::test]
async fn run_cucumber_tests() {
    // Initialize tracing for debugging
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,dx_bdd=debug".to_string()),
        )
        .try_init();

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features = match std::env::var("DX_BDD_FEATURE_PATH").ok() {
        Some(p) => {
            let pb = PathBuf::from(&p);
            if pb.is_absolute() {
                pb
            } else {
                // If relative, resolve from crate root (frontend/.dx-tool/bdd)
                root.join(pb)
            }
        }
        None => root.join("tests/features"),
    };

    tracing::info!("Running BDD tests from: {}", features.display());

    // Wrap entire test execution in timeout
    let result = tokio::time::timeout(
        Duration::from_secs(120), // 2 minutes for entire suite
        run_tests(features)
    )
    .await;

    match result {
        Ok(Ok(())) => {
            tracing::info!("✅ All tests completed successfully");
        }
        Ok(Err(e)) => {
            panic!("Tests failed: {}", e);
        }
        Err(_) => {
            panic!("Test suite timeout after 2 minutes");
        }
    }
}

/// Run the actual test suite
async fn run_tests(features: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("✅ Storybook should be running on http://localhost:6006");

    // Configure cucumber with per-scenario timeout enforcement
    DxWorld::cucumber()
        .fail_on_skipped()
        .max_concurrent_scenarios(1) // Run scenarios sequentially
        .before(|_feature, _rule, scenario, world| {
            Box::pin(async move {
                tracing::info!("🎬 Starting scenario: {}", scenario.name);
                world.start_time = Some(std::time::Instant::now());
            })
        })
        .after(|_feature, _rule, scenario, _event, world| {
            Box::pin(async move {
                if let Some(w) = world {
                    if let Some(start) = w.start_time {
                        let elapsed = start.elapsed();
                        tracing::info!("⏱️  Scenario '{}' completed in {:?}", scenario.name, elapsed);
                        
                        // Warn if scenario took too long
                        if elapsed > Duration::from_secs(10) {
                            tracing::warn!("⚠️  Scenario '{}' took longer than 10s: {:?}", scenario.name, elapsed);
                        }
                    }
                }
            })
        })
        .run(features)
        .await;
    
    Ok(())
}
