// BDD integration test for llama-orch system tests
// Created by: TEAM-040
// Modified by: TEAM-051 (added global queen-rbee instance)
// Modified by: TEAM-054 (added mock rbee-hive on port 9200)
// Modified by: TEAM-061 (added global timeout wrapper and signal handlers)
// Modified by: TEAM-063 (removed mock infrastructure)
// Modified by: TEAM-072 (added per-scenario timeout enforcement)
// Modified by: TEAM-076 (converted from binary to integration test)
//
// This is an INTEGRATION TEST, not a binary.
// Run with: cargo test
// The cucumber crate will automatically discover and run this test.

use cucumber::World as _;
use std::path::PathBuf;
use std::time::Duration;
use test_harness_bdd::steps::world::World;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[tokio::test]
async fn run_cucumber_tests() {
    // Enable BDD test mode
    std::env::set_var("LLORCH_BDD_MODE", "1");

    // Initialize tracing for debugging
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,test_harness_bdd=debug".to_string()),
        )
        .try_init();

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

    // Create scenario timeout watchdog
    let timeout_flag = Arc::new(AtomicBool::new(false));
    let timeout_flag_clone = timeout_flag.clone();
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();
    
    // Spawn watchdog that kills hung scenarios
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            
            // Exit watchdog if tests completed
            if shutdown_flag_clone.load(Ordering::Relaxed) {
                break;
            }
            
            if timeout_flag_clone.load(Ordering::Relaxed) {
                tracing::error!("❌ SCENARIO TIMEOUT DETECTED - KILLING PROCESSES");
                cleanup_all_processes();
                panic!("Scenario timeout exceeded");
            }
        }
    });
    
    // Wrap entire test execution in timeout
    let result = tokio::time::timeout(
        Duration::from_secs(300), // 5 minutes for entire suite
        run_tests(features, timeout_flag)
    )
    .await;
    
    // Signal watchdog to exit
    shutdown_flag.store(true, Ordering::Relaxed);

    // Explicit cleanup before exit
    test_harness_bdd::steps::global_queen::cleanup_global_queen();
    
    match result {
        Ok(Ok(())) => {
            tracing::info!("✅ All tests completed successfully");
        }
        Ok(Err(e)) => {
            cleanup_all_processes();
            panic!("Tests failed: {}", e);
        }
        Err(_) => {
            cleanup_all_processes();
            panic!("Test suite timeout after 5 minutes");
        }
    }
}

/// Run the actual test suite
async fn run_tests(features: PathBuf, timeout_flag: Arc<AtomicBool>) -> Result<(), Box<dyn std::error::Error>> {
    // Start global queen-rbee instance before running tests
    test_harness_bdd::steps::global_queen::start_global_queen_rbee().await;

    tracing::info!("✅ Real servers ready:");
    tracing::info!("   - queen-rbee: http://127.0.0.1:8080");

    // Configure cucumber with per-scenario timeout enforcement
    World::cucumber()
        .fail_on_skipped()
        .max_concurrent_scenarios(1) // Run scenarios sequentially to avoid port conflicts
        .before(move |_feature, _rule, scenario, world| {
            let timeout_flag = timeout_flag.clone();
            Box::pin(async move {
                tracing::info!("🎬 Starting scenario: {}", scenario.name);
                world.start_time = Some(std::time::Instant::now());
                
                // Spawn timeout watchdog for this scenario
                let scenario_name = scenario.name.clone();
                let timeout_flag_clone = timeout_flag.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    tracing::error!("❌ SCENARIO TIMEOUT: '{}' exceeded 60 seconds!", scenario_name);
                    timeout_flag_clone.store(true, Ordering::Relaxed);
                });
            })
        })
        .after(|_feature, _rule, scenario, _event, world| {
            Box::pin(async move {
                if let Some(w) = world {
                    if let Some(start) = w.start_time {
                        let elapsed = start.elapsed();
                        tracing::info!("⏱️  Scenario '{}' completed in {:?}", scenario.name, elapsed);
                        
                        // Warn if scenario took too long
                        if elapsed > Duration::from_secs(45) {
                            tracing::warn!("⚠️  Scenario '{}' took longer than 45s: {:?}", scenario.name, elapsed);
                        }
                    }
                }
            })
        })
        .run(features)
        .await;
    
    Ok(())
}

/// Cleanup all spawned processes
/// Called on panic or timeout
fn cleanup_all_processes() {
    tracing::info!("🧹 Cleaning up all test processes...");
    
    // Kill any remaining test processes by name
    let processes = ["bdd-runner", "mock-worker", "queen-rbee"];
    
    for proc_name in &processes {
        let _ = std::process::Command::new("pkill")
            .arg("-9")
            .arg("-f")
            .arg(proc_name)
            .output();
    }
    
    // Give processes time to die
    std::thread::sleep(Duration::from_millis(500));
    
    tracing::info!("✅ Cleanup complete");
}
