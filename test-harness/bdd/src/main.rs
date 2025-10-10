// BDD runner for llama-orch system tests
// Created by: TEAM-040
// Modified by: TEAM-051 (added global queen-rbee instance)
// Modified by: TEAM-054 (added mock rbee-hive on port 9200)
// Modified by: TEAM-061 (added global timeout wrapper and signal handlers)

mod steps;
mod mock_rbee_hive;

use cucumber::World as _;
use std::path::PathBuf;
use std::time::Duration;
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

    // TEAM-061: Set up panic handler to cleanup processes
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        eprintln!("💥 PANIC: {:?}", panic_info);
        eprintln!("🧹 Attempting cleanup before exit...");
        cleanup_all_processes();
        default_panic(panic_info);
    }));

    // TEAM-061: Spawn Ctrl+C handler for clean shutdown
    tokio::spawn(async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                tracing::warn!("🛑 Ctrl+C received, cleaning up...");
                cleanup_all_processes();
                std::process::exit(130); // Standard exit code for SIGINT
            }
            Err(e) => {
                tracing::error!("Failed to listen for Ctrl+C: {}", e);
            }
        }
    });

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

    // TEAM-061: Wrap entire test execution in timeout
    let result = tokio::time::timeout(
        Duration::from_secs(300), // 5 minutes for entire suite
        run_tests(features)
    )
    .await;

    match result {
        Ok(Ok(())) => {
            tracing::info!("✅ All tests completed successfully");
            std::process::exit(0);
        }
        Ok(Err(e)) => {
            tracing::error!("❌ Tests failed: {}", e);
            cleanup_all_processes();
            std::process::exit(1);
        }
        Err(_) => {
            tracing::error!("❌ Test suite timeout after 5 minutes");
            cleanup_all_processes();
            std::process::exit(124); // Standard timeout exit code
        }
    }
}

/// TEAM-061: Run the actual test suite
async fn run_tests(features: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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
    tokio::time::sleep(Duration::from_millis(1000)).await;
    
    tracing::info!("✅ Mock servers ready:");
    tracing::info!("   - queen-rbee: http://127.0.0.1:8080");
    tracing::info!("   - rbee-hive:  http://127.0.0.1:9200");

    World::cucumber().fail_on_skipped().run(features).await;
    
    // TEAM-051: Cleanup happens automatically via Drop in global_queen module
    Ok(())
}

/// TEAM-061: Cleanup all spawned processes
/// Called on panic, Ctrl+C, or timeout
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
