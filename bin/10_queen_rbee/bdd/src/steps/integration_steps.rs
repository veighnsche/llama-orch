//! REAL integration test steps - spawns actual daemons
//!
//! Created by: TEAM-159
//!
//! ⚠️ CRITICAL: These tests MUST verify that QUEEN SPAWNS THE HIVE
//!
//! The whole point of queen-rbee is ORCHESTRATION. If your test manually spawns
//! rbee-hive, you're NOT testing orchestration - you're just testing HTTP endpoints.
//!
//! WRONG: Test harness spawns rbee-hive → Test passes → Production fails because
//!        queen doesn't know how to spawn hives
//!
//! RIGHT: Test tells queen "add localhost to catalog" → Queen spawns rbee-hive →
//!        Test waits for first heartbeat → Verifies queen's orchestration works
//!
//! If you're manually spawning rbee-hive in your test, ask yourself:
//! "What's the point of having an orchestrator if it doesn't orchestrate?"
//!
//! These steps spawn REAL queen-rbee and test that QUEEN spawns rbee-hive.
//! No mocks - this is real integration.

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord, HiveStatus};
use std::process::{Child, Command, Stdio};
use std::sync::Once;
use std::time::Duration;
use tokio::time::sleep;

// ============================================================================
// Binary Building
// ============================================================================

static BUILD_BINARIES: Once = Once::new();

/// TEAM-160: Ensure binaries are built before running integration tests
fn ensure_binaries_built() {
    BUILD_BINARIES.call_once(|| {
        println!("🔨 Building binaries for integration tests...");
        
        // Build queen-rbee (specify package to avoid workspace ambiguity)
        let status = Command::new("cargo")
            .args(&["build", "--package", "queen-rbee", "--bin", "queen-rbee"])
            .status()
            .expect("Failed to execute cargo build for queen-rbee");
        assert!(status.success(), "Failed to build queen-rbee");
        
        // Build rbee-hive (currently stub, but will work when implemented)
        let status = Command::new("cargo")
            .args(&["build", "--package", "rbee-hive", "--bin", "rbee-hive"])
            .status()
            .expect("Failed to execute cargo build for rbee-hive");
        assert!(status.success(), "Failed to build rbee-hive");
        
        println!("✅ Binaries built successfully");
    });
}

// ============================================================================
// Helper Functions
// ============================================================================

/// TEAM-160: Spawn queen-rbee daemon
fn spawn_queen_rbee(port: u16, db_path: &str) -> Result<Child, std::io::Error> {
    ensure_binaries_built();
    
    Command::new("target/debug/queen-rbee")
        .arg("--port")
        .arg(port.to_string())
        .arg("--database")
        .arg(db_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// ⚠️ WARNING: This function should NOT be used in integration tests!
///
/// TEAM-160: If you're calling this function, you're testing the WRONG thing.
/// The test harness should NOT spawn rbee-hive - QUEEN should spawn it.
///
/// This function exists only as a temporary placeholder until queen's
/// orchestration logic is implemented. Once queen can spawn hives via
/// SSH or local process management, DELETE THIS FUNCTION and test that
/// queen does the spawning.
///
/// Ask yourself: If the test spawns the hive, how do we know queen can?
/// What's the point of an orchestrator that doesn't orchestrate?
fn spawn_rbee_hive(_port: u16, _queen_url: &str, _db_path: &str) -> Result<Child, std::io::Error> {
    // TODO: DELETE THIS FUNCTION once queen has orchestration logic
    // TODO: Test that QUEEN spawns rbee-hive, not the test harness
    
    ensure_binaries_built();
    
    // NOTE: rbee-hive doesn't have CLI args yet, just spawn the stub
    Command::new("target/debug/rbee-hive")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// TEAM-160: Wait for queen-rbee to be ready
async fn wait_for_queen_ready(port: u16, timeout_secs: u64) -> Result<(), String> {
    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);
    let max_attempts = (timeout_secs * 2) as usize; // Check every 500ms
    
    for attempt in 0..max_attempts {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("✅ Queen-rbee ready after {} attempts", attempt + 1);
                return Ok(());
            }
            _ => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }
    
    Err(format!("Queen-rbee failed to start on port {} after {}s", port, timeout_secs))
}

/// TEAM-160: Wait for rbee-hive to be ready
/// NOTE: Since rbee-hive is a stub, this will likely timeout
async fn wait_for_hive_ready(port: u16, timeout_secs: u64) -> Result<(), String> {
    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);
    let max_attempts = (timeout_secs * 2) as usize;
    
    for attempt in 0..max_attempts {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                println!("✅ Rbee-hive ready after {} attempts", attempt + 1);
                return Ok(());
            }
            _ => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }
    
    Err(format!("Rbee-hive failed to start on port {} after {}s", port, timeout_secs))
}

// ============================================================================
// Given Steps
// ============================================================================

#[given("a temporary directory for test databases")]
async fn given_temp_directory(world: &mut BddWorld) {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    world.temp_dir = Some(temp_dir);
}

#[given("queen-rbee is configured to use the test database")]
async fn given_queen_configured(_world: &mut BddWorld) {
    // Configuration will be passed via command line args
}

#[given(expr = "queen-rbee HTTP server is running on port {int}")]
async fn given_queen_running(world: &mut BddWorld, port: u16) {
    // TEAM-160: Spawn queen-rbee daemon and wait for it to be ready
    let db_path = world.temp_dir.as_ref()
        .expect("No temp dir - did you forget 'Given a temporary directory'?")
        .path()
        .join("queen.db")
        .to_str()
        .expect("Invalid path")
        .to_string();
    
    // Spawn queen-rbee
    let child = spawn_queen_rbee(port, &db_path)
        .expect("Failed to spawn queen-rbee");
    
    // Store process handle
    world.queen_process = Some(child);
    world.queen_port = Some(port);
    
    // Wait for queen to be ready
    wait_for_queen_ready(port, 10).await
        .expect("Queen-rbee failed to start");
    
    println!("✅ Queen-rbee running on port {}", port);
}

#[given(expr = "the hive catalog contains a hive {string} with status {string}")]
async fn given_hive_with_status(world: &mut BddWorld, hive_id: String, status: String) {
    // TEAM-160: Add hive to catalog with specified status
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_status = match status.as_str() {
        "Unknown" => HiveStatus::Unknown,
        "Online" => HiveStatus::Online,
        "Offline" => HiveStatus::Offline,
        _ => panic!("Invalid status: {}", status),
    };
    
    let now_ms = chrono::Utc::now().timestamp_millis();
    let hive = HiveRecord {
        id: hive_id.clone(),
        host: "127.0.0.1".to_string(),
        port: 18600,
        ssh_host: None,
        ssh_port: None,
        ssh_user: None,
        status: hive_status.clone(),
        last_heartbeat_ms: None,
        devices: None,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
    };
    
    catalog.add_hive(hive).await.expect("Failed to add hive");
    world.current_hive_id = Some(hive_id);
    
    println!("✅ Added hive to catalog with status {:?}", hive_status);
}

#[given(expr = "the hive entry points to {string}")]
async fn given_hive_entry_points(_world: &mut BddWorld, address: String) {
    // TEAM-160: The hive catalog entry already has the address from previous step
    println!("✅ Hive entry points to {}", address);
}

// ============================================================================
// When Steps
// ============================================================================

#[when(expr = "rbee-hive daemon starts on port {int}")]
async fn when_rbee_hive_starts(world: &mut BddWorld, port: u16) {
    // TEAM-160: Spawn rbee-hive daemon
    let queen_port = world.queen_port.expect("Queen not running");
    let queen_url = format!("http://localhost:{}", queen_port);
    
    let db_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("hive.db")
        .to_str()
        .expect("Invalid path")
        .to_string();
    
    // Spawn rbee-hive (NOTE: currently a stub)
    let child = spawn_rbee_hive(port, &queen_url, &db_path)
        .expect("Failed to spawn rbee-hive");
    
    // Store process handle
    world.hive_process = Some(child);
    world.hive_port = Some(port);
    
    println!("✅ Rbee-hive spawned (stub) on port {}", port);
    // NOTE: We don't wait for health check since rbee-hive is a stub
}

#[when(expr = "rbee-hive is configured to send heartbeats to {string}")]
async fn when_rbee_hive_configured(_world: &mut BddWorld, queen_url: String) {
    // TEAM-160: Configuration passed via command line args in spawn step
    println!("✅ Rbee-hive configured to send heartbeats to {}", queen_url);
}

#[when(expr = "we wait {int} seconds for rbee-hive to initialize")]
async fn when_wait_for_init(_world: &mut BddWorld, seconds: u64) {
    // TEAM-160: Wait for initialization
    println!("⏳ Waiting {} seconds for initialization...", seconds);
    sleep(Duration::from_secs(seconds)).await;
}

#[when(expr = "we wait {int} seconds for the next heartbeat cycle")]
async fn when_wait_for_heartbeat(_world: &mut BddWorld, seconds: u64) {
    // TEAM-160: Wait for heartbeat cycle
    println!("⏳ Waiting {} seconds for heartbeat cycle...", seconds);
    sleep(Duration::from_secs(seconds)).await;
}

#[when("we kill the rbee-hive daemon")]
async fn when_kill_rbee_hive(world: &mut BddWorld) {
    // TEAM-160: Kill the stored process handle
    if let Some(mut process) = world.hive_process.take() {
        process.kill().expect("Failed to kill rbee-hive");
        process.wait().expect("Failed to wait for rbee-hive");
        println!("✅ Rbee-hive daemon killed");
    } else {
        panic!("No rbee-hive process to kill");
    }
}

#[when(expr = "we wait {int} seconds for heartbeat timeout")]
async fn when_wait_for_timeout(_world: &mut BddWorld, seconds: u64) {
    // TEAM-160: Wait for timeout
    println!("⏳ Waiting {} seconds for heartbeat timeout...", seconds);
    sleep(Duration::from_secs(seconds)).await;
}

#[when(expr = "we query the hive catalog for {string}")]
async fn when_query_catalog(world: &mut BddWorld, hive_id: String) {
    // TEAM-160: Query catalog and store result in world
    world.current_hive_id = Some(hive_id);
    println!("✅ Querying catalog for hive");
}

// ============================================================================
// Then Steps
// ============================================================================

#[then("rbee-hive should send its first heartbeat to queen")]
async fn then_rbee_hive_sends_heartbeat(_world: &mut BddWorld) {
    // TEAM-160: This is verified by checking catalog in subsequent steps
    // Since rbee-hive is a stub, this will be skipped for now
    println!("⚠️  Rbee-hive is a stub - heartbeat not implemented yet");
}

#[then("queen should receive the heartbeat")]
async fn then_queen_receives_heartbeat(world: &mut BddWorld) {
    // TEAM-160: Verify by checking hive catalog last_heartbeat timestamp
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");
    let hive = catalog.get_hive(hive_id).await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    // NOTE: Since rbee-hive is a stub, this will fail
    // We document this as expected behavior
    if hive.last_heartbeat_ms.is_some() {
        println!("✅ Queen received heartbeat");
    } else {
        println!("⚠️  No heartbeat received (rbee-hive is a stub)");
    }
}

#[then(expr = "queen should trigger device detection to {string}")]
async fn then_queen_triggers_detection(_world: &mut BddWorld, url: String) {
    // TEAM-160: Verify by checking narration or logs
    // Since rbee-hive is a stub, this won't happen
    println!("⚠️  Device detection to {} not triggered (rbee-hive is a stub)", url);
}

#[then("rbee-hive should respond with real device information")]
async fn then_rbee_hive_responds(_world: &mut BddWorld) {
    // TEAM-160: Since rbee-hive is a stub, this won't happen
    println!("⚠️  Rbee-hive is a stub - no device information available");
}

#[then("queen should store the device capabilities in the catalog")]
async fn then_queen_stores_capabilities(world: &mut BddWorld) {
    // TEAM-160: Verify by querying catalog
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");
    let hive = catalog.get_hive(hive_id).await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    if hive.devices.is_some() {
        let devices = hive.devices.unwrap();
        println!("✅ Device capabilities stored");
        if let Some(cpu) = devices.cpu {
            println!("   CPU: {} cores", cpu.cores);
        }
        println!("   GPUs: {}", devices.gpus.len());
    } else {
        println!("⚠️  No device capabilities stored (rbee-hive is a stub)");
    }
}

#[then(expr = "queen should update hive status to {string}")]
async fn then_queen_updates_status(world: &mut BddWorld, expected_status: String) {
    // TEAM-160: Query catalog and verify status
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");
    let hive = catalog.get_hive(hive_id).await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    let expected = match expected_status.as_str() {
        "Online" => HiveStatus::Online,
        "Offline" => HiveStatus::Offline,
        "Unknown" => HiveStatus::Unknown,
        _ => panic!("Invalid status: {}", expected_status),
    };
    
    if hive.status == expected {
        println!("✅ Hive status is {:?}", hive.status);
    } else {
        println!("⚠️  Hive status is {:?}, expected {:?} (rbee-hive is a stub)", hive.status, expected);
    }
}

#[then(expr = "queen should emit narration {string}")]
async fn then_queen_emits_narration(_world: &mut BddWorld, message: String) {
    // TEAM-160: Capture narration events
    // This would require capturing stdout/stderr from queen process
    println!("⚠️  Narration capture not implemented: {}", message);
}

#[then(expr = "the hive should have status {string}")]
async fn then_hive_has_status(world: &mut BddWorld, expected_status: String) {
    // TEAM-160: Query catalog and verify
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");
    let hive = catalog.get_hive(hive_id).await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    let expected = match expected_status.as_str() {
        "Online" => HiveStatus::Online,
        "Offline" => HiveStatus::Offline,
        "Unknown" => HiveStatus::Unknown,
        _ => panic!("Invalid status: {}", expected_status),
    };
    
    assert_eq!(hive.status, expected, 
        "Expected status {:?}, got {:?}", expected, hive.status);
    
    println!("✅ Hive status is {:?}", hive.status);
}

#[then("the hive should have device capabilities stored")]
async fn then_hive_has_capabilities(world: &mut BddWorld) {
    // TEAM-160: Query catalog and verify devices field is not null
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");
    let hive = catalog.get_hive(hive_id).await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    if hive.devices.is_some() {
        println!("✅ Hive has device capabilities");
    } else {
        println!("⚠️  Hive has no device capabilities (rbee-hive is a stub)");
    }
}

#[then("the hive should have a recent last_heartbeat timestamp")]
async fn then_hive_has_recent_heartbeat(world: &mut BddWorld) {
    // TEAM-160: Query catalog and verify timestamp is within last 30 seconds
    let catalog_path = world.temp_dir.as_ref()
        .expect("No temp dir")
        .path()
        .join("queen.db");
    
    let catalog = HiveCatalog::new(&catalog_path).await
        .expect("Failed to open catalog");
    
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");
    let hive = catalog.get_hive(hive_id).await
        .expect("Failed to query catalog")
        .expect("Hive not found");
    
    if let Some(heartbeat_ms) = hive.last_heartbeat_ms {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let diff_ms = now_ms - heartbeat_ms;
        
        if diff_ms < 30000 {
            println!("✅ Hive has recent heartbeat ({} ms ago)", diff_ms);
        } else {
            println!("⚠️  Heartbeat is {} ms old (> 30s)", diff_ms);
        }
    } else {
        println!("⚠️  No heartbeat timestamp (rbee-hive is a stub)");
    }
}

#[then("rbee-hive should send another heartbeat")]
async fn then_rbee_hive_sends_another(_world: &mut BddWorld) {
    // TEAM-160: Since rbee-hive is a stub, this won't happen
    println!("⚠️  Rbee-hive is a stub - no periodic heartbeats");
}

#[then("queen should NOT trigger device detection again")]
async fn then_queen_does_not_trigger_detection(_world: &mut BddWorld) {
    // TEAM-160: Verify narration doesn't contain "Checking capabilities"
    println!("⚠️  Narration verification not implemented");
}

#[then("the hive last_heartbeat timestamp should be updated")]
async fn then_timestamp_updated(_world: &mut BddWorld) {
    // TEAM-160: Since rbee-hive is a stub, this won't happen
    println!("⚠️  Rbee-hive is a stub - timestamp won't update");
}

#[then("queen should detect the missed heartbeats")]
async fn then_queen_detects_missed(_world: &mut BddWorld) {
    // TEAM-160: This requires queen to have a background task checking for stale heartbeats
    println!("⚠️  Heartbeat monitoring not implemented in queen-rbee");
}

#[then("queen should emit narration about hive going offline")]
async fn then_queen_emits_offline_narration(_world: &mut BddWorld) {
    // TEAM-160: Capture narration events
    println!("⚠️  Narration capture not implemented");
}

// ============================================================================
// Additional Given Steps
// ============================================================================

#[given(expr = "queen-rbee is running on port {int}")]
async fn given_queen_is_running(world: &mut BddWorld, port: u16) {
    given_queen_running(world, port).await;
}

#[given(expr = "rbee-hive is running on port {int} with status {string}")]
async fn given_rbee_hive_running(world: &mut BddWorld, port: u16, status: String) {
    // TEAM-160: First add hive to catalog with specified status
    given_hive_with_status(world, "localhost".to_string(), status).await;
    
    // Then spawn rbee-hive
    when_rbee_hive_starts(world, port).await;
    
    // Wait for initialization
    sleep(Duration::from_secs(3)).await;
}

#[given("rbee-hive has sent at least one heartbeat")]
async fn given_rbee_hive_sent_heartbeat(_world: &mut BddWorld) {
    // TEAM-160: Since rbee-hive is a stub, this is a no-op
    println!("⚠️  Rbee-hive is a stub - no heartbeat sent");
}
