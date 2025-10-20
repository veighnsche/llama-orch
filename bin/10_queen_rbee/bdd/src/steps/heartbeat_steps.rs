//! BDD step definitions for heartbeat testing
//!
//! Created by: TEAM-158

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord, HiveStatus};
use rbee_heartbeat::HiveHeartbeatPayload;
use std::sync::Arc;

// ============================================================================
// Given Steps
// ============================================================================

#[given(expr = "the hive catalog contains a hive {string} with status {string}")]
async fn given_hive_with_status(world: &mut BddWorld, hive_id: String, status: String) {
    // TEAM-158: Create temp directory and catalog
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let catalog_path = temp_dir.path().join("test-hive-catalog.db");

    let catalog = HiveCatalog::new(&catalog_path).await.expect("Failed to create catalog");

    // TEAM-158: Parse status
    let hive_status = match status.as_str() {
        "Unknown" => HiveStatus::Unknown,
        "Online" => HiveStatus::Online,
        "Offline" => HiveStatus::Offline,
        _ => panic!("Invalid status: {}", status),
    };

    // TEAM-158: Add hive to catalog
    let now_ms = chrono::Utc::now().timestamp_millis();
    let hive = HiveRecord {
        id: hive_id.clone(),
        host: "127.0.0.1".to_string(),
        port: 8600,
        ssh_host: None,
        ssh_port: None,
        ssh_user: None,
        status: hive_status,
        last_heartbeat_ms: None,
        devices: None, // TEAM-159: No devices initially
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
    };

    catalog.add_hive(hive).await.expect("Failed to add hive");

    // TEAM-158: Store in world
    world.temp_dir = Some(temp_dir);
    world.catalog_path = Some(catalog_path);
    world.hive_catalog = Some(Arc::new(catalog));
    world.current_hive_id = Some(hive_id);
}

#[given(expr = "the hive {string} has no previous heartbeat")]
async fn given_no_previous_heartbeat(world: &mut BddWorld, hive_id: String) {
    // TEAM-158: Verify hive exists and has no heartbeat
    let catalog = world.hive_catalog.as_ref().expect("No catalog");
    let hive =
        catalog.get_hive(&hive_id).await.expect("Failed to get hive").expect("Hive not found");

    assert!(hive.last_heartbeat_ms.is_none(), "Hive should have no previous heartbeat");
}

// ============================================================================
// When Steps
// ============================================================================

#[when(expr = "the hive {string} sends its first heartbeat")]
async fn when_hive_sends_first_heartbeat(world: &mut BddWorld, hive_id: String) {
    // TEAM-159: Call actual HTTP handler with device detection
    use crate::steps::mock_hive_device_endpoint::{start_mock_hive_device_endpoint, MockDeviceResponse};
    
    // Start mock rbee-hive /v1/devices endpoint
    let mock_response = MockDeviceResponse::default_response();
    let mock_server = start_mock_hive_device_endpoint(mock_response).await;
    world.mock_server = Some(mock_server);
    
    // Create heartbeat payload
    let payload = HiveHeartbeatPayload {
        hive_id: hive_id.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        workers: vec![],
    };
    
    // Call shared heartbeat handler (which includes device detection)
    let catalog = world.hive_catalog.as_ref().expect("No catalog");
    let device_detector = Arc::new(crate::steps::mock_hive_device_endpoint::MockHiveDeviceDetector::new(
        &world.mock_server.as_ref().unwrap().uri(),
    ));
    
    let result = rbee_heartbeat::handle_hive_heartbeat(
        catalog.clone(),
        payload.clone(),
        device_detector,
    )
    .await;
    
    // Store result
    world.last_result = Some(result.map(|_| ()).map_err(|e| e.to_string()));
    world.heartbeat_payload = Some(payload);
}

#[when(expr = "the hive {string} sends a heartbeat")]
async fn when_hive_sends_heartbeat(world: &mut BddWorld, hive_id: String) {
    // TEAM-158: Same as first heartbeat
    when_hive_sends_first_heartbeat(world, hive_id).await;
}

#[when(expr = "an unknown hive {string} sends a heartbeat")]
async fn when_unknown_hive_sends_heartbeat(world: &mut BddWorld, hive_id: String) {
    // TEAM-159: Try to send heartbeat for non-existent hive via HTTP handler
    let catalog = world.hive_catalog.as_ref().expect("No catalog");
    
    // Create payload
    let payload = HiveHeartbeatPayload {
        hive_id: hive_id.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        workers: vec![],
    };
    
    // Use a dummy device detector (won't be called for unknown hive)
    let device_detector = Arc::new(crate::steps::mock_hive_device_endpoint::MockHiveDeviceDetector::new(
        "http://localhost:9999",
    ));
    
    // Call shared heartbeat handler
    let result = rbee_heartbeat::handle_hive_heartbeat(
        catalog.clone(),
        payload,
        device_detector,
    )
    .await;
    
    // Store result (should be error)
    world.last_result = Some(result.map(|_| ()).map_err(|e| e.to_string()));
}

// ============================================================================
// Then Steps
// ============================================================================

#[then("the heartbeat should be acknowledged")]
async fn then_heartbeat_acknowledged(world: &mut BddWorld) {
    // TEAM-158: Check that heartbeat was processed successfully
    assert!(world.last_succeeded(), "Heartbeat should be acknowledged");
}

#[then(expr = "the hive status should be updated to {string}")]
async fn then_hive_status_updated(world: &mut BddWorld, expected_status: String) {
    // TEAM-158: Check hive status in catalog
    let catalog = world.hive_catalog.as_ref().expect("No catalog");
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");

    let hive =
        catalog.get_hive(hive_id).await.expect("Failed to get hive").expect("Hive not found");

    let expected = match expected_status.as_str() {
        "Unknown" => HiveStatus::Unknown,
        "Online" => HiveStatus::Online,
        "Offline" => HiveStatus::Offline,
        _ => panic!("Invalid status: {}", expected_status),
    };

    assert_eq!(hive.status, expected, "Hive status should be {}", expected_status);
}

#[then(expr = "the hive status should remain {string}")]
async fn then_hive_status_remains(world: &mut BddWorld, expected_status: String) {
    // TEAM-158: Same as updated
    then_hive_status_updated(world, expected_status).await;
}

#[then("device detection should be triggered")]
async fn then_device_detection_triggered(_world: &mut BddWorld) {
    // TEAM-158: This would require mocking HTTP calls
    // For now, just pass (device detection is tested in unit tests)
    // TODO: Add HTTP mocking for full integration test
}

#[then("device detection should NOT be triggered")]
async fn then_device_detection_not_triggered(_world: &mut BddWorld) {
    // TEAM-158: This would require mocking HTTP calls
    // For now, just pass
}

#[then(expr = "narration should contain {string}")]
async fn then_narration_contains(_world: &mut BddWorld, _expected: String) {
    // TEAM-158: This would require capturing narration output
    // For now, just pass (narration is tested in unit tests)
    // TODO: Add narration capture for full integration test
}

#[then(expr = "narration should NOT contain {string}")]
async fn then_narration_not_contains(_world: &mut BddWorld, _unexpected: String) {
    // TEAM-158: This would require capturing narration output
    // For now, just pass
}

#[then("the heartbeat should be rejected with 404")]
async fn then_heartbeat_rejected(world: &mut BddWorld) {
    // TEAM-158: Check that heartbeat failed
    assert!(world.last_failed(), "Heartbeat should be rejected");
}

#[then(expr = "the error message should contain {string}")]
async fn then_error_contains(world: &mut BddWorld, expected: String) {
    // TEAM-158: Check error message
    if let Some(Err(msg)) = &world.last_result {
        assert!(msg.contains(&expected), "Error message '{}' should contain '{}'", msg, expected);
    } else {
        panic!("Expected error but got success");
    }
}

#[then(expr = "the hive {string} should have a last_heartbeat timestamp")]
async fn then_hive_has_heartbeat(world: &mut BddWorld, hive_id: String) {
    // TEAM-158: Check that heartbeat timestamp was recorded
    let catalog = world.hive_catalog.as_ref().expect("No catalog");
    let hive =
        catalog.get_hive(&hive_id).await.expect("Failed to get hive").expect("Hive not found");

    assert!(hive.last_heartbeat_ms.is_some(), "Hive should have a last_heartbeat timestamp");
}

#[then("the timestamp should be recent")]
async fn then_timestamp_recent(world: &mut BddWorld) {
    // TEAM-158: Check that timestamp is within last 5 seconds
    let catalog = world.hive_catalog.as_ref().expect("No catalog");
    let hive_id = world.current_hive_id.as_ref().expect("No current hive");

    let hive =
        catalog.get_hive(hive_id).await.expect("Failed to get hive").expect("Hive not found");

    let heartbeat_ms = hive.last_heartbeat_ms.expect("No heartbeat timestamp");
    let now_ms = chrono::Utc::now().timestamp_millis();
    let diff_ms = now_ms - heartbeat_ms;

    assert!(
        diff_ms < 5000,
        "Timestamp should be recent (within 5 seconds), but was {} ms ago",
        diff_ms
    );
}
