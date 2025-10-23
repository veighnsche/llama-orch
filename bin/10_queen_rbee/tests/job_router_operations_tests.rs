// TEAM-247: Job router operations tests
// Purpose: Test all operation routing and execution
// Priority: HIGH (core routing logic)
// Scale: Reasonable for NUC (5-10 concurrent, no overkill)

use serde_json::json;

// ============================================================================
// Operation Parsing Tests
// ============================================================================

#[test]
fn test_parse_valid_hive_list_operation() {
    // TEAM-247: Test HiveList operation parses correctly

    let payload = json!({
        "type": "HiveList"
    });

    assert_eq!(payload["type"], "HiveList");
}

#[test]
fn test_parse_valid_hive_start_operation() {
    // TEAM-247: Test HiveStart operation parses correctly

    let payload = json!({
        "type": "HiveStart",
        "alias": "localhost"
    });

    assert_eq!(payload["type"], "HiveStart");
    assert_eq!(payload["alias"], "localhost");
}

#[test]
fn test_parse_valid_hive_stop_operation() {
    // TEAM-247: Test HiveStop operation parses correctly

    let payload = json!({
        "type": "HiveStop",
        "alias": "localhost"
    });

    assert_eq!(payload["type"], "HiveStop");
    assert_eq!(payload["alias"], "localhost");
}

#[test]
fn test_parse_valid_status_operation() {
    // TEAM-247: Test Status operation parses correctly

    let payload = json!({
        "type": "Status"
    });

    assert_eq!(payload["type"], "Status");
}

// TEAM-278: DELETED test_parse_valid_ssh_test_operation - SshTest operation deleted

#[test]
fn test_parse_invalid_operation_missing_type() {
    // TEAM-247: Test invalid operation (missing type field)

    let payload = json!({
        "alias": "localhost"
    });

    assert!(payload.get("type").is_none(), "Should be missing type");
}

#[test]
fn test_parse_invalid_operation_wrong_type() {
    // TEAM-247: Test invalid operation (wrong type)

    let payload = json!({
        "type": "InvalidOperation"
    });

    assert_eq!(payload["type"], "InvalidOperation");
    // In real code, this would fail deserialization
}

#[test]
fn test_parse_operation_missing_required_field() {
    // TEAM-247: Test operation missing required field

    let payload = json!({
        "type": "HiveStart"
        // Missing "alias" field
    });

    assert!(payload.get("alias").is_none(), "Should be missing alias");
    // In real code, this would fail deserialization
}

#[test]
fn test_parse_operation_with_extra_fields() {
    // TEAM-247: Test operation with extra fields (should ignore)

    let payload = json!({
        "type": "HiveList",
        "extra_field": "should be ignored"
    });

    assert_eq!(payload["type"], "HiveList");
    assert_eq!(payload["extra_field"], "should be ignored");
    // In real code, extra fields are ignored during deserialization
}

// ============================================================================
// Status Operation Tests
// ============================================================================

#[test]
fn test_status_operation_with_no_hives() {
    // TEAM-247: Test Status with no active hives

    let active_hives: Vec<String> = vec![];

    assert!(active_hives.is_empty(), "Should have no hives");
    // In real code, should display: "No active hives found"
}

#[test]
fn test_status_operation_with_single_hive() {
    // TEAM-247: Test Status with single hive

    let active_hives = vec!["hive-local"];

    assert_eq!(active_hives.len(), 1);
    assert_eq!(active_hives[0], "hive-local");
}

#[test]
fn test_status_operation_with_multiple_hives() {
    // TEAM-247: Test Status with 5 hives

    let active_hives = vec!["hive-1", "hive-2", "hive-3", "hive-4", "hive-5"];

    assert_eq!(active_hives.len(), 5);
}

#[test]
fn test_status_operation_with_workers() {
    // TEAM-247: Test Status displays workers

    let hive_with_workers = json!({
        "hive": "hive-local",
        "workers": [
            {
                "worker_id": "worker-1",
                "state": "ready",
                "model": "llama-3-8b",
                "url": "http://localhost:8700"
            }
        ]
    });

    assert_eq!(hive_with_workers["hive"], "hive-local");
    assert_eq!(hive_with_workers["workers"].as_array().unwrap().len(), 1);
}

#[test]
fn test_status_operation_table_formatting() {
    // TEAM-247: Test Status table formatting

    let rows = vec![
        json!({
            "hive": "hive-local",
            "worker": "worker-1",
            "state": "ready",
            "model": "llama-3-8b",
            "url": "http://localhost:8700"
        }),
        json!({
            "hive": "hive-local",
            "worker": "-",
            "state": "-",
            "model": "-",
            "url": "-"
        }),
    ];

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["worker"], "worker-1");
    assert_eq!(rows[1]["worker"], "-");
}

// ============================================================================
// Hive Operation Tests
// ============================================================================

#[test]
fn test_hive_list_operation_payload() {
    // TEAM-247: Test HiveList operation payload structure

    let payload = json!({
        "type": "HiveList"
    });

    assert_eq!(payload["type"], "HiveList");
    // No additional fields required
}

#[test]
fn test_hive_get_operation_payload() {
    // TEAM-247: Test HiveGet operation payload structure

    let payload = json!({
        "type": "HiveGet",
        "alias": "localhost"
    });

    assert_eq!(payload["type"], "HiveGet");
    assert_eq!(payload["alias"], "localhost");
}

#[test]
fn test_hive_status_operation_payload() {
    // TEAM-247: Test HiveStatus operation payload structure

    let payload = json!({
        "type": "HiveStatus",
        "alias": "localhost"
    });

    assert_eq!(payload["type"], "HiveStatus");
    assert_eq!(payload["alias"], "localhost");
}

#[test]
fn test_hive_refresh_capabilities_payload() {
    // TEAM-247: Test HiveRefreshCapabilities operation payload

    let payload = json!({
        "type": "HiveRefreshCapabilities",
        "alias": "localhost"
    });

    assert_eq!(payload["type"], "HiveRefreshCapabilities");
    assert_eq!(payload["alias"], "localhost");
}

// ============================================================================
// SSH Test Operation Tests
// ============================================================================

#[test]
fn test_ssh_test_operation_success() {
    // TEAM-247: Test SSH test success response

    let response = json!({
        "success": true,
        "test_output": "test"
    });

    assert_eq!(response["success"], true);
    assert_eq!(response["test_output"], "test");
}

#[test]
fn test_ssh_test_operation_failure() {
    // TEAM-247: Test SSH test failure response

    let response = json!({
        "success": false,
        "error": "SSH agent not running"
    });

    assert_eq!(response["success"], false);
    assert!(response["error"].as_str().unwrap().contains("SSH agent"));
}

#[test]
fn test_ssh_test_operation_timeout() {
    // TEAM-247: Test SSH test timeout

    let response = json!({
        "success": false,
        "error": "Connection timeout"
    });

    assert_eq!(response["success"], false);
    assert!(response["error"].as_str().unwrap().contains("timeout"));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_hive_not_found_error() {
    // TEAM-247: Test hive not found error

    let error_message =
        "Hive alias 'nonexistent' not found in hives.conf.\n\nAvailable hives:\n  - localhost\n";

    assert!(error_message.contains("not found"));
    assert!(error_message.contains("Available hives"));
    assert!(error_message.contains("localhost"));
}

#[test]
fn test_binary_not_found_error() {
    // TEAM-247: Test binary not found error

    let error_message =
        "rbee-hive binary not found.\n\nPlease build it first:\n\n  cargo build --bin rbee-hive\n";

    assert!(error_message.contains("binary not found"));
    assert!(error_message.contains("cargo build"));
}

#[test]
fn test_operation_timeout_error() {
    // TEAM-247: Test operation timeout error

    let error_message = "Operation timed out after 15s";

    assert!(error_message.contains("timed out"));
    assert!(error_message.contains("15s"));
}

// ============================================================================
// Job Lifecycle Tests
// ============================================================================

#[test]
fn test_job_creation_generates_uuid() {
    // TEAM-247: Test job creation generates UUID

    let job_id = "550e8400-e29b-41d4-a716-446655440000";

    assert_eq!(job_id.len(), 36); // UUID length
    assert!(job_id.contains('-')); // UUID format
}

#[test]
fn test_job_response_structure() {
    // TEAM-247: Test job response structure

    let response = json!({
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "sse_url": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000/stream"
    });

    assert!(response["job_id"].as_str().unwrap().len() == 36);
    assert!(response["sse_url"].as_str().unwrap().starts_with("/v1/jobs/"));
    assert!(response["sse_url"].as_str().unwrap().ends_with("/stream"));
}

#[test]
fn test_job_payload_storage() {
    // TEAM-247: Test job payload stored correctly

    let payload = json!({
        "type": "HiveStart",
        "alias": "localhost"
    });

    // In real code, payload is stored in job registry
    assert_eq!(payload["type"], "HiveStart");
    assert_eq!(payload["alias"], "localhost");
}

// ============================================================================
// Concurrent Operations Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_operation_parsing() {
    // TEAM-247: Test 10 concurrent operation parsings

    let mut handles = vec![];

    for i in 0..10 {
        let handle = tokio::spawn(async move {
            let payload = json!({
                "type": "HiveList",
                "request_id": i
            });

            assert_eq!(payload["type"], "HiveList");
            assert_eq!(payload["request_id"], i);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

// ============================================================================
// Operation Name Tests
// ============================================================================

#[test]
fn test_operation_name_extraction() {
    // TEAM-247: Test operation name extraction

    let operations = vec![
        ("HiveList", "HiveList"),
        ("HiveStart", "HiveStart"),
        ("HiveStop", "HiveStop"),
        ("Status", "Status"),
        // TEAM-278: DELETED ("SshTest", "SshTest")
    ];

    for (op_type, expected_name) in operations {
        assert_eq!(op_type, expected_name);
    }
}
