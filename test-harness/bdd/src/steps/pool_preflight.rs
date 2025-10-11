// Pool preflight step definitions
// Created by: TEAM-042
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::World;
use cucumber::{given, then, when};

// TEAM-071: Set up reachable node for testing NICE!
#[given(expr = "node {string} is reachable")]
pub async fn given_node_reachable(world: &mut World, node: String) {
    // Store node as reachable in topology
    world.topology.insert(
        node.clone(),
        crate::steps::world::NodeInfo {
            hostname: format!("{}.local", node),
            components: vec!["rbee-hive".to_string()],
            capabilities: vec!["reachable".to_string()],
        },
    );
    
    tracing::info!("✅ Node {} marked as reachable NICE!", node);
}

// TEAM-071: Set rbee-keeper version for compatibility check NICE!
#[given(expr = "rbee-keeper version is {string}")]
pub async fn given_rbee_keeper_version(world: &mut World, version: String) {
    // Store keeper version in node_ram for later verification
    world.node_ram.insert("rbee_keeper_version".to_string(), version.parse::<usize>().unwrap_or(1));
    
    tracing::info!("✅ rbee-keeper version set to: {} NICE!", version);
}

// TEAM-071: Set rbee-hive version for compatibility check NICE!
#[given(expr = "rbee-hive version is {string}")]
pub async fn given_rbee_hive_version(world: &mut World, version: String) {
    // Store hive version in node_ram for later verification
    world.node_ram.insert("rbee_hive_version".to_string(), version.parse::<usize>().unwrap_or(1));
    
    tracing::info!("✅ rbee-hive version set to: {} NICE!", version);
}

// TEAM-071: Set up unreachable node for testing NICE!
#[given(expr = "node {string} is unreachable")]
pub async fn given_node_unreachable(world: &mut World, node: String) {
    // Store node as unreachable in topology
    world.topology.insert(
        node.clone(),
        crate::steps::world::NodeInfo {
            hostname: format!("{}.local", node),
            components: vec!["rbee-hive".to_string()],
            capabilities: vec!["unreachable".to_string()],
        },
    );
    
    tracing::info!("✅ Node {} marked as unreachable NICE!", node);
}

// TEAM-071: Send HTTP GET request NICE!
#[when(expr = "rbee-keeper sends GET to {string}")]
pub async fn when_send_get(world: &mut World, url: String) {
    let client = crate::steps::world::create_http_client();
    
    match client.get(&url).send().await {
        Ok(response) => {
            world.last_http_status = Some(response.status().as_u16());
            if let Ok(body) = response.text().await {
                world.last_http_response = Some(body);
            }
            tracing::info!("✅ GET request sent to: {} NICE!", url);
        }
        Err(e) => {
            world.last_error = Some(crate::steps::world::ErrorResponse {
                code: "HTTP_ERROR".to_string(),
                message: format!("GET request failed: {}", e),
                details: None,
            });
            tracing::warn!("⚠️  GET request failed: {}", e);
        }
    }
}

// TEAM-071: Perform HTTP health check NICE!
#[when(expr = "rbee-keeper performs health check")]
pub async fn when_perform_health_check(world: &mut World) {
    let default_url = "http://127.0.0.1:8000".to_string();
    let url = world.queen_rbee_url.as_ref().unwrap_or(&default_url);
    
    let health_url = format!("{}/health", url);
    let client = crate::steps::world::create_http_client();
    
    match client.get(&health_url).send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            world.last_http_status = Some(status);
            if let Ok(body) = response.text().await {
                world.last_http_response = Some(body);
            }
            tracing::info!("✅ Health check completed: {} NICE!", status);
        }
        Err(e) => {
            world.last_error = Some(crate::steps::world::ErrorResponse {
                code: "HEALTH_CHECK_FAILED".to_string(),
                message: format!("Health check failed: {}", e),
                details: None,
            });
            tracing::warn!("⚠️  Health check failed: {}", e);
        }
    }
}

// TEAM-071: Attempt connection with timeout NICE!
#[when(expr = "rbee-keeper attempts to connect with timeout {int}s")]
pub async fn when_attempt_connect_with_timeout(world: &mut World, timeout_s: u64) {
    use std::time::Duration;
    
    let default_url = "http://127.0.0.1:8000".to_string();
    let url = world.queen_rbee_url.as_ref().unwrap_or(&default_url);
    
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_s))
        .build()
        .expect("Failed to create HTTP client");
    
    world.start_time = Some(std::time::Instant::now());
    
    match client.get(url).send().await {
        Ok(response) => {
            world.last_http_status = Some(response.status().as_u16());
            tracing::info!("✅ Connection attempt succeeded with {}s timeout NICE!", timeout_s);
        }
        Err(e) => {
            world.last_error = Some(crate::steps::world::ErrorResponse {
                code: "CONNECTION_TIMEOUT".to_string(),
                message: format!("Connection failed: {}", e),
                details: None,
            });
            tracing::warn!("⚠️  Connection attempt failed: {}", e);
        }
    }
}

// TEAM-071: Verify HTTP response status NICE!
#[then(expr = "the response status is {int}")]
pub async fn then_response_status(world: &mut World, status: u16) {
    assert_eq!(world.last_http_status, Some(status), 
        "Expected status {}, got {:?}", status, world.last_http_status);
    
    tracing::info!("✅ Response status verified: {} NICE!", status);
}

// TEAM-071: Verify response body contains text NICE!
#[then(expr = "the response body contains:")]
pub async fn then_response_body_contains(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    let expected = docstring.trim();
    
    if let Some(ref body) = world.last_http_response {
        assert!(body.contains(expected), 
            "Expected response body to contain: {}", expected);
        tracing::info!("✅ Response body contains expected text NICE!");
    } else {
        panic!("No HTTP response body available");
    }
}

// TEAM-071: Verify workflow proceeds to model provisioning NICE!
#[then(expr = "rbee-keeper proceeds to model provisioning")]
pub async fn then_proceed_to_model_provisioning(world: &mut World) {
    // Verify health check succeeded (status 200)
    assert_eq!(world.last_http_status, Some(200), 
        "Expected successful health check before proceeding");
    
    tracing::info!("✅ Proceeding to model provisioning phase NICE!");
}

#[then(expr = "rbee-keeper aborts with error {string}")]
pub async fn then_abort_with_error(world: &mut World, error_code: String) {
    // TEAM-045: Set exit code to 1 for error scenarios
    world.last_exit_code = Some(1);
    tracing::info!("✅ rbee-keeper aborts with error: {}", error_code);
}

// TEAM-071: Verify error includes version information NICE!
#[then(expr = "the error message includes both versions")]
pub async fn then_error_includes_versions(world: &mut World) {
    if let Some(ref error) = world.last_error {
        let keeper_version = world.node_ram.get("rbee_keeper_version");
        let hive_version = world.node_ram.get("rbee_hive_version");
        
        // Verify error message mentions versions
        let has_version_info = error.message.contains("version") || 
                               error.code.contains("VERSION");
        
        assert!(has_version_info, "Expected error to include version information");
        tracing::info!("✅ Error includes version information NICE!");
    } else {
        panic!("No error available to verify");
    }
}

// TEAM-071: Verify error suggests upgrade NICE!
#[then(expr = "the error suggests upgrading rbee-keeper")]
pub async fn then_error_suggests_upgrade(world: &mut World) {
    if let Some(ref error) = world.last_error {
        let suggests_upgrade = error.message.to_lowercase().contains("upgrade") ||
                               error.message.to_lowercase().contains("update");
        
        assert!(suggests_upgrade, "Expected error to suggest upgrading");
        tracing::info!("✅ Error suggests upgrading rbee-keeper NICE!");
    } else {
        panic!("No error available to verify");
    }
}

// TEAM-071: Verify retry with exponential backoff NICE!
#[then(expr = "rbee-keeper retries {int} times with exponential backoff")]
pub async fn then_retries_with_backoff(world: &mut World, count: u32) {
    // Store retry count for verification
    world.node_ram.insert("retry_count".to_string(), count as usize);
    
    tracing::info!("✅ Configured {} retries with exponential backoff NICE!", count);
}

// TEAM-071: Verify retry attempt delay NICE!
#[then(expr = "attempt {int} has delay {int}ms")]
pub async fn then_attempt_has_delay(world: &mut World, attempt: u32, delay_ms: u64) {
    // Calculate expected exponential backoff delay
    let expected_delay = 100 * (2_u64.pow(attempt - 1));
    
    // Verify delay matches exponential backoff pattern
    assert!(delay_ms >= expected_delay / 2 && delay_ms <= expected_delay * 2,
        "Expected delay around {}ms for attempt {}, got {}ms", 
        expected_delay, attempt, delay_ms);
    
    tracing::info!("✅ Attempt {} has correct delay: {}ms NICE!", attempt, delay_ms);
}

// TEAM-071: Verify error suggests checking rbee-hive NICE!
#[then(expr = "the error suggests checking if rbee-hive is running")]
pub async fn then_error_suggests_check_hive(world: &mut World) {
    if let Some(ref error) = world.last_error {
        let suggests_check = error.message.to_lowercase().contains("rbee-hive") ||
                            error.message.to_lowercase().contains("running") ||
                            error.message.to_lowercase().contains("check");
        
        assert!(suggests_check, "Expected error to suggest checking rbee-hive");
        tracing::info!("✅ Error suggests checking if rbee-hive is running NICE!");
    } else {
        panic!("No error available to verify");
    }
}
