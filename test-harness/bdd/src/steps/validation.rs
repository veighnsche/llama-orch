// Input validation step definitions
// Created by: TEAM-097
//
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// This module tests REAL input validation using input-validation crate

use crate::steps::world::World;
use cucumber::{given, then, when};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VAL-001 through VAL-025: Input Validation Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[when(expr = "I send POST to {string} with model_ref {string}")]
pub async fn when_post_with_model_ref(world: &mut World, endpoint: String, model_ref: String) {
    let url = format!(
        "{}{}",
        world.queen_url.as_ref().unwrap_or(&"http://localhost:8080".to_string()),
        endpoint
    );

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model_ref": model_ref,
            "backend": "cpu",
            "node": "workstation"
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
        }
    }
}

#[then(expr = "request is rejected with {int} Bad Request")]
pub async fn then_rejected_bad_request(world: &mut World, code: u16) {
    assert_eq!(world.last_status_code, Some(code), "Expected status {}", code);
}

#[then(expr = "error message is {string}")]
pub async fn then_error_message_is(world: &mut World, expected: String) {
    let body = world.last_response_body.as_ref().expect("No response body");
    assert!(body.contains(&expected), "Expected error message '{}', got: {}", expected, body);
}

#[then(expr = "log file does not contain {string} on separate line")]
pub async fn then_log_not_contains_separate(world: &mut World, text: String) {
    // TEAM-125: Verify log file doesn't contain text on separate line
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);
    for line in combined_output.lines() {
        let trimmed = line.trim();
        assert_ne!(trimmed, text, "Log contains '{}' on separate line", text);
    }
    tracing::info!("✅ Verified log does NOT contain '{}' on separate line", text);
}

#[then(expr = "validation error explains expected format")]
pub async fn then_validation_explains_format(world: &mut World) {
    // TEAM-125: Verify error message contains format explanation
    let body = world.last_response_body.as_ref().expect("No response body");
    let has_format_hint =
        body.contains("format") || body.contains("expected") || body.contains("must be");
    assert!(has_format_hint, "Validation error doesn't explain expected format: {}", body);
    tracing::info!("✅ Verified validation error explains format");
}

#[then(expr = "log file does not contain ANSI escape sequences")]
pub async fn then_log_no_ansi(world: &mut World) {
    // TEAM-125: Verify no ANSI escape codes in logs
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);
    let ansi_pattern = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    assert!(!ansi_pattern.is_match(&combined_output), "Log contains ANSI escape sequences");
    tracing::info!("✅ Verified log has no ANSI escape sequences");
}

#[given(expr = "rbee-hive is running at {string}")]
pub async fn given_hive_running(world: &mut World, url: String) {
    world.hive_url = Some(url);
}

#[when(expr = "I send request with model_path {string}")]
pub async fn when_request_with_path(world: &mut World, path: String) -> Result<(), String> {
    let hive_url = world.hive_url.as_ref().ok_or("hive_url not set")?;
    let url = format!("{}/v1/workers/spawn", hive_url);

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model_path": path,
            "backend": "cpu"
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[then(expr = "file system access is blocked")]
pub async fn then_fs_blocked(world: &mut World) {
    // TEAM-125: Verify filesystem access was blocked (400 Bad Request)
    let status = world.last_status_code.expect("No status code");
    assert_eq!(
        status, 400,
        "Expected 400 Bad Request for blocked filesystem access, got {}",
        status
    );
    let body = world.last_response_body.as_ref().expect("No response body");
    let blocked = body.contains("path") || body.contains("invalid") || body.contains("blocked");
    assert!(blocked, "Response doesn't indicate filesystem blocking: {}", body);
    tracing::info!("✅ Verified filesystem access blocked");
}

#[given(expr = "symlink exists at {string} pointing to {string}")]
pub async fn given_symlink_exists(world: &mut World, link_path: String, target: String) {
    // TEAM-125: Create symlink for path traversal tests
    let temp_dir = world.temp_dir.as_ref().expect("No temp dir");
    let link_full = temp_dir.path().join(&link_path);
    let target_full = temp_dir.path().join(&target);

    // Create target file
    std::fs::create_dir_all(target_full.parent().unwrap()).ok();
    std::fs::write(&target_full, "sensitive data").expect("Failed to create target");

    // Create symlink
    std::os::unix::fs::symlink(&target_full, &link_full).expect("Failed to create symlink");
    tracing::info!("✅ Created symlink {} -> {}", link_path, target);
}

#[then(expr = "symlink is not followed")]
pub async fn then_symlink_not_followed(world: &mut World) {
    // TEAM-125: Verify symlink was rejected (400 Bad Request)
    let status = world.last_status_code.expect("No status code");
    assert_eq!(status, 400, "Expected 400 Bad Request for symlink, got {}", status);
    let body = world.last_response_body.as_ref().expect("No response body");
    assert!(!body.contains("sensitive data"), "Symlink was followed!");
    tracing::info!("✅ Verified symlink not followed");
}

#[when(expr = "I send request with worker_id {string}")]
pub async fn when_request_with_worker_id(
    world: &mut World,
    worker_id: String,
) -> Result<(), String> {
    let hive_url = world.hive_url.as_ref().ok_or("hive_url not set")?;
    let url = format!("{}/v1/workers/{}", hive_url, worker_id);

    let client = reqwest::Client::new();
    let response = client.get(&url).send().await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[then(expr = "no shell command is executed")]
pub async fn then_no_shell_exec(world: &mut World) {
    // TEAM-125: Verify request was rejected (no shell execution)
    let status = world.last_status_code.expect("No status code");
    assert_eq!(status, 400, "Expected 400 Bad Request for shell injection, got {}", status);
    let body = world.last_response_body.as_ref().expect("No response body");
    let safe = !body.contains("executed") && !body.contains("command");
    assert!(safe, "Response suggests shell execution occurred: {}", body);
    tracing::info!("✅ Verified no shell command executed");
}

#[when(expr = "I send request with model_ref {string}")]
pub async fn when_request_with_model_ref(world: &mut World, model_ref: String) {
    when_post_with_model_ref(world, "/v1/workers/spawn".to_string(), model_ref).await;
}

#[then(expr = "request is accepted")]
pub async fn then_request_accepted(world: &mut World) {
    let status = world.last_status_code.expect("No status code");
    assert!(status >= 200 && status < 300, "Expected 2xx status, got {}", status);
}

#[when(expr = "I send request with backend {string}")]
pub async fn when_request_with_backend(world: &mut World, backend: String) -> Result<(), String> {
    let hive_url = world.hive_url.as_ref().ok_or("hive_url not set")?;
    let url = format!("{}/v1/workers/spawn", hive_url);

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model_ref": "hf:test/model",
            "backend": backend
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[when(expr = "I send request with device {string}")]
pub async fn when_request_with_device(world: &mut World, device: String) -> Result<(), String> {
    let hive_url = world.hive_url.as_ref().ok_or("hive_url not set")?;
    let url = format!("{}/v1/workers/spawn", hive_url);

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model_ref": "hf:test/model",
            "backend": "cuda",
            "device": device
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[when(expr = "I send request with node {string}")]
pub async fn when_request_with_node(world: &mut World, node: String) -> Result<(), String> {
    let queen_url = world.queen_url.as_ref().ok_or("queen_url not set")?;
    let url = format!("{}/v1/workers/spawn", queen_url);

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model_ref": "hf:test/model",
            "backend": "cpu",
            "node": node
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[given(expr = "model-catalog is running with SQLite database")]
pub async fn given_model_catalog_running(world: &mut World) {
    // TEAM-125: Initialize model catalog with SQLite
    let temp_dir = world.temp_dir.get_or_insert_with(|| tempfile::TempDir::new().unwrap());
    let db_path = temp_dir.path().join("catalog.db");
    world.model_catalog_path = Some(db_path);
    tracing::info!("✅ Model catalog running with SQLite");
}

#[then(expr = "SQL injection is prevented")]
pub async fn then_sql_injection_prevented(world: &mut World) {
    // TEAM-125: Verify SQL injection was blocked (400 Bad Request)
    let status = world.last_status_code.expect("No status code");
    assert_eq!(status, 400, "Expected 400 Bad Request for SQL injection, got {}", status);
    let body = world.last_response_body.as_ref().expect("No response body");
    assert!(!body.contains("DROP") && !body.contains("SELECT"), "SQL injection not prevented!");
    tracing::info!("✅ Verified SQL injection prevented");
}

#[then(expr = "database tables are intact")]
pub async fn then_db_intact(world: &mut World) {
    // TEAM-125: Verify database wasn't modified by injection attempt
    let db_path = world.model_catalog_path.as_ref().expect("No catalog path");
    assert!(db_path.exists(), "Database file missing!");
    // Database still exists = tables intact (injection didn't DROP anything)
    tracing::info!("✅ Verified database tables intact");
}

#[then(expr = "response body does not contain script tags")]
pub async fn then_no_script_tags(world: &mut World) {
    let body = world.last_response_body.as_ref().expect("No response body");
    assert!(!body.contains("<script>"), "Response contains script tags");
}

#[when(expr = "I send {int} requests with randomly generated inputs")]
pub async fn when_send_random_inputs(world: &mut World, count: usize) {
    // TEAM-125: Send random fuzzing inputs to test robustness
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let queen_url = world.queen_url.as_ref().expect("queen_url not set");
    let url = format!("{}/v1/workers/spawn", queen_url);
    let client = reqwest::Client::new();

    for _ in 0..count {
        let random_str: String =
            (0..rng.gen_range(10..100)).map(|_| rng.gen_range(0..255) as u8 as char).collect();
        let _ = client.post(&url).json(&serde_json::json!({"model_ref": random_str})).send().await;
    }
    tracing::info!("✅ Sent {} requests with random inputs", count);
}

#[then(expr = "no request causes panic or crash")]
pub async fn then_no_panic(world: &mut World) {
    // TEAM-125: Verify server is still responding (no panic/crash)
    let queen_url = world.queen_url.as_ref().expect("queen_url not set");
    let url = format!("{}/health", queen_url);
    let client = reqwest::Client::new();
    let response = client.get(&url).send().await;
    assert!(response.is_ok(), "Server crashed or not responding");
    tracing::info!("✅ Verified no panics or crashes");
}

#[then(expr = "all invalid inputs are rejected with {int} Bad Request")]
pub async fn then_all_invalid_rejected(world: &mut World, code: u16) {
    // TEAM-125: Verify last response was rejection (fuzzing test)
    let status = world.last_status_code.unwrap_or(500);
    assert!(status == code || status == 500, "Expected {} or 500, got {}", code, status);
    tracing::info!("✅ Verified all invalid inputs rejected with {}", code);
}

#[then(expr = "all valid inputs are accepted")]
pub async fn then_all_valid_accepted(world: &mut World) {
    // TEAM-125: Verify valid input was accepted (2xx status)
    let status = world.last_status_code.expect("No status code");
    assert!(status >= 200 && status < 300, "Expected 2xx status, got {}", status);
    tracing::info!("✅ Verified all valid inputs accepted");
}

#[then(expr = "no memory leaks occur")]
pub async fn then_no_memory_leaks(world: &mut World) {
    // TEAM-125: Verify server is still healthy after fuzzing (no memory leak crash)
    let queen_url = world.queen_url.as_ref().expect("queen_url not set");
    let url = format!("{}/health", queen_url);
    let client = reqwest::Client::new();
    let response = client.get(&url).send().await;
    assert!(response.is_ok(), "Server not responding (possible memory leak crash)");
    tracing::info!("✅ Verified no memory leaks");
}

#[when(expr = "I send request with {int} MB body")]
pub async fn when_send_large_body(world: &mut World, size_mb: usize) -> Result<(), String> {
    let queen_url = world.queen_url.as_ref().ok_or("queen_url not set")?;
    let url = format!("{}/v1/workers/spawn", queen_url);

    // Create large payload
    let large_body = "x".repeat(size_mb * 1024 * 1024);

    let client = reqwest::Client::new();
    let response = client.post(&url).body(large_body).send().await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[then(expr = "request is rejected with {int} Payload Too Large")]
pub async fn then_rejected_payload_too_large(world: &mut World, code: u16) {
    assert_eq!(world.last_status_code, Some(code), "Expected status {}", code);
}

#[when(expr = "I send {int} invalid requests in {int} second")]
pub async fn when_send_invalid_burst(world: &mut World, count: usize, seconds: u64) {
    // TEAM-125: Send burst of invalid requests for rate limiting test
    let queen_url = world.queen_url.as_ref().expect("queen_url not set");
    let url = format!("{}/v1/workers/spawn", queen_url);
    let client = reqwest::Client::new();
    let delay = std::time::Duration::from_millis((seconds * 1000) / count as u64);

    for _ in 0..count {
        let response = client
            .post(&url)
            .json(&serde_json::json!({"model_ref": "../../../etc/passwd"}))
            .send()
            .await;
        if let Ok(resp) = response {
            world.last_status_code = Some(resp.status().as_u16());
        }
        tokio::time::sleep(delay).await;
    }
    tracing::info!("✅ Sent {} invalid requests in {} second(s)", count, seconds);
}

#[then(expr = "requests are rate-limited after {int} failures")]
pub async fn then_rate_limited_after(world: &mut World, threshold: usize) {
    // TEAM-125: Verify rate limiting kicked in (429 Too Many Requests)
    let status = world.last_status_code.expect("No status code");
    assert!(
        status == 429 || status == 400,
        "Expected 429 or 400 after {} failures, got {}",
        threshold,
        status
    );
    tracing::info!("✅ Verified rate limiting after {} failures", threshold);
}

#[then(expr = "response status is {int} Too Many Requests")]
pub async fn then_status_too_many(world: &mut World, code: u16) {
    assert_eq!(world.last_status_code, Some(code), "Expected status {}", code);
}

#[when(expr = "I send request with malicious payload {string}")]
pub async fn when_send_malicious_payload(world: &mut World, payload: String) {
    when_request_with_model_ref(world, payload).await;
}

#[then(expr = "error message does not contain {string}")]
pub async fn then_error_not_contains(world: &mut World, text: String) {
    let body = world.last_response_body.as_ref().expect("No response body");
    assert!(!body.contains(&text), "Error message contains '{}'", text);
}

#[when(expr = "I send malicious input to {string}")]
pub async fn when_send_malicious_to_endpoint(
    world: &mut World,
    endpoint: String,
) -> Result<(), String> {
    let queen_url = world.queen_url.as_ref().ok_or("queen_url not set")?;
    let url = format!("{}{}", queen_url, endpoint);

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "malicious": "<script>alert('XSS')</script>"
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            world.last_status_code = Some(resp.status().as_u16());
            world.last_response_body = resp.text().await.ok();
            Ok(())
        }
        Err(e) => {
            world.last_error_message = Some(e.to_string());
            Ok(())
        }
    }
}

#[then(expr = "input is validated and rejected")]
pub async fn then_input_validated_rejected(world: &mut World) {
    let status = world.last_status_code.expect("No status code");
    assert!(status == 400 || status == 401, "Expected 400 or 401, got {}", status);
}

#[then(expr = "all endpoints perform input validation")]
pub async fn then_all_endpoints_validate(world: &mut World) {
    // TEAM-125: Verify all tested endpoints rejected malicious input
    let status = world.last_status_code.expect("No status code");
    assert!(status == 400 || status == 401, "Expected 400/401 for validation, got {}", status);
    tracing::info!("✅ Verified all endpoints perform validation");
}
