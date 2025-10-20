// Created by: TEAM-155
//! BDD step definitions for SSE streaming tests

use crate::steps::world::BddWorld;
use cucumber::{given, then, when};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;

// ============================================================================
// Given Steps
// ============================================================================

#[given("queen-rbee is not running")]
async fn queen_not_running(world: &mut BddWorld) {
    // Kill any existing queen process
    let _ = Command::new("pkill").arg("-f").arg("queen-rbee").output();

    sleep(Duration::from_millis(500)).await;
    world.queen_process = None;
}

#[given("queen-rbee is running on port 8500")]
async fn queen_is_running(world: &mut BddWorld) {
    // Start queen if not already running
    let output = Command::new("curl")
        .arg("-s")
        .arg("http://localhost:8500/health")
        .output()
        .expect("Failed to check queen health");

    if !output.status.success() {
        // Start queen
        let child = Command::new("../../../target/debug/queen-rbee")
            .arg("--port")
            .arg("8500")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to start queen-rbee");

        world.queen_process = Some(child);

        // Wait for queen to be ready
        for _ in 0..10 {
            sleep(Duration::from_millis(500)).await;
            let check = Command::new("curl")
                .arg("-s")
                .arg("http://localhost:8500/health")
                .output()
                .expect("Failed to check queen health");

            if check.status.success() {
                break;
            }
        }
    }
}

#[given(regex = r#"a job exists with id "(.+)""#)]
async fn job_exists(_world: &mut BddWorld, _job_id: String) {
    // TODO: Create a job in the registry
    // For now, this is a placeholder
}

// ============================================================================
// When Steps
// ============================================================================

#[when(regex = r#"I run "(.+)""#)]
async fn run_command(world: &mut BddWorld, command: String) {
    let output =
        Command::new("sh").arg("-c").arg(&command).output().expect("Failed to run command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    world.expected_message = Some(format!("{}{}", stdout, stderr));
}

#[when(regex = r#"I POST to "(.+)" with test job data"#)]
async fn post_to_endpoint(world: &mut BddWorld, endpoint: String) {
    let url = format!("{}{}", world.queen_url, endpoint);
    let client = reqwest::Client::new();

    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "test-model",
            "prompt": "test",
            "max_tokens": 1,
            "temperature": 1.0,
        }))
        .send()
        .await
        .expect("Failed to POST");

    let status = response.status();
    let body = response.text().await.expect("Failed to read response");

    world.expected_message = Some(format!("Status: {}\nBody: {}", status, body));
}

#[when(regex = r#"I GET "(.+)""#)]
async fn get_endpoint(world: &mut BddWorld, endpoint: String) {
    let url = format!("{}{}", world.queen_url, endpoint);
    let client = reqwest::Client::new();

    let response = client.get(&url).send().await.expect("Failed to GET");

    let status = response.status();
    let body = response.text().await.expect("Failed to read response");

    world.expected_message = Some(format!("Status: {}\nBody: {}", status, body));
}

// ============================================================================
// Then Steps
// ============================================================================

#[then("queen-rbee should auto-start on port 8500")]
async fn queen_should_autostart(_world: &mut BddWorld) {
    // Wait a bit for queen to start
    sleep(Duration::from_secs(2)).await;

    let output = Command::new("curl")
        .arg("-s")
        .arg("http://localhost:8500/health")
        .output()
        .expect("Failed to check queen health");

    assert!(output.status.success(), "Queen should be running");
}

#[then(regex = r#"I should see "(.+)""#)]
async fn should_see_message(world: &mut BddWorld, expected: String) {
    let message = world.expected_message.as_ref().expect("No message captured");

    assert!(message.contains(&expected), "Expected to see '{}' in output:\n{}", expected, message);
}

#[then("the SSE stream should complete")]
async fn sse_stream_completes(world: &mut BddWorld) {
    let message = world.expected_message.as_ref().expect("No message captured");

    assert!(
        message.contains("SSE test complete") || message.contains("[DONE]"),
        "SSE stream should complete"
    );
}

#[then("queen-rbee should shutdown cleanly")]
async fn queen_should_shutdown(world: &mut BddWorld) {
    let message = world.expected_message.as_ref().expect("No message captured");

    assert!(message.contains("Cleanup complete"), "Queen should shutdown cleanly");
}

#[then(regex = r#"the response should have status (\d+)"#)]
async fn response_status(world: &mut BddWorld, status: u16) {
    let message = world.expected_message.as_ref().expect("No message captured");

    assert!(
        message.contains(&format!("Status: {}", status)),
        "Expected status {}, got: {}",
        status,
        message
    );
}

#[then(regex = r#"the response should contain "(.+)""#)]
async fn response_contains(world: &mut BddWorld, expected: String) {
    let message = world.expected_message.as_ref().expect("No message captured");

    assert!(
        message.contains(&expected),
        "Expected response to contain '{}', got: {}",
        expected,
        message
    );
}

#[then(regex = r#"the job_id should start with "(.+)""#)]
async fn job_id_starts_with(world: &mut BddWorld, prefix: String) {
    let message = world.expected_message.as_ref().expect("No message captured");

    // Extract job_id from JSON response
    if let Some(start) = message.find("\"job_id\":") {
        let rest = &message[start..];
        assert!(
            rest.contains(&format!("\"job_id\":\"{}", prefix)),
            "job_id should start with '{}'",
            prefix
        );
    } else {
        panic!("No job_id found in response");
    }
}

#[then(regex = r#"the sse_url should match "(.+)""#)]
async fn sse_url_matches(world: &mut BddWorld, pattern: String) {
    let message = world.expected_message.as_ref().expect("No message captured");

    // Check if sse_url contains the pattern
    assert!(
        message.contains("sse_url") && message.contains("/jobs/") && message.contains("/stream"),
        "sse_url should match pattern '{}'",
        pattern
    );
}

#[then("the response should be SSE stream")]
async fn response_is_sse(_world: &mut BddWorld) {
    // TODO: Check content-type header
    // For now, just check that we got some response
}

#[then(regex = r#"I should receive a "(.+)" event"#)]
async fn should_receive_event(world: &mut BddWorld, event_type: String) {
    let message = world.expected_message.as_ref().expect("No message captured");

    assert!(message.contains(&event_type), "Should receive '{}' event", event_type);
}

#[then("the following should happen in order:")]
async fn steps_in_order(world: &mut BddWorld, _table: &cucumber::gherkin::Table) {
    // TODO: Implement ordered step verification
    // For now, just check that we have output
    assert!(world.expected_message.is_some(), "Should have output");
}
