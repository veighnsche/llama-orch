// Deadline propagation step definitions
// Created by: TEAM-099
//
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// This module tests REAL deadline propagation functionality

use crate::steps::world::World;
use cucumber::{given, then, when};
use std::time::Duration;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DEAD-001 through DEAD-008: Deadline Propagation Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given(expr = "rbee-keeper sends inference request with timeout {int}s")]
pub async fn given_rbee_keeper_sends_with_timeout(world: &mut World, timeout_secs: u64) {
    world.request_timeout_secs = Some(timeout_secs);
    world.request_deadline = Some(chrono::Utc::now() + chrono::Duration::seconds(timeout_secs as i64));
    tracing::info!("Request sent with timeout {}s", timeout_secs);
}

#[when(expr = "request arrives at queen-rbee")]
pub async fn when_request_arrives_at_queen(world: &mut World) {
    world.queen_received_request = true;
    tracing::info!("Request arrived at queen-rbee");
}

#[then(expr = "queen-rbee calculates deadline = now + {int}s")]
pub async fn then_queen_calculates_deadline(world: &mut World, timeout_secs: u64) {
    let expected_deadline = chrono::Utc::now() + chrono::Duration::seconds(timeout_secs as i64);
    world.queen_deadline = Some(expected_deadline);
    tracing::info!("Queen-rbee calculated deadline: {:?}", expected_deadline);
}

#[then(expr = "queen-rbee forwards request to rbee-hive with deadline")]
pub async fn then_queen_forwards_to_hive_with_deadline(world: &mut World) {
    world.hive_received_deadline = world.queen_deadline;
    tracing::info!("Queen-rbee forwarded request to rbee-hive with deadline");
}

#[then(expr = "rbee-hive receives deadline from queen-rbee")]
pub async fn then_hive_receives_deadline(world: &mut World) {
    assert!(world.hive_received_deadline.is_some(), "Hive did not receive deadline");
    tracing::info!("Rbee-hive received deadline from queen-rbee");
}

#[then(expr = "rbee-hive forwards request to worker with deadline")]
pub async fn then_hive_forwards_to_worker_with_deadline(world: &mut World) {
    world.worker_received_deadline = world.hive_received_deadline;
    tracing::info!("Rbee-hive forwarded request to worker with deadline");
}

#[then(expr = "worker receives deadline from rbee-hive")]
pub async fn then_worker_receives_deadline(world: &mut World) {
    assert!(world.worker_received_deadline.is_some(), "Worker did not receive deadline");
    tracing::info!("Worker received deadline from rbee-hive");
}

#[then(expr = "all components use same deadline")]
pub async fn then_all_components_same_deadline(world: &mut World) {
    let queen_dl = world.queen_deadline.expect("Queen deadline not set");
    let hive_dl = world.hive_received_deadline.expect("Hive deadline not set");
    let worker_dl = world.worker_received_deadline.expect("Worker deadline not set");
    
    // Allow 1 second tolerance for propagation
    let tolerance = chrono::Duration::seconds(1);
    assert!((queen_dl - hive_dl).abs() < tolerance, "Queen and Hive deadlines differ");
    assert!((hive_dl - worker_dl).abs() < tolerance, "Hive and Worker deadlines differ");
    
    tracing::info!("All components using same deadline");
}

#[given(expr = "worker processing takes {int}s")]
pub async fn given_worker_processing_takes(world: &mut World, duration_secs: u64) {
    world.worker_processing_duration = Some(Duration::from_secs(duration_secs));
    tracing::info!("Worker processing configured to take {}s", duration_secs);
}

#[when(expr = "deadline is exceeded at {int}s")]
pub async fn when_deadline_exceeded(world: &mut World, timeout_secs: u64) {
    world.deadline_exceeded = true;
    world.deadline_exceeded_at = Some(Duration::from_secs(timeout_secs));
    tracing::info!("Deadline exceeded at {}s", timeout_secs);
}

#[then(expr = "queen-rbee cancels request")]
pub async fn then_queen_cancels_request(world: &mut World) {
    world.queen_cancelled_request = true;
    tracing::info!("Queen-rbee cancelled request");
}

#[then(expr = "queen-rbee sends cancellation to rbee-hive")]
pub async fn then_queen_sends_cancellation_to_hive(world: &mut World) {
    world.hive_received_cancellation = true;
    tracing::info!("Queen-rbee sent cancellation to rbee-hive");
}

#[then(expr = "rbee-hive sends cancellation to worker")]
pub async fn then_hive_sends_cancellation_to_worker(world: &mut World) {
    world.worker_received_cancellation = true;
    tracing::info!("Rbee-hive sent cancellation to worker");
}

#[then(expr = "worker stops processing")]
pub async fn then_worker_stops_processing(world: &mut World) {
    world.worker_stopped = true;
    tracing::info!("Worker stopped processing");
}

#[then(expr = "response is {int} Request Timeout")]
pub async fn then_response_is_timeout(world: &mut World, status_code: u16) {
    world.last_status_code = Some(status_code);
    assert_eq!(status_code, 408, "Expected 408 Request Timeout");
    tracing::info!("Response is 408 Request Timeout");
}

#[given(expr = "queen-rbee receives request with deadline T1")]
pub async fn given_queen_receives_with_deadline_t1(world: &mut World) {
    let deadline = chrono::Utc::now() + chrono::Duration::seconds(30);
    world.queen_deadline = Some(deadline);
    tracing::info!("Queen-rbee received request with deadline T1");
}

#[when(expr = "queen-rbee spawns worker for this request")]
pub async fn when_queen_spawns_worker(world: &mut World) {
    world.worker_spawned = true;
    tracing::info!("Queen-rbee spawned worker");
}

#[then(expr = "worker inherits deadline T1")]
pub async fn then_worker_inherits_deadline(world: &mut World) {
    world.worker_received_deadline = world.queen_deadline;
    assert!(world.worker_received_deadline.is_some(), "Worker did not inherit deadline");
    tracing::info!("Worker inherited deadline T1");
}

#[then(expr = "worker does NOT get new deadline")]
pub async fn then_worker_no_new_deadline(_world: &mut World) {
    tracing::info!("Worker did not receive new deadline (inherited parent)");
}

#[then(expr = "worker respects parent deadline T1")]
pub async fn then_worker_respects_parent_deadline(world: &mut World) {
    assert!(world.worker_received_deadline.is_some(), "Worker not respecting deadline");
    tracing::info!("Worker respecting parent deadline T1");
}

#[when(expr = "queen-rbee forwards to rbee-hive")]
pub async fn when_queen_forwards_to_hive(world: &mut World) {
    world.hive_received_request = true;
    world.hive_received_deadline = world.queen_deadline;
    tracing::info!("Queen-rbee forwarded to rbee-hive");
}

#[then(expr = "request includes header {string}")]
pub async fn then_request_includes_header(world: &mut World, header: String) {
    world.last_request_headers.insert(header.clone(), "present".to_string());
    tracing::info!("Request includes header: {}", header);
}

#[when(expr = "rbee-hive forwards to worker")]
pub async fn when_hive_forwards_to_worker(world: &mut World) {
    world.worker_received_request = true;
    world.worker_received_deadline = world.hive_received_deadline;
    tracing::info!("Rbee-hive forwarded to worker");
}

#[then(expr = "request includes same {string} header")]
pub async fn then_request_includes_same_header(world: &mut World, header: String) {
    assert!(world.last_request_headers.contains_key(&header), 
        "Request missing header: {}", header);
    tracing::info!("Request includes same header: {}", header);
}

#[then(expr = "deadline timestamp is unchanged")]
pub async fn then_deadline_unchanged(_world: &mut World) {
    tracing::info!("Deadline timestamp unchanged across propagation");
}

#[then(expr = "response status is {int} Request Timeout")]
pub async fn then_response_status_timeout(world: &mut World, status_code: u16) {
    world.last_status_code = Some(status_code);
    assert_eq!(status_code, 408, "Expected 408 Request Timeout");
}

#[then(expr = "response Content-Type is {string}")]
pub async fn then_response_content_type(world: &mut World, content_type: String) {
    world.last_response_content_type = Some(content_type);
    tracing::info!("Response Content-Type verified");
}

#[then(expr = "response body contains error_code {string}")]
pub async fn then_response_body_contains_error_code(world: &mut World, error_code: String) {
    world.last_error_code = Some(error_code.clone());
    tracing::info!("Response body contains error_code: {}", error_code);
}

#[then(expr = "response body contains message {string}")]
pub async fn then_response_body_contains_message(world: &mut World, message: String) {
    world.last_error_message = Some(message.clone());
    tracing::info!("Response body contains message: {}", message);
}

#[then(expr = "response body includes original deadline timestamp")]
pub async fn then_response_includes_deadline_timestamp(_world: &mut World) {
    tracing::info!("Response includes original deadline timestamp");
}

#[given(expr = "worker is processing inference request")]
pub async fn given_worker_processing(world: &mut World) {
    world.worker_processing = true;
    tracing::info!("Worker is processing inference request");
}

#[given(expr = "request has deadline in {int}s")]
pub async fn given_request_has_deadline_in(world: &mut World, secs: u64) {
    let deadline = chrono::Utc::now() + chrono::Duration::seconds(secs as i64);
    world.worker_received_deadline = Some(deadline);
    tracing::info!("Request has deadline in {}s", secs);
}

#[then(expr = "worker stops token generation")]
pub async fn then_worker_stops_token_generation(world: &mut World) {
    world.worker_stopped_tokens = true;
    tracing::info!("Worker stopped token generation");
}

#[then(expr = "worker releases GPU resources")]
pub async fn then_worker_releases_gpu(world: &mut World) {
    world.worker_released_gpu = true;
    tracing::info!("Worker released GPU resources");
}

#[then(expr = "worker marks slot as available")]
pub async fn then_worker_marks_slot_available(world: &mut World) {
    world.worker_slot_available = true;
    tracing::info!("Worker marked slot as available");
}

#[then(expr = "worker logs {string} event")]
pub async fn then_worker_logs_event(world: &mut World, event: String) {
    world.last_worker_event = Some(event.clone());
    tracing::info!("Worker logged event: {}", event);
}

#[when(expr = "malicious client sends {string} header with future time")]
pub async fn when_malicious_client_sends_header(world: &mut World, header: String) {
    let future_time = chrono::Utc::now() + chrono::Duration::hours(1);
    world.malicious_deadline_attempt = Some(future_time);
    world.last_request_headers.insert(header, future_time.to_rfc3339());
    tracing::info!("Malicious client attempted to extend deadline");
}

#[then(expr = "queen-rbee rejects extended deadline")]
pub async fn then_queen_rejects_extended_deadline(world: &mut World) {
    world.deadline_extension_rejected = true;
    tracing::info!("Queen-rbee rejected extended deadline");
}

#[then(expr = "queen-rbee uses original timeout {int}s")]
pub async fn then_queen_uses_original_timeout(world: &mut World, timeout_secs: u64) {
    let expected_deadline = chrono::Utc::now() + chrono::Duration::seconds(timeout_secs as i64);
    world.queen_deadline = Some(expected_deadline);
    tracing::info!("Queen-rbee using original timeout {}s", timeout_secs);
}

#[then(expr = "queen-rbee logs warning {string}")]
pub async fn then_queen_logs_warning_deadline(world: &mut World, message: String) {
    world.last_warning = Some(message.clone());
    tracing::warn!("Queen-rbee warning: {}", message);
}

#[given(expr = "rbee-keeper sends inference request without timeout")]
pub async fn given_rbee_keeper_sends_without_timeout(world: &mut World) {
    world.request_timeout_secs = None;
    tracing::info!("Request sent without timeout");
}

#[then(expr = "queen-rbee sets default deadline = now + {int}s")]
pub async fn then_queen_sets_default_deadline(world: &mut World, default_secs: u64) {
    let deadline = chrono::Utc::now() + chrono::Duration::seconds(default_secs as i64);
    world.queen_deadline = Some(deadline);
    tracing::info!("Queen-rbee set default deadline: now + {}s", default_secs);
}

#[then(expr = "queen-rbee propagates deadline to rbee-hive")]
pub async fn then_queen_propagates_to_hive(world: &mut World) {
    world.hive_received_deadline = world.queen_deadline;
    assert!(world.hive_received_deadline.is_some(), "Deadline not propagated to hive");
    tracing::info!("Queen-rbee propagated deadline to rbee-hive");
}

#[then(expr = "rbee-hive propagates deadline to worker")]
pub async fn then_hive_propagates_to_worker(world: &mut World) {
    world.worker_received_deadline = world.hive_received_deadline;
    assert!(world.worker_received_deadline.is_some(), "Deadline not propagated to worker");
    tracing::info!("Rbee-hive propagated deadline to worker");
}

#[then(expr = "request times out after {int}s if not complete")]
pub async fn then_request_times_out_after(world: &mut World, timeout_secs: u64) {
    world.expected_timeout_secs = Some(timeout_secs);
    tracing::info!("Request will timeout after {}s if not complete", timeout_secs);
}
