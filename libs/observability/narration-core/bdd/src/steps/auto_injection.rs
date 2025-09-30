// Step definitions for auto-injection behaviors

use cucumber::{then, when};
use observability_narration_core::{narrate_auto, narrate_full, service_identity, current_timestamp_ms, NarrationFields};
use crate::steps::world::World;
use std::thread;
use std::time::Duration;

#[when("I get the service identity")]
pub async fn when_get_service_identity(world: &mut World) {
    world.service_identity = service_identity();
}

#[when("I get timestamp 1")]
pub async fn when_get_timestamp_1(world: &mut World) {
    world.timestamp_1 = current_timestamp_ms();
}

#[when(regex = r"^I wait (\d+) milliseconds$")]
pub async fn when_wait_ms(_world: &mut World, ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

#[when("I get timestamp 2")]
pub async fn when_get_timestamp_2(world: &mut World) {
    world.timestamp_2 = current_timestamp_ms();
}

#[when("I narrate_auto without emitted_by")]
pub async fn when_narrate_auto_without_emitted_by(_world: &mut World) {
    narrate_auto(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate_auto with emitted_by "([^"]+)"$"#)]
pub async fn when_narrate_auto_with_emitted_by(_world: &mut World, emitted_by: String) {
    narrate_auto(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        emitted_by: Some(emitted_by),
        ..Default::default()
    });
}

#[when("I narrate_auto without emitted_at_ms")]
pub async fn when_narrate_auto_without_timestamp(_world: &mut World) {
    narrate_auto(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when(regex = r"^I narrate_auto with emitted_at_ms (\d+)$")]
pub async fn when_narrate_auto_with_timestamp(_world: &mut World, timestamp: u64) {
    narrate_auto(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        emitted_at_ms: Some(timestamp),
        ..Default::default()
    });
}

#[when("I narrate_full without trace_id")]
pub async fn when_narrate_full_without_trace_id(_world: &mut World) {
    narrate_full(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate_full with trace_id "([^"]+)"$"#)]
pub async fn when_narrate_full_with_trace_id(_world: &mut World, trace_id: String) {
    narrate_full(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        trace_id: Some(trace_id),
        ..Default::default()
    });
}

#[then(regex = r#"^it matches the pattern "([^"]+)"$"#)]
pub async fn then_matches_pattern(world: &mut World, pattern: String) {
    let re = regex::Regex::new(&pattern).expect("Invalid regex pattern");
    assert!(re.is_match(&world.service_identity), 
        "Service identity '{}' does not match pattern '{}'", 
        world.service_identity, pattern);
}

#[then(regex = r#"^it contains "([^"]+)"$"#)]
pub async fn then_contains(world: &mut World, text: String) {
    assert!(world.service_identity.contains(&text),
        "Service identity '{}' does not contain '{}'",
        world.service_identity, text);
}

#[then("timestamp 2 is greater than or equal to timestamp 1")]
pub async fn then_timestamp_2_gte_1(world: &mut World) {
    assert!(world.timestamp_2 >= world.timestamp_1,
        "Timestamp 2 ({}) should be >= timestamp 1 ({})",
        world.timestamp_2, world.timestamp_1);
}

#[then("the captured narration has emitted_by set")]
pub async fn then_has_emitted_by_set(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(captured[0].emitted_by.is_some(), "emitted_by should be set");
}

#[then("the captured narration has emitted_at_ms set")]
pub async fn then_has_emitted_at_ms_set(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(captured[0].emitted_at_ms.is_some(), "emitted_at_ms should be set");
}

#[then(regex = r#"^the captured narration emitted_by contains "([^"]+)"$"#)]
pub async fn then_emitted_by_contains(world: &mut World, text: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    let emitted_by = captured[0].emitted_by.as_ref().expect("emitted_by not set");
    assert!(emitted_by.contains(&text), 
        "emitted_by '{}' should contain '{}'", emitted_by, text);
}

#[then(regex = r"^the captured narration has emitted_at_ms greater than (\d+)$")]
pub async fn then_emitted_at_ms_gt(world: &mut World, threshold: u64) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    let timestamp = captured[0].emitted_at_ms.expect("emitted_at_ms not set");
    assert!(timestamp > threshold, 
        "emitted_at_ms ({}) should be > {}", timestamp, threshold);
}

#[then("the captured narration may have trace_id")]
pub async fn then_may_have_trace_id(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    // May or may not have trace_id depending on OTEL context
    // This is a permissive assertion
}
