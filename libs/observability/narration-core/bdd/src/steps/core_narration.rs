// Step definitions for core narration behaviors

use crate::steps::world::World;
use cucumber::{given, then, when};
use observability_narration_core::{human, narrate, CaptureAdapter, NarrationFields};

#[given("a clean capture adapter")]
pub async fn given_clean_adapter(world: &mut World) {
    let adapter = CaptureAdapter::install();
    adapter.clear();
    world.adapter = Some(adapter);
}

#[when(regex = "^I narrate with actor (.+), action (.+), target (.+), and human (.+)$")]
pub async fn when_narrate_full(
    world: &mut World,
    actor: String,
    action: String,
    target: String,
    human_text: String,
) {
    narrate(NarrationFields {
        actor: Box::leak(actor.into_boxed_str()),
        action: Box::leak(action.into_boxed_str()),
        target,
        human: human_text,
        ..Default::default()
    });
}

#[when(regex = "^I narrate with human text (.+)$")]
pub async fn when_narrate_with_human(world: &mut World, human_text: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: human_text,
        ..Default::default()
    });
}

#[when(regex = "^I narrate with correlation_id (.+)$")]
pub async fn when_narrate_with_correlation_id(world: &mut World, correlation_id: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        correlation_id: Some(correlation_id),
        ..Default::default()
    });
}

#[when(regex = "^I narrate with session_id (.+)$")]
pub async fn when_narrate_with_session_id(world: &mut World, session_id: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        session_id: Some(session_id),
        ..Default::default()
    });
}

#[when(regex = "^I narrate with pool_id (.+)$")]
pub async fn when_narrate_with_pool_id(world: &mut World, pool_id: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        pool_id: Some(pool_id),
        ..Default::default()
    });
}

#[when(regex = "^I narrate with emitted_by (.+)$")]
pub async fn when_narrate_with_emitted_by(world: &mut World, emitted_by: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        emitted_by: Some(emitted_by),
        ..Default::default()
    });
}

#[when(regex = r"^I narrate with emitted_at_ms (\d+)$")]
pub async fn when_narrate_with_emitted_at_ms(world: &mut World, timestamp: u64) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        emitted_at_ms: Some(timestamp),
        ..Default::default()
    });
}

#[when(regex = "^I narrate with trace_id (.+)$")]
pub async fn when_narrate_with_trace_id(world: &mut World, trace_id: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        trace_id: Some(trace_id),
        ..Default::default()
    });
}

#[when(
    regex = "^I call legacy human\\(\\) with actor (.+), action (.+), target (.+), message (.+)$"
)]
pub async fn when_call_legacy_human(
    world: &mut World,
    actor: String,
    action: String,
    target: String,
    message: String,
) {
    #[allow(deprecated)]
    human(Box::leak(actor.into_boxed_str()), Box::leak(action.into_boxed_str()), &target, message);
}

#[then("the narration is captured")]
pub async fn then_narration_captured(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "Expected narration to be captured");
}

#[then(regex = "^the captured narration has actor (.+)$")]
pub async fn then_captured_has_actor(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].actor, expected);
}

#[then(regex = "^the captured narration has action (.+)$")]
pub async fn then_captured_has_action(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].action, expected);
}

#[then(regex = "^the captured narration has target (.+)$")]
pub async fn then_captured_has_target(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].target, expected);
}

#[then(regex = "^the captured narration has human (.+)$")]
pub async fn then_captured_has_human(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].human, expected);
}

#[then(regex = "^the captured narration human text does not contain (.+)$")]
pub async fn then_human_not_contains(world: &mut World, text: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(!captured[0].human.contains(&text), "Human text should not contain '{}'", text);
}

#[then(regex = "^the captured narration human text contains (.+)$")]
pub async fn then_human_contains(world: &mut World, text: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(captured[0].human.contains(&text), "Human text should contain '{}'", text);
}

#[then(regex = "^the captured narration has correlation_id (.+)$")]
pub async fn then_captured_has_correlation_id(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].correlation_id.as_ref(), Some(&expected));
}

#[then(regex = "^the captured narration has session_id (.+)$")]
pub async fn then_captured_has_session_id(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].session_id.as_ref(), Some(&expected));
}

#[then(regex = "^the captured narration has pool_id (.+)$")]
pub async fn then_captured_has_pool_id(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].pool_id.as_ref(), Some(&expected));
}

#[then(regex = "^the captured narration has emitted_by (.+)$")]
pub async fn then_captured_has_emitted_by(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].emitted_by.as_ref(), Some(&expected));
}

#[then(regex = r"^the captured narration has emitted_at_ms (\d+)$")]
pub async fn then_captured_has_emitted_at_ms(world: &mut World, expected: u64) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].emitted_at_ms, Some(expected));
}

#[then(regex = "^the captured narration has trace_id (.+)$")]
pub async fn then_captured_has_trace_id(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].trace_id.as_ref(), Some(&expected));
}

#[then("the captured narration has no correlation_id")]
pub async fn then_no_correlation_id(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].correlation_id, None);
}

#[then("the captured narration has no session_id")]
pub async fn then_no_session_id(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].session_id, None);
}
