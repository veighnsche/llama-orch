// Step definitions for test capture behaviors

use cucumber::{then, when};
use observability_narration_core::{narrate, CaptureAdapter, NarrationFields};
use crate::steps::world::World;

#[when("I install a capture adapter")]
pub async fn when_install_adapter(world: &mut World) {
    world.adapter = Some(CaptureAdapter::install());
}

#[when(regex = r#"^I narrate with human "([^"]+)"$"#)]
pub async fn when_narrate_with_human(_world: &mut World, human: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human,
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate with actor "([^"]+)"$"#)]
pub async fn when_narrate_with_actor(_world: &mut World, actor: String) {
    narrate(NarrationFields {
        actor: Box::leak(actor.into_boxed_str()),
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("I narrate without correlation_id")]
pub async fn when_narrate_without_correlation_id(_world: &mut World) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("I narrate without provenance")]
pub async fn when_narrate_without_provenance(_world: &mut World) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("I clear the capture adapter")]
pub async fn when_clear_adapter(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.clear();
}

#[then("the adapter is available")]
pub async fn then_adapter_available(world: &mut World) {
    assert!(world.adapter.is_some(), "Adapter should be available");
}

#[then(regex = r"^captured events count is (\d+)$")]
pub async fn then_captured_count(world: &mut World, expected: usize) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert_eq!(captured.len(), expected, 
        "Expected {} captured events, got {}", expected, captured.len());
}

#[then(regex = r#"^assert_includes "([^"]+)" passes$"#)]
pub async fn then_assert_includes_passes(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_includes(&substring);
}

#[then(regex = r#"^assert_includes "([^"]+)" fails$"#)]
pub async fn then_assert_includes_fails(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_includes(&substring);
    }));
    assert!(result.is_err(), "assert_includes should have failed");
}

#[then(regex = r#"^assert_field "([^"]+)" "([^"]+)" passes$"#)]
pub async fn then_assert_field_passes(world: &mut World, field: String, value: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_field(&field, &value);
}

#[then(regex = r#"^assert_field "([^"]+)" "([^"]+)" fails$"#)]
pub async fn then_assert_field_fails(world: &mut World, field: String, value: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_field(&field, &value);
    }));
    assert!(result.is_err(), "assert_field should have failed");
}

#[then("assert_correlation_id_present passes")]
pub async fn then_assert_correlation_id_present_passes(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_correlation_id_present();
}

#[then("assert_correlation_id_present fails")]
pub async fn then_assert_correlation_id_present_fails(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_correlation_id_present();
    }));
    assert!(result.is_err(), "assert_correlation_id_present should have failed");
}

#[then("assert_provenance_present passes")]
pub async fn then_assert_provenance_present_passes(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_provenance_present();
}

#[then("assert_provenance_present fails")]
pub async fn then_assert_provenance_present_fails(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_provenance_present();
    }));
    assert!(result.is_err(), "assert_provenance_present should have failed");
}
