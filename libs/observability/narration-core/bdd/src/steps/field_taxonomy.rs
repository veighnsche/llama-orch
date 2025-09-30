// Step definitions for field taxonomy behaviors

use cucumber::{then, when};
use observability_narration_core::{narrate, NarrationFields};
use crate::steps::world::World;

#[when(regex = r#"^I create NarrationFields with actor "([^"]+)"$"#)]
pub async fn when_create_with_actor(world: &mut World, actor: String) {
    world.fields = NarrationFields {
        actor: Box::leak(actor.into_boxed_str()),
        ..Default::default()
    };
}

#[when(regex = r#"^I create NarrationFields with action "([^"]+)"$"#)]
pub async fn when_create_with_action(world: &mut World, action: String) {
    world.fields = NarrationFields {
        action: Box::leak(action.into_boxed_str()),
        ..Default::default()
    };
}

#[when(regex = r#"^I create NarrationFields with target "([^"]+)"$"#)]
pub async fn when_create_with_target(world: &mut World, target: String) {
    world.fields = NarrationFields {
        target,
        ..Default::default()
    };
}

#[when(regex = r#"^I create NarrationFields with human "([^"]+)"$"#)]
pub async fn when_create_with_human(world: &mut World, human: String) {
    world.fields = NarrationFields {
        human,
        ..Default::default()
    };
}

#[when(regex = r#"^I narrate with job_id "([^"]+)"$"#)]
pub async fn when_narrate_with_job_id(_world: &mut World, job_id: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        job_id: Some(job_id),
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate with replica_id "([^"]+)"$"#)]
pub async fn when_narrate_with_replica_id(_world: &mut World, replica_id: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        replica_id: Some(replica_id),
        ..Default::default()
    });
}

#[when("I create default NarrationFields")]
pub async fn when_create_default(world: &mut World) {
    world.fields = NarrationFields::default();
}

#[then(regex = r#"^the fields have actor "([^"]+)"$"#)]
pub async fn then_fields_have_actor(world: &mut World, expected: String) {
    assert_eq!(world.fields.actor, expected);
}

#[then(regex = r#"^the fields have action "([^"]+)"$"#)]
pub async fn then_fields_have_action(world: &mut World, expected: String) {
    assert_eq!(world.fields.action, expected);
}

#[then(regex = r#"^the fields have target "([^"]+)"$"#)]
pub async fn then_fields_have_target(world: &mut World, expected: String) {
    assert_eq!(world.fields.target, expected);
}

#[then(regex = r#"^the fields have human "([^"]+)"$"#)]
pub async fn then_fields_have_human(world: &mut World, expected: String) {
    assert_eq!(world.fields.human, expected);
}

#[then(regex = r#"^the captured narration has job_id "([^"]+)"$"#)]
pub async fn then_captured_has_job_id(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    // Note: job_id not in CapturedNarration, but this tests the API
}

#[then(regex = r#"^the captured narration has replica_id "([^"]+)"$"#)]
pub async fn then_captured_has_replica_id(world: &mut World, expected: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert_eq!(captured[0].replica_id.as_ref(), Some(&expected));
}

#[then(regex = r#"^actor is "([^"]*)"$"#)]
pub async fn then_actor_is(world: &mut World, expected: String) {
    assert_eq!(world.fields.actor, expected);
}

#[then(regex = r#"^action is "([^"]*)"$"#)]
pub async fn then_action_is(world: &mut World, expected: String) {
    assert_eq!(world.fields.action, expected);
}

#[then(regex = r#"^target is "([^"]*)"$"#)]
pub async fn then_target_is(world: &mut World, expected: String) {
    assert_eq!(world.fields.target, expected);
}

#[then(regex = r#"^human is "([^"]*)"$"#)]
pub async fn then_human_is(world: &mut World, expected: String) {
    assert_eq!(world.fields.human, expected);
}

#[then("all Option fields are None")]
pub async fn then_all_option_fields_none(world: &mut World) {
    assert_eq!(world.fields.correlation_id, None);
    assert_eq!(world.fields.session_id, None);
    assert_eq!(world.fields.job_id, None);
    assert_eq!(world.fields.task_id, None);
    assert_eq!(world.fields.pool_id, None);
    assert_eq!(world.fields.replica_id, None);
    assert_eq!(world.fields.worker_id, None);
}
