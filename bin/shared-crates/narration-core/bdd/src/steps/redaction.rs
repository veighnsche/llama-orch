// Step definitions for redaction behaviors

use crate::steps::world::World;
use cucumber::{then, when};
use observability_narration_core::{redact_secrets, RedactionPolicy};

#[when("I create a default redaction policy")]
pub async fn when_create_default_policy(world: &mut World) {
    world.redaction_policy = Some(RedactionPolicy::default());
}

#[when(regex = r#"^I redact "([^"]+)"$"#)]
pub async fn when_redact(world: &mut World, input: String) {
    world.redaction_input = input.clone();
    world.redaction_output = redact_secrets(&input, RedactionPolicy::default());
}

#[when(regex = r#"^I redact "([^"]+)" with mask_uuids enabled$"#)]
pub async fn when_redact_with_uuid_mask(world: &mut World, input: String) {
    world.redaction_input = input.clone();
    let mut policy = RedactionPolicy::default();
    policy.mask_uuids = true;
    world.redaction_output = redact_secrets(&input, policy);
}

#[when(regex = r#"^I redact "([^"]+)" with replacement "([^"]+)"$"#)]
pub async fn when_redact_with_replacement(world: &mut World, input: String, replacement: String) {
    world.redaction_input = input.clone();
    let mut policy = RedactionPolicy::default();
    policy.replacement = replacement;
    world.redaction_output = redact_secrets(&input, policy);
}

#[then("mask_bearer_tokens is true")]
pub async fn then_mask_bearer_tokens_true(world: &mut World) {
    let policy = world.redaction_policy.as_ref().expect("Policy not created");
    assert!(policy.mask_bearer_tokens);
}

#[then("mask_api_keys is true")]
pub async fn then_mask_api_keys_true(world: &mut World) {
    let policy = world.redaction_policy.as_ref().expect("Policy not created");
    assert!(policy.mask_api_keys);
}

#[then("mask_uuids is false")]
pub async fn then_mask_uuids_false(world: &mut World) {
    let policy = world.redaction_policy.as_ref().expect("Policy not created");
    assert!(!policy.mask_uuids);
}

#[then(regex = r#"^replacement is "([^"]+)"$"#)]
pub async fn then_replacement_is(world: &mut World, expected: String) {
    let policy = world.redaction_policy.as_ref().expect("Policy not created");
    assert_eq!(policy.replacement, expected);
}

#[then(regex = r#"^the output is "([^"]+)"$"#)]
pub async fn then_output_is(world: &mut World, expected: String) {
    assert_eq!(
        world.redaction_output, expected,
        "Expected output '{}', got '{}'",
        expected, world.redaction_output
    );
}

#[then(regex = r#"^the output does not contain "([^"]+)"$"#)]
pub async fn then_output_not_contains(world: &mut World, text: String) {
    assert!(
        !world.redaction_output.contains(&text),
        "Output should not contain '{}', but got '{}'",
        text,
        world.redaction_output
    );
}

#[then(regex = r#"^the output contains "([^"]+)"$"#)]
pub async fn then_output_contains(world: &mut World, text: String) {
    assert!(
        world.redaction_output.contains(&text),
        "Output should contain '{}', but got '{}'",
        text,
        world.redaction_output
    );
}
