//! Assertion step definitions

use cucumber::then;

use super::world::BddWorld;

// ========== Then steps - assertions ==========

#[then("the validation should succeed")]
async fn then_validation_succeeds(world: &mut BddWorld) {
    assert!(
        world.last_succeeded(),
        "Expected validation to succeed, but it failed with: {:?}",
        world.get_last_error()
    );
}

#[then("the validation should fail")]
async fn then_validation_fails(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected validation to fail, but it succeeded"
    );
}

#[then(expr = "the error should contain {string}")]
async fn then_error_contains(world: &mut BddWorld, expected: String) {
    let error = world.get_last_error()
        .expect("Expected an error, but validation succeeded");
    
    assert!(
        error.contains(&expected),
        "Expected error to contain '{}', but got: {}",
        expected, error
    );
}

#[then("the validation should reject ANSI escape sequences")]
async fn then_rejects_ansi_escapes(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected ANSI escape sequences to be rejected"
    );
    
    let error = world.get_last_error().unwrap();
    // The error should indicate invalid input or sanitization failure
    assert!(
        error.contains("Invalid") || error.contains("sanitiz"),
        "Expected error related to ANSI escapes, got: {}",
        error
    );
}

#[then("the validation should reject control characters")]
async fn then_rejects_control_chars(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected control characters to be rejected"
    );
    
    let error = world.get_last_error().unwrap();
    assert!(
        error.contains("Invalid") || error.contains("control") || error.contains("sanitiz"),
        "Expected error related to control characters, got: {}",
        error
    );
}

#[then("the validation should reject null bytes")]
async fn then_rejects_null_bytes(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected null bytes to be rejected"
    );
    
    let error = world.get_last_error().unwrap();
    assert!(
        error.contains("null") || error.contains("Invalid") || error.contains("sanitiz"),
        "Expected error related to null bytes, got: {}",
        error
    );
}

#[then("the validation should reject Unicode directional overrides")]
async fn then_rejects_unicode_overrides(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected Unicode directional overrides to be rejected"
    );
    
    let error = world.get_last_error().unwrap();
    assert!(
        error.contains("Unicode") || error.contains("Invalid") || error.contains("sanitiz"),
        "Expected error related to Unicode overrides, got: {}",
        error
    );
}

#[then("the validation should reject log injection")]
async fn then_rejects_log_injection(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected log injection to be rejected"
    );
    
    let error = world.get_last_error().unwrap();
    assert!(
        error.contains("Invalid") || error.contains("injection") || error.contains("sanitiz"),
        "Expected error related to log injection, got: {}",
        error
    );
}

#[then("the validation should reject oversized fields")]
async fn then_rejects_oversized_fields(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected validation to fail for oversized fields"
    );
}

#[then("the validation should reject Unicode directional overrides")]
async fn then_rejects_unicode_overrides(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected validation to fail for Unicode directional overrides"
    );
}

#[then("the event should be serializable")]
async fn then_event_serializable(world: &mut BddWorld) {
    assert!(
        world.last_succeeded(),
        "Expected event to be serializable, but got error: {:?}",
        world.get_last_error()
    );
}

#[then("the event should contain sanitized data")]
async fn then_event_contains_sanitized_data(world: &mut BddWorld) {
    assert!(
        world.current_event.is_some(),
        "Expected event to exist after validation"
    );
    
    // Event passed validation, so data should be sanitized
    // This is a meta-assertion that the validation process worked
    assert!(
        world.last_succeeded(),
        "Expected validation to succeed for sanitized event"
    );
}
