//! Assertion step definitions

use cucumber::then;
use input_validation::ValidationError;

use super::world::BddWorld;

// Then steps - assertions

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
    assert!(world.last_failed(), "Expected validation to fail, but it succeeded");
}

#[then(expr = "the error should be {string}")]
async fn then_error_should_be(world: &mut BddWorld, error_type: String) {
    let error = world.get_last_error().expect("Expected an error, but validation succeeded");

    let matches = match error_type.as_str() {
        "Empty" => matches!(error, ValidationError::Empty),
        "TooLong" => matches!(error, ValidationError::TooLong { .. }),
        "InvalidCharacters" => matches!(error, ValidationError::InvalidCharacters { .. }),
        "NullByte" => matches!(error, ValidationError::NullByte),
        "PathTraversal" => matches!(error, ValidationError::PathTraversal),
        "WrongLength" => matches!(error, ValidationError::WrongLength { .. }),
        "InvalidHex" => matches!(error, ValidationError::InvalidHex { .. }),
        "OutOfRange" => matches!(error, ValidationError::OutOfRange { .. }),
        "ControlCharacter" => matches!(error, ValidationError::ControlCharacter { .. }),
        "AnsiEscape" => matches!(error, ValidationError::AnsiEscape),
        "ShellMetacharacter" => matches!(error, ValidationError::ShellMetacharacter { .. }),
        "PathOutsideRoot" => matches!(error, ValidationError::PathOutsideRoot),
        _ => panic!("Unknown error type: {}", error_type),
    };

    assert!(matches, "Expected error type '{}', but got: {:?}", error_type, error);
}

#[then("the validation should reject SQL injection")]
async fn then_rejects_sql_injection(world: &mut BddWorld) {
    assert!(world.last_failed(), "Expected SQL injection to be rejected");

    let error = world.get_last_error().unwrap();
    assert!(
        matches!(error, ValidationError::ShellMetacharacter { .. }),
        "Expected ShellMetacharacter error for SQL injection, got: {:?}",
        error
    );
}

#[then("the validation should reject command injection")]
async fn then_rejects_command_injection(world: &mut BddWorld) {
    assert!(world.last_failed(), "Expected command injection to be rejected");

    let error = world.get_last_error().unwrap();
    assert!(
        matches!(error, ValidationError::ShellMetacharacter { .. }),
        "Expected ShellMetacharacter error for command injection, got: {:?}",
        error
    );
}

#[then("the validation should reject log injection")]
async fn then_rejects_log_injection(world: &mut BddWorld) {
    assert!(world.last_failed(), "Expected log injection to be rejected");

    let error = world.get_last_error().unwrap();
    assert!(
        matches!(error, ValidationError::ShellMetacharacter { .. })
            || matches!(error, ValidationError::AnsiEscape),
        "Expected ShellMetacharacter or AnsiEscape error for log injection, got: {:?}",
        error
    );
}

#[then("the validation should reject path traversal")]
async fn then_rejects_path_traversal(world: &mut BddWorld) {
    assert!(world.last_failed(), "Expected path traversal to be rejected");

    let error = world.get_last_error().unwrap();
    assert!(
        matches!(error, ValidationError::PathTraversal),
        "Expected PathTraversal error, got: {:?}",
        error
    );
}
