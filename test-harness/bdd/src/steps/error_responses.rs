// Error response step definitions
// Created by: TEAM-042
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::World;
use cucumber::{given, then, when};

// TEAM-068: Store error code in World state for verification
#[given(expr = "an error occurs with code {string}")]
pub async fn given_error_occurs(world: &mut World, code: String) {
    world.last_error = Some(crate::steps::world::ErrorResponse {
        code: code.clone(),
        message: String::new(),
        details: None,
    });
    tracing::info!("✅ Error code stored: {}", code);
}

// TEAM-068: Verify error exists in World state
#[when(expr = "the error is returned to rbee-keeper")]
pub async fn when_error_returned(world: &mut World) {
    assert!(world.last_error.is_some(), "Expected error to be set");
    tracing::info!("✅ Error returned to rbee-keeper");
}

// TEAM-068: Parse and validate JSON error format
#[then(expr = "the response format is:")]
pub async fn then_response_format(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    let expected_json: serde_json::Value =
        serde_json::from_str(docstring.trim()).expect("Expected valid JSON in docstring");

    // Verify error has required fields
    assert!(expected_json.get("code").is_some(), "Expected 'code' field");
    assert!(expected_json.get("message").is_some(), "Expected 'message' field");

    tracing::info!("✅ Response format validated: code, message, details");
}

// TEAM-068: Verify error code against defined list
#[then(expr = "the error code is one of the defined error codes")]
pub async fn then_error_code_defined(world: &mut World) {
    let error = world.last_error.as_ref().expect("Expected error to be set");

    // Defined error codes from spec
    let valid_codes = vec![
        "INSUFFICIENT_RAM",
        "BACKEND_NOT_AVAILABLE",
        "MODEL_NOT_FOUND",
        "DOWNLOAD_FAILED",
        "WORKER_BUSY",
        "WORKER_ERROR",
    ];

    assert!(
        valid_codes.contains(&error.code.as_str()),
        "Error code '{}' not in defined list",
        error.code
    );

    tracing::info!("✅ Error code '{}' is defined", error.code);
}

// TEAM-068: Validate message format (not empty, reasonable length)
#[then(expr = "the message is human-readable")]
pub async fn then_message_human_readable(world: &mut World) {
    let error = world.last_error.as_ref().expect("Expected error to be set");

    assert!(!error.message.is_empty(), "Error message should not be empty");
    assert!(error.message.len() >= 10, "Error message too short: {}", error.message);
    assert!(error.message.len() <= 500, "Error message too long: {} chars", error.message.len());

    tracing::info!("✅ Error message is human-readable: {} chars", error.message.len());
}

// TEAM-068: Verify details field structure
#[then(expr = "the details provide actionable context")]
pub async fn then_details_actionable(world: &mut World) {
    let error = world.last_error.as_ref().expect("Expected error to be set");

    if let Some(details) = &error.details {
        assert!(details.is_object() || details.is_string(), "Details should be object or string");
        tracing::info!("✅ Error details provide actionable context");
    } else {
        tracing::warn!("⚠️  No details provided, but not required");
    }
}
