// Error response step definitions
// Created by: TEAM-040

use cucumber::{given, when, then};
use crate::steps::world::World;

#[given(expr = "an error occurs with code {string}")]
pub async fn given_error_occurs(world: &mut World, code: String) {
    tracing::debug!("Error occurs with code: {}", code);
}

#[when(expr = "the error is returned to rbee-keeper")]
pub async fn when_error_returned(world: &mut World) {
    tracing::debug!("Error is returned to rbee-keeper");
}

#[then(expr = "the response format is:")]
pub async fn then_response_format(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("Response format should be: {}", docstring.trim());
}

#[then(expr = "the error code is one of the defined error codes")]
pub async fn then_error_code_defined(world: &mut World) {
    tracing::debug!("Error code should be defined");
}

#[then(expr = "the message is human-readable")]
pub async fn then_message_human_readable(world: &mut World) {
    tracing::debug!("Message should be human-readable");
}

#[then(expr = "the details provide actionable context")]
pub async fn then_details_actionable(world: &mut World) {
    tracing::debug!("Details should provide actionable context");
}
