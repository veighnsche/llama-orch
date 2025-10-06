//! BDD steps for worker-http

mod world;

pub use world::HttpWorld;

use cucumber::{given, then, when};
use worker_http::validation::ExecuteRequest;

// Request validation scenarios
#[given(expr = "a request with job_id {string}")]
async fn given_job_id(world: &mut HttpWorld, job_id: String) {
    let req = world.request.get_or_insert_with(|| ExecuteRequest {
        job_id: String::new(),
        prompt: "test".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    });
    req.job_id = job_id;
}

#[given(expr = "a request with prompt {string}")]
async fn given_prompt(world: &mut HttpWorld, prompt: String) {
    let req = world.request.get_or_insert_with(|| ExecuteRequest {
        job_id: "test".to_string(),
        prompt: String::new(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    });
    req.prompt = prompt;
}

#[given(expr = "a request with max_tokens {int}")]
async fn given_max_tokens(world: &mut HttpWorld, max_tokens: u32) {
    let req = world.request.get_or_insert_with(|| ExecuteRequest {
        job_id: "test".to_string(),
        prompt: "test".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    });
    req.max_tokens = max_tokens;
}

#[given(expr = "a request with temperature {float}")]
async fn given_temperature(world: &mut HttpWorld, temperature: f32) {
    let req = world.request.get_or_insert_with(|| ExecuteRequest {
        job_id: "test".to_string(),
        prompt: "test".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    });
    req.temperature = temperature;
}

#[when("I validate the request")]
async fn when_validate_request(world: &mut HttpWorld) {
    let req = world.request.as_ref().expect("request not set");
    match req.validate() {
        Ok(()) => {
            world.validation_passed = true;
        }
        Err(e) => {
            world.validation_error = Some(e.field.clone());
            world.validation_passed = false;
        }
    }
}

#[then("the validation should pass")]
async fn then_validation_passes(world: &mut HttpWorld) {
    assert!(world.validation_passed, "validation failed");
}

#[then("the validation should fail")]
async fn then_validation_fails(world: &mut HttpWorld) {
    assert!(!world.validation_passed, "validation should have failed");
}

#[then(expr = "the error field should be {string}")]
async fn then_error_field(world: &mut HttpWorld, expected: String) {
    let actual = world.validation_error.as_ref().expect("no validation error");
    assert_eq!(actual, &expected, "error field mismatch");
}

// SSE streaming scenarios
#[given("an SSE event stream")]
async fn given_sse_stream(world: &mut HttpWorld) {
    world.event_count = 0;
    world.event_types.clear();
}

#[when(expr = "I send a {string} event")]
async fn when_send_event(world: &mut HttpWorld, event_type: String) {
    world.event_count += 1;
    world.event_types.push(event_type.clone());

    if event_type == "end" || event_type == "error" {
        world.has_terminal_event = true;
    }
}

#[then(expr = "the stream should have {int} events")]
async fn then_event_count(world: &mut HttpWorld, expected: usize) {
    assert_eq!(world.event_count, expected, "event count mismatch");
}

#[then("the stream should have a terminal event")]
async fn then_has_terminal(world: &mut HttpWorld) {
    assert!(world.has_terminal_event, "no terminal event");
}

#[then(expr = "the event order should be {string}")]
async fn then_event_order(world: &mut HttpWorld, expected_order: String) {
    let actual_order = world.event_types.join(" -> ");
    assert_eq!(actual_order, expected_order, "event order mismatch");
}
