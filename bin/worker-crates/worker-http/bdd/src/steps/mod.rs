//! BDD steps for worker-http

mod world;

pub use world::HttpWorld;

use cucumber::{given, then, when};

// Server lifecycle scenarios
#[given(expr = "an HTTP server on port {int}")]
async fn given_http_server(world: &mut HttpWorld, port: u16) {
    world.port = Some(port);
}

#[when("I start the server")]
async fn when_start_server(world: &mut HttpWorld) {
    // TODO: Start server
    world.server_running = true;
}

#[when("I send a shutdown signal")]
async fn when_shutdown(world: &mut HttpWorld) {
    // TODO: Send shutdown
    world.shutdown_sent = true;
}

#[then("the server should be running")]
async fn then_server_running(world: &mut HttpWorld) {
    assert!(world.server_running, "server not running");
}

#[then("the server should shut down gracefully")]
async fn then_graceful_shutdown(world: &mut HttpWorld) {
    assert!(world.shutdown_sent, "shutdown not sent");
    // TODO: Verify graceful shutdown
}

// SSE streaming scenarios
#[given("an SSE stream")]
async fn given_sse_stream(world: &mut HttpWorld) {
    world.has_sse_stream = true;
}

#[when("I send a token event")]
async fn when_send_token(world: &mut HttpWorld) {
    // TODO: Send token event
    world.events_sent += 1;
}

#[when("I close the stream")]
async fn when_close_stream(world: &mut HttpWorld) {
    // TODO: Close stream
    world.stream_closed = true;
}

#[then(expr = "the client should receive {int} events")]
async fn then_receive_events(world: &mut HttpWorld, expected: usize) {
    assert_eq!(world.events_sent, expected, "event count mismatch");
}

#[then("the stream should close cleanly")]
async fn then_stream_closes(world: &mut HttpWorld) {
    assert!(world.stream_closed, "stream not closed");
}

// Request validation scenarios
#[given(expr = "a request with {string} header")]
async fn given_request_header(world: &mut HttpWorld, header: String) {
    world.request_headers.push(header);
}

#[when("I validate the request")]
async fn when_validate_request(world: &mut HttpWorld) {
    // TODO: Validate request
    world.validation_passed = true;
}

#[then("the validation should pass")]
async fn then_validation_passes(world: &mut HttpWorld) {
    assert!(world.validation_passed, "validation failed");
}

#[then("the validation should fail")]
async fn then_validation_fails(world: &mut HttpWorld) {
    assert!(!world.validation_passed, "validation should have failed");
}
