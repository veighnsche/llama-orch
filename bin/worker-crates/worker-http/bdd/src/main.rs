//! BDD test runner for worker-http
//!
//! Tests HTTP server lifecycle, SSE streaming, and graceful shutdown.

mod steps;

use cucumber::World;
use steps::HttpWorld;

#[tokio::main]
async fn main() {
    HttpWorld::cucumber().run_and_exit("tests/features").await;
}
