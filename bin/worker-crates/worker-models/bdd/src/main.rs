//! BDD test runner for worker-models
//!
//! Tests model adapter factory and architecture detection.

mod steps;

use cucumber::World;
use steps::ModelsWorld;

#[tokio::main]
async fn main() {
    ModelsWorld::cucumber().run_and_exit("tests/features").await;
}
