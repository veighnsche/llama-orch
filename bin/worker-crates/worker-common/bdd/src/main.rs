//! BDD test runner for worker-common
//!
//! Tests common types, error handling, and sampling configuration.

mod steps;

use cucumber::World;
use steps::CommonWorld;

#[tokio::main]
async fn main() {
    CommonWorld::cucumber().run_and_exit("tests/features").await;
}
