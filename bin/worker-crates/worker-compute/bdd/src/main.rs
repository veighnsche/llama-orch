//! BDD test runner for worker-compute
//!
//! Tests compute backend trait implementations.

mod steps;

use cucumber::World;
use steps::ComputeWorld;

#[tokio::main]
async fn main() {
    ComputeWorld::cucumber()
        .run_and_exit("tests/features")
        .await;
}
