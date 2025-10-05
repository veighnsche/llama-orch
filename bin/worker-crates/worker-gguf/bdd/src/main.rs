//! BDD test runner for worker-gguf
//!
//! Tests GGUF file format parsing and metadata extraction.

mod steps;

use cucumber::World;
use steps::GGUFWorld;

#[tokio::main]
async fn main() {
    GGUFWorld::cucumber()
        .run_and_exit("tests/features")
        .await;
}
