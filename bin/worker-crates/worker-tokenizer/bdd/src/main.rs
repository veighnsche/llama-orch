//! BDD test runner for worker-tokenizer
//!
//! Tests tokenization (BPE encoding/decoding, UTF-8 safety).

mod steps;

use cucumber::World;
use steps::TokenizerWorld;

#[tokio::main]
async fn main() {
    TokenizerWorld::cucumber().run_and_exit("tests/features").await;
}
