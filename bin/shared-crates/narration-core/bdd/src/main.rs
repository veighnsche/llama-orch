// BDD runner for narration-core

mod steps;

use cucumber::World as _;
use steps::world::World;

#[tokio::main]
async fn main() {
    World::cucumber().run_and_exit("tests/features").await;
}
