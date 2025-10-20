// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-152: Implemented BDD runner for queen-lifecycle
// Purpose: BDD tests for queen-lifecycle

mod steps;

use cucumber::World as _;
use steps::world::World;

#[tokio::main]
async fn main() {
    World::cucumber().run("tests/features/").await;
}
