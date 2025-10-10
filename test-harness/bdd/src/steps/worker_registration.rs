// Worker registration step definitions
// Created by: TEAM-040

use cucumber::{when, then};
use crate::steps::world::World;

#[when(expr = "rbee-hive registers the worker")]
pub async fn when_register_worker(world: &mut World) {
    tracing::debug!("Registering worker");
}

#[then(expr = "the in-memory HashMap is updated with:")]
pub async fn then_hashmap_updated(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");
    tracing::debug!("HashMap should be updated with {} fields", table.rows.len() - 1);
}

#[then(regex = r"^the registration is ephemeral \(lost on rbee-hive restart\)$")]
pub async fn then_registration_ephemeral(_world: &mut World) {
    tracing::debug!("Registration is ephemeral");
}
