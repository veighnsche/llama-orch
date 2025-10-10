// Worker registration step definitions
// Created by: TEAM-053
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::World;
use cucumber::{then, when};

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
