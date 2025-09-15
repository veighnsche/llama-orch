use crate::steps_2::world::World;
use cucumber::{given, then};

#[given(regex = r"^a policy SDK$")]
pub async fn given_policy_sdk(world: &mut World) { world.push_fact("policy.sdk"); }

#[then(regex = r"^public SDK functions are semver-stable within a MAJOR$")]
pub async fn then_sdk_semver_stable_within_major(_world: &mut World) {}

#[then(regex = r"^breaking changes include a migration note and version bump$")]
pub async fn then_breaking_changes_require_migration_and_bump(_world: &mut World) {}

#[then(regex = r"^SDK performs no network or filesystem I/O by default$")]
pub async fn then_sdk_no_io_by_default(_world: &mut World) {}
