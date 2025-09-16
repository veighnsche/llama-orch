use crate::steps::world::World;
use cucumber::{then, when};

#[when(regex = r"^I set model state Deprecated with deadline_ms$")]
pub async fn when_set_state_deprecated_with_deadline(world: &mut World) {
    world.push_fact("lifecycle.deprecate");
}

#[then(regex = r"^new sessions are blocked with MODEL_DEPRECATED$")]
pub async fn then_new_sessions_blocked_model_deprecated(_world: &mut World) {}

#[when(regex = r"^I set model state Retired$")]
pub async fn when_set_state_retired(world: &mut World) {
    world.push_fact("lifecycle.retire");
}

#[then(regex = r"^pools unload and archives retained$")]
pub async fn then_pools_unload_archives_retained(_world: &mut World) {}

#[then(regex = r"^model_state gauge is exported$")]
pub async fn then_model_state_gauge_exported(_world: &mut World) {}
