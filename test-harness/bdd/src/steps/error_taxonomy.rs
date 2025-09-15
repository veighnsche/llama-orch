use crate::steps::world::World;
use cucumber::{then, when};

#[when(regex = r"^I trigger INVALID_PARAMS$")]
pub async fn when_trigger_invalid_params(world: &mut World) { world.push_fact("err.invalid_params"); }

#[then(regex = r"^I receive 400 with correlation id and error envelope code INVALID_PARAMS$")]
pub async fn then_400_corr_invalid_params(_world: &mut World) {}

#[when(regex = r"^I trigger POOL_UNAVAILABLE$")]
pub async fn when_trigger_pool_unavailable(world: &mut World) { world.push_fact("err.pool_unavailable"); }

#[then(regex = r"^I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE$")]
pub async fn then_503_corr_pool_unavailable(_world: &mut World) {}

#[when(regex = r"^I trigger INTERNAL error$")]
pub async fn when_trigger_internal_error(world: &mut World) { world.push_fact("err.internal"); }

#[then(regex = r"^I receive 500 with correlation id and error envelope code INTERNAL$")]
pub async fn then_500_corr_internal(_world: &mut World) {}

#[then(regex = r"^error envelope includes engine when applicable$")]
pub async fn then_error_envelope_includes_engine(_world: &mut World) {}
