use crate::steps::world::World;
use cucumber::{given, then};

#[given(regex = r"^no API key is provided$")]
pub async fn given_no_api_key(world: &mut World) {
    world.push_fact("auth.none");
}

#[then(regex = r"^I receive 401 Unauthorized$")]
pub async fn then_401_unauthorized(_world: &mut World) {}

#[given(regex = r"^an invalid API key is provided$")]
pub async fn given_invalid_api_key(world: &mut World) {
    world.push_fact("auth.invalid");
}

#[then(regex = r"^I receive 403 Forbidden$")]
pub async fn then_403_forbidden(_world: &mut World) {}
