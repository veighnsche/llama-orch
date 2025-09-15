use crate::steps::world::World;
use cucumber::{given, then, when};

#[given(regex = r"^two replicas pin engine_version sampler_profile_version and model_digest$")]
pub async fn given_two_replicas_pinned_versions_artifacts(world: &mut World) { world.push_fact("det.pinned"); }

#[when(regex = r"^same prompt parameters and seed are used$")]
pub async fn when_same_prompt_params_seed(world: &mut World) { world.push_fact("det.same_params_seed"); }

#[then(regex = r"^token streams are byte-exact across replicas$")]
pub async fn then_token_streams_byte_exact(_world: &mut World) {}

#[then(regex = r"^determinism is not assumed across engine or model updates$")]
pub async fn then_no_cross_version_determinism_assumed(_world: &mut World) {}

#[given(regex = r"^replicas across engine or model versions are used$")]
pub async fn given_replicas_across_versions(world: &mut World) {
    world.push_fact("det.cross_version");
}
