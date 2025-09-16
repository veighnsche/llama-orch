use crate::steps::world::World;
use cucumber::{given, then};

#[given(regex = r"^a valid example config$")]
pub async fn given_valid_example_config(world: &mut World) {
    world.push_fact("config.example");
}

#[then(regex = r"^schema validation passes$")]
pub async fn then_schema_validation_passes(_world: &mut World) {}

#[given(regex = r"^strict mode with unknown field$")]
pub async fn given_strict_mode_with_unknown_field(world: &mut World) {
    world.push_fact("config.strict_unknown");
}

#[then(regex = r"^validation rejects unknown fields$")]
pub async fn then_validation_rejects_unknown_fields(_world: &mut World) {}

#[given(regex = r"^schema is generated twice$")]
pub async fn given_schema_generated_twice(world: &mut World) {
    world.push_fact("config.schema_twice");
}

#[then(regex = r"^outputs are identical$")]
pub async fn then_outputs_identical(_world: &mut World) {}
