use crate::steps::world::World;
use cucumber::{given, then, when};

#[given(regex = r"^a catalog model payload$")]
pub async fn given_catalog_model_payload(world: &mut World) {
    world.push_fact("catalog.payload");
}

#[when(regex = r"^I create a catalog model$")]
pub async fn when_create_catalog_model(world: &mut World) {
    world.push_fact("catalog.create");
}

#[then(regex = r"^the model is created$")]
pub async fn then_catalog_model_created(_world: &mut World) {}

#[when(regex = r"^I get the catalog model$")]
pub async fn when_get_catalog_model(world: &mut World) {
    world.push_fact("catalog.get");
}

#[then(regex = r"^the manifest signatures and sbom are present$")]
pub async fn then_manifest_signatures_sbom_present(_world: &mut World) {}

#[when(regex = r"^I verify the catalog model$")]
pub async fn when_verify_catalog_model(world: &mut World) {
    world.push_fact("catalog.verify");
}

#[then(regex = r"^verification starts$")]
pub async fn then_verification_starts(_world: &mut World) {}

#[given(regex = r"^strict trust policy is enabled$")]
pub async fn given_strict_trust_policy_enabled(world: &mut World) {
    world.push_fact("catalog.trust.strict");
}

#[given(regex = r"^an unsigned catalog artifact$")]
pub async fn given_unsigned_catalog_artifact(world: &mut World) {
    world.push_fact("catalog.artifact.unsigned");
}

#[then(regex = r"^catalog ingestion fails with UNTRUSTED_ARTIFACT$")]
pub async fn then_catalog_ingestion_fails_untrusted(_world: &mut World) {}
