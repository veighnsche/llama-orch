use crate::steps::world::World;
use cucumber::{given, then, when};
use http::Method;
use serde_json::json;

#[given(regex = r"^a catalog model payload$")]
pub async fn given_catalog_model_payload(world: &mut World) {
    world.push_fact("catalog.payload");
}

#[when(regex = r"^I create a catalog model$")]
pub async fn when_create_catalog_model(world: &mut World) {
    world.push_fact("catalog.create");
    // Determine if unsigned/strict scenario was set
    let mut signed = true;
    for ev in world.all_facts() {
        if let Some(stage) = ev.get("stage").and_then(|v| v.as_str()) {
            if stage == "catalog.artifact.unsigned" {
                signed = false;
            }
        }
    }
    let body = json!({ "id": "model0", "signed": signed });
    let _ = world
        .http_call(Method::POST, "/v1/catalog", Some(body))
        .await;
}

#[then(regex = r"^the model is created$")]
pub async fn then_catalog_model_created(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::CREATED));
}

#[when(regex = r"^I get the catalog model$")]
pub async fn when_get_catalog_model(world: &mut World) {
    world.push_fact("catalog.get");
    let _ = world
        .http_call(Method::GET, "/v1/catalog/model0", None)
        .await;
}

#[then(regex = r"^the manifest signatures and sbom are present$")]
pub async fn then_manifest_signatures_sbom_present(world: &mut World) {
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert!(v.get("signatures").is_some());
    assert!(v.get("sbom").is_some());
}

#[when(regex = r"^I verify the catalog model$")]
pub async fn when_verify_catalog_model(world: &mut World) {
    world.push_fact("catalog.verify");
    let _ = world
        .http_call(Method::POST, "/v1/catalog/model0/verify", None)
        .await;
}

#[then(regex = r"^verification starts$")]
pub async fn then_verification_starts(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::ACCEPTED));
}

#[given(regex = r"^strict trust policy is enabled$")]
pub async fn given_strict_trust_policy_enabled(world: &mut World) {
    world.push_fact("catalog.trust.strict");
    world
        .extra_headers
        .push(("X-Trust-Policy".into(), "strict".into()));
}

#[given(regex = r"^an unsigned catalog artifact$")]
pub async fn given_unsigned_catalog_artifact(world: &mut World) {
    world.push_fact("catalog.artifact.unsigned");
}

#[then(regex = r"^catalog ingestion fails with UNTRUSTED_ARTIFACT$")]
pub async fn then_catalog_ingestion_fails_untrusted(world: &mut World) {
    assert_eq!(world.last_status, Some(http::StatusCode::BAD_REQUEST));
    let v: serde_json::Value = serde_json::from_str(world.last_body.as_ref().unwrap()).unwrap();
    assert_eq!(v["code"], "UNTRUSTED_ARTIFACT");
}
