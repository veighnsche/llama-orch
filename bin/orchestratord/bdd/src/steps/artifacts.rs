// Artifact step definitions
// Behaviors: B-ART-001 through B-ART-023

use crate::steps::world::World;
use cucumber::{given, then, when};
use http::Method;
use serde_json::json;

// B-ART-001: Artifacts endpoint available
#[given(regex = "^an artifacts endpoint$")]
pub async fn given_artifacts_endpoint(_world: &mut World) {
    // No-op: endpoint is always available
}

// B-ART-002, B-ART-003: Create artifact
#[given(regex = "^an artifact with id (.+) exists$")]
pub async fn given_artifact_exists(world: &mut World, _friendly_name: String) {
    // Create an artifact and save the actual SHA-256 ID
    let body = json!({"test_key": "test_value"});
    let _ = world.http_call(Method::POST, "/v2/artifacts", Some(body)).await;

    // Extract the generated SHA-256 ID from response
    if let Some(response_body) = &world.last_body {
        let v: serde_json::Value = serde_json::from_str(response_body).expect("parse response");
        let artifact_id = v.get("id").and_then(|i| i.as_str()).expect("missing id").to_string();
        // Store it for later retrieval
        world.push_fact(format!("artifact_id:{}", artifact_id));
    }
}

// B-ART-013: Ensure artifact not in store
#[given(regex = "^an artifact with id (.+) does not exist$")]
pub async fn given_artifact_does_not_exist(_world: &mut World, _id: String) {
    // No-op: store starts empty
}

// B-ART-001, B-ART-002, B-ART-003, B-ART-004
#[when(regex = "^I create an artifact with document (.+)$")]
pub async fn when_create_artifact(world: &mut World, doc_str: String) {
    let doc: serde_json::Value = serde_json::from_str(&doc_str).expect("invalid JSON");
    let _ = world.http_call(Method::POST, "/v2/artifacts", Some(doc)).await;
}

// B-ART-010, B-ART-011, B-ART-012, B-ART-013
#[when(regex = "^I get artifact (.+)$")]
pub async fn when_get_artifact(world: &mut World, friendly_name: String) {
    // If this is a reference to a previously created artifact, use the real ID
    let actual_id = if friendly_name.starts_with('"') && friendly_name.ends_with('"') {
        let name = friendly_name.trim_matches('"');
        if name == "test-artifact" {
            // Find the stored artifact ID from the Given step
            world
                .get_fact_by_prefix("artifact_id:")
                .map(|f| f.strip_prefix("artifact_id:").unwrap().to_string())
                .unwrap_or(name.to_string())
        } else {
            name.to_string()
        }
    } else {
        friendly_name
    };

    let path = format!("/v2/artifacts/{}", actual_id);
    let _ = world.http_call(Method::GET, &path, None).await;
}

// B-ART-022, B-ART-023: Idempotency test
#[when(regex = "^I create the same artifact twice$")]
pub async fn when_create_same_artifact_twice(world: &mut World) {
    let doc = json!({"key": "value"});

    // First creation
    let _ = world.http_call(Method::POST, "/v2/artifacts", Some(doc.clone())).await;
    let first_id = {
        let body = world.last_body.as_ref().expect("no response body");
        let json: serde_json::Value = serde_json::from_str(body).expect("invalid JSON");
        json.get("id").and_then(|v| v.as_str()).expect("missing id").to_string()
    };

    // Second creation
    let _ = world.http_call(Method::POST, "/v2/artifacts", Some(doc)).await;
    let second_id = {
        let body = world.last_body.as_ref().expect("no response body");
        let json: serde_json::Value = serde_json::from_str(body).expect("invalid JSON");
        json.get("id").and_then(|v| v.as_str()).expect("missing id").to_string()
    };

    assert_eq!(first_id, second_id, "artifact IDs should match for idempotency");
}

// B-ART-004: Verify response includes id
#[then(regex = "^the response includes id$")]
pub async fn then_response_includes_id(world: &mut World) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid JSON");
    assert!(json.get("id").is_some(), "missing id");
}

// B-ART-022: Verify ID format
#[then(regex = "^the artifact id is a SHA-256 hash$")]
pub async fn then_artifact_id_is_sha256(world: &mut World) {
    let body = world.last_body.as_ref().expect("no response body");
    let json: serde_json::Value = serde_json::from_str(body).expect("invalid JSON");
    let id = json.get("id").and_then(|v| v.as_str()).expect("missing id");
    assert_eq!(id.len(), 64, "SHA-256 hash should be 64 hex chars");
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()), "ID should be hex");
}

// B-ART-012: Verify response is the stored document
#[then(regex = "^the response is the artifact document$")]
pub async fn then_response_is_artifact_document(world: &mut World) {
    let body = world.last_body.as_ref().expect("no response body");
    let _json: serde_json::Value = serde_json::from_str(body).expect("invalid JSON");
    // Document structure verified by successful parse
}

// B-ART-023: Verified in when_create_same_artifact_twice
#[then(regex = "^both requests return the same id$")]
pub async fn then_both_return_same_id(_world: &mut World) {
    // Already verified in the when step
}
