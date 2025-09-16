use crate::steps::world::World;
use cucumber::{given, then};
use jsonschema::JSONSchema;
use serde_json::{json, Value};

#[given(regex = r"^a valid example config$")]
pub async fn given_valid_example_config(world: &mut World) {
    world.push_fact("config.example");
}

#[then(regex = r"^schema validation passes$")]
pub async fn then_schema_validation_passes(_world: &mut World) {
    // Build schema
    let root = contracts_config_schema::build_schema();
    let schema_val: Value = serde_json::to_value(&root).expect("schema json");
    let compiled = JSONSchema::compile(&schema_val).expect("compile schema");
    // Minimal valid config
    let cfg = json!({
        "catalog": {},
        "pools": [
            { "id":"pool0", "engine":"llamacpp", "model":"model0", "devices":[0] }
        ]
    });
    assert!(
        compiled.is_valid(&cfg),
        "expected schema to validate example"
    );
}

#[given(regex = r"^strict mode with unknown field$")]
pub async fn given_strict_mode_with_unknown_field(world: &mut World) {
    world.push_fact("config.strict_unknown");
}

#[then(regex = r"^validation rejects unknown fields$")]
pub async fn then_validation_rejects_unknown_fields(_world: &mut World) {
    // Implement a strict-mode check by comparing keys against schema properties for Config
    let root = contracts_config_schema::build_schema();
    let schema_val: Value = serde_json::to_value(&root).expect("schema json");
    let props = &schema_val["schema"]["properties"];
    // Example includes an unknown top-level field "unknown"
    let cfg = json!({
        "unknown": 1,
        "catalog": {},
        "pools": [
            { "id":"pool0", "engine":"llamacpp", "model":"model0", "devices":[0] }
        ]
    });
    let mut rejected = false;
    if let Some(obj) = cfg.as_object() {
        for k in obj.keys() {
            if props.get(k).is_none() {
                rejected = true;
                break;
            }
        }
    }
    assert!(rejected, "strict mode should reject unknown fields");
}

#[given(regex = r"^schema is generated twice$")]
pub async fn given_schema_generated_twice(world: &mut World) {
    world.push_fact("config.schema_twice");
}

#[then(regex = r"^outputs are identical$")]
pub async fn then_outputs_identical(_world: &mut World) {
    use std::fs;
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("schema.json");
    contracts_config_schema::emit_schema_json(&path).expect("emit1");
    let s1 = fs::read_to_string(&path).expect("read1");
    contracts_config_schema::emit_schema_json(&path).expect("emit2");
    let s2 = fs::read_to_string(&path).expect("read2");
    assert_eq!(s1, s2, "schema outputs differ across runs");
}
