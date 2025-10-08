use contracts_config_schema::build_schema;
use jsonschema::{Draft, Validator};
use serde_json::json;

#[test]
fn example_config_validates() {
    // Build the schema
    let schema = build_schema();
    let compiled = Validator::options()
        .with_draft(Draft::Draft7)
        .build(&serde_json::to_value(&schema).unwrap())
        .expect("schema compiles");

    // Example config from SPEC §8 (simplified)
    let cfg = json!({
        "pools": [
            {
                "id": "llama3-8b-q4-gpu0",
                "engine": "llamacpp",
                "model": "sha256:deadbeef",
                "ctx": 8192,
                "devices": [0],
                "tensor_split": null,
                "preload": true,
                "require_same_engine_version": true,
                "sampler_profile_version": "v1",
                "queue": { "capacity": 256, "full_policy": "reject" }
            }
        ]
    });

    if let Err(error) = compiled.validate(&cfg) {
        eprintln!("Validation error: {}", error);
        panic!("example config failed schema validation");
    }
}
