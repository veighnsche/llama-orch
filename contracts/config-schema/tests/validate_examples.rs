use contracts_config_schema::build_schema;
use jsonschema::{Draft, JSONSchema};
use serde_json::json;

#[test]
fn example_config_validates() {
    // Build the schema
    let schema = build_schema();
    let compiled = JSONSchema::options()
        .with_draft(Draft::Draft7)
        .compile(&serde_json::to_value(&schema).unwrap())
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

    let result = compiled.validate(&cfg);
    if let Err(errors) = result {
        for e in errors {
            eprintln!("schema error: {}", e);
        }
        panic!("example config failed schema validation");
    }
}
