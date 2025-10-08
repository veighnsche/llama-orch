use contracts_config_schema::build_schema;
use jsonschema::{Validator, Draft};
use serde_json::json;

#[test]
fn v32_fields_validate() {
    let schema = build_schema();
    let compiled = Validator::options()
        .with_draft(Draft::Draft7)
        .build(&serde_json::to_value(&schema).unwrap())
        .expect("schema compiles");

    let cfg = json!({
        "catalog": {
            "trust_policy": {
                "mode": "permissive",
                "allowed_registries": ["registry.example.com"],
                "require_signature": false,
                "require_sbom": false
            }
        },
        "pools": [
            {
                "id": "llama3-8b",
                "engine": "llamacpp",
                "model": "sha256:feedface",
                "devices": [0],
                "queue": { "capacity": 128, "full_policy": "reject" },
                "admission": {
                    "priorities": [
                        {"name": "interactive", "queue_capacity": 96, "rate_limit_rps": 20},
                        {"name": "batch", "queue_capacity": 32, "rate_limit_rps": 5}
                    ]
                },
                "timeouts": {"wall_ms": 60000, "idle_ms": 5000}
            }
        ]
    });

    if let Err(error) = compiled.validate(&cfg) {
        eprintln!("Validation error: {}", error);
        panic!("v3.2 example config failed schema validation");
    }
}
