use contracts_config_schema::build_schema;
use jsonschema::{Draft, JSONSchema};
use serde_json::json;

#[test]
fn v32_fields_validate() {
    let schema = build_schema();
    let compiled = JSONSchema::options()
        .with_draft(Draft::Draft7)
        .compile(&serde_json::to_value(&schema).unwrap())
        .expect("schema compiles");

    let cfg = json!({
        "catalog": {
            "trust_policy": {
                "mode": "strict",
                "allowed_registries": ["registry.example.com"],
                "require_signature": true,
                "require_sbom": true,
                "ca_roots": ["Example Root CA"]
            }
        },
        "pools": [
            {
                "id": "llama3-8b-wfq",
                "engine": "llamacpp",
                "model": "sha256:feedface",
                "devices": [0],
                "queue": { "capacity": 256, "full_policy": "reject" },
                "admission": {
                    "priorities": [
                        {"name": "interactive", "queue_capacity": 128, "rate_limit_rps": 20, "weight": 8},
                        {"name": "batch", "queue_capacity": 128, "rate_limit_rps": 5, "weight": 1}
                    ],
                    "fairness": {
                        "kind": "wfq",
                        "tenants": [
                            {"id": "tenantA", "weight": 5, "max_concurrent": 8, "rps": 50},
                            {"id": "tenantB", "weight": 1, "max_concurrent": 2, "rps": 10}
                        ]
                    }
                },
                "preemption": {
                    "mode": "soft",
                    "hard_requirements": {
                        "engine_capability": "interruptible_decode",
                        "max_preemptions_per_min": 10,
                        "protect_priorities": ["interactive"]
                    }
                },
                "timeouts": {"wall_ms": 60000, "idle_ms": 5000}
            }
        ]
    });

    let result = compiled.validate(&cfg);
    if let Err(errors) = result {
        for e in errors { eprintln!("schema error: {}", e); }
        panic!("v3.2 example config failed schema validation");
    }
}
