// Provider verification against OpenAPI v2-only endpoints.
use openapiv3::{OpenAPI, Operation, ReferenceOr, StatusCode};
use serde_json::Value;
use serde_yaml as syaml;
use std::{fs, path::PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is bin/orchestratord; go up two levels to workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().to_path_buf()
}

#[test]
fn meta_control_and_artifacts_contracts_v2() {
    use openapiv3::ReferenceOr as R;
    let root = repo_root();

    // Meta: capabilities under /v2/meta/capabilities
    let meta_path = root.join("contracts/openapi/meta.yaml");
    let meta_spec: OpenAPI =
        serde_yaml::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
    let item = match meta_spec.paths.paths.get("/v2/meta/capabilities").expect("meta path exists") {
        R::Item(it) => it,
        _ => panic!("unexpected ref in meta paths"),
    };
    let op = item.get.as_ref().expect("GET op exists on /v2/meta/capabilities");
    assert!(op.responses.responses.keys().any(|c| matches!(c, StatusCode::Code(200))));

    // Control: pools and workers
    let control_path = root.join("contracts/openapi/control.yaml");
    let control_spec: OpenAPI =
        serde_yaml::from_str(&fs::read_to_string(&control_path).unwrap()).unwrap();
    let get_item =
        |template: &str| match control_spec.paths.paths.get(template).expect("path exists") {
            R::Item(it) => it,
            _ => panic!("unexpected ref in control paths"),
        };
    let drain = get_item("/v2/pools/{id}/drain");
    assert!(drain
        .post
        .as_ref()
        .expect("post exists")
        .responses
        .responses
        .keys()
        .any(|c| matches!(c, StatusCode::Code(202))));
    let reload = get_item("/v2/pools/{id}/reload");
    assert!(reload
        .post
        .as_ref()
        .expect("post exists")
        .responses
        .responses
        .keys()
        .any(|c| matches!(c, StatusCode::Code(202))));
    let health = get_item("/v2/pools/{id}/health");
    assert!(
        health
            .get
            .as_ref()
            .expect("get exists")
            .responses
            .responses
            .keys()
            .any(|c| matches!(c, StatusCode::Code(200)))
            || health.get.as_ref().unwrap().responses.default.is_some()
    );
    let workers = get_item("/v2/workers/register");
    assert!(workers
        .post
        .as_ref()
        .expect("post exists")
        .responses
        .responses
        .keys()
        .any(|c| matches!(c, StatusCode::Code(200))));

    // Artifacts
    let artifacts_path = root.join("contracts/openapi/artifacts.yaml");
    let artifacts_spec: OpenAPI =
        serde_yaml::from_str(&fs::read_to_string(&artifacts_path).unwrap()).unwrap();
    let item = match artifacts_spec.paths.paths.get("/v2/artifacts").expect("path exists") {
        R::Item(it) => it,
        _ => panic!("unexpected ref in artifacts paths"),
    };
    assert!(item
        .post
        .as_ref()
        .expect("post exists")
        .responses
        .responses
        .keys()
        .any(|c| matches!(c, StatusCode::Code(201))));
    let item = match artifacts_spec.paths.paths.get("/v2/artifacts/{id}").expect("path exists") {
        R::Item(it) => it,
        _ => panic!("unexpected ref in artifacts paths"),
    };
    assert!(item
        .get
        .as_ref()
        .expect("get exists")
        .responses
        .responses
        .keys()
        .any(|c| matches!(c, StatusCode::Code(200))));
}

#[test]
fn data_endpoints_and_sse_metrics_contract_v2() {
    // Load data plane OpenAPI as raw YAML for simple key checks
    let root = repo_root();
    let oapi_path = root.join("contracts/openapi/data.yaml");
    let raw: syaml::Value = syaml::from_str(&fs::read_to_string(&oapi_path).unwrap()).unwrap();

    // POST /v2/tasks exists and declares 202
    let post202 = &raw["paths"]["/v2/tasks"]["post"]["responses"]["202"];
    assert!(post202.is_mapping(), "POST /v2/tasks 202 response missing");

    // GET /v2/tasks/{id}/events exists and declares 200
    let get200 = &raw["paths"]["/v2/tasks/{id}/events"]["get"]["responses"]["200"];
    assert!(get200.is_mapping(), "GET /v2/tasks/{id}/events 200 response missing");

    // SSEMetrics fields
    let props = &raw["components"]["schemas"]["SSEMetrics"]["properties"];
    for key in [
        "on_time_probability",
        "queue_depth",
        "kv_warmth",
        "tokens_budget_remaining",
        "time_budget_remaining_ms",
        "cost_budget_remaining",
    ] {
        assert!(props.get(key).is_some(), "SSEMetrics missing {}", key);
    }
}

#[test]
fn control_openapi_sanity_v2() {
    use openapiv3::ReferenceOr as R;
    let root = repo_root();
    let oapi_path = root.join("contracts/openapi/control.yaml");
    let spec: OpenAPI = serde_yaml::from_str(&fs::read_to_string(oapi_path).unwrap()).unwrap();
    let paths = spec.paths;

    let get_op = |template: &str, method: &str| -> &Operation {
        let item = match paths.paths.get(template).expect("path exists") {
            R::Item(it) => it,
            _ => panic!("unexpected $ref in paths"),
        };
        match method {
            "get" => item.get.as_ref().expect("GET op exists"),
            "post" => item.post.as_ref().expect("POST op exists"),
            other => panic!("unsupported method {}", other),
        }
    };

    // POST /v2/pools/{id}/drain -> 202
    let drain = get_op("/v2/pools/{id}/drain", "post");
    assert!(drain.responses.responses.keys().any(|c| matches!(c, StatusCode::Code(202))));

    // POST /v2/pools/{id}/reload -> 202
    let reload = get_op("/v2/pools/{id}/reload", "post");
    assert!(reload.responses.responses.keys().any(|c| matches!(c, StatusCode::Code(202))));

    // GET /v2/pools/{id}/health -> 200
    let health = get_op("/v2/pools/{id}/health", "get");
    assert!(
        health.responses.responses.keys().any(|c| matches!(c, StatusCode::Code(200)))
            || health.responses.default.is_some()
    );
}

#[test]
fn provider_paths_match_pacts() {
    let root = repo_root();
    let oapi_path = root.join("contracts/openapi/data.yaml");
    let spec: OpenAPI = serde_yaml::from_str(&fs::read_to_string(oapi_path).unwrap()).unwrap();
    let paths = spec.paths;

    let pacts_dir = root.join("contracts/pacts");
    if !pacts_dir.exists() {
        // No pacts yet; treat as scaffold pass
        return;
    }

    for entry in fs::read_dir(&pacts_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            let pact: Value =
                serde_json::from_str(&fs::read_to_string(entry.path()).unwrap()).unwrap();
            let interactions = pact["interactions"].as_array().cloned().unwrap_or_default();
            for interaction in interactions {
                let method = interaction["request"]["method"].as_str().unwrap_or("").to_lowercase();
                let path = interaction["request"]["path"].as_str().unwrap_or("");
                let status = interaction["response"]["status"].as_u64().unwrap_or(0) as u16;

                fn matches_template(template: &str, actual: &str) -> bool {
                    let t_parts: Vec<&str> =
                        template.split('/').filter(|s| !s.is_empty()).collect();
                    let a_parts: Vec<&str> = actual.split('/').filter(|s| !s.is_empty()).collect();
                    if t_parts.len() != a_parts.len() {
                        return false;
                    }
                    for (t, a) in t_parts.iter().zip(a_parts.iter()) {
                        if t.starts_with('{') && t.ends_with('}') {
                            if a.is_empty() {
                                return false;
                            }
                            continue;
                        }
                        if t != a {
                            return false;
                        }
                    }
                    true
                }

                let mut item_opt = None;
                let mut matched_template: Option<String> = None;
                for (p, pi_ref) in paths.iter() {
                    if matches_template(p, path) {
                        match pi_ref {
                            ReferenceOr::Item(pi) => {
                                item_opt = Some(pi);
                                matched_template = Some(p.clone());
                            }
                            ReferenceOr::Reference { .. } => {
                                panic!("path {} is a $ref; unsupported in test", p)
                            }
                        }
                        break;
                    }
                }
                let item = item_opt.unwrap_or_else(|| panic!("pact path not in OpenAPI: {}", path));
                let op = match method.as_str() {
                    "get" => item.get.as_ref(),
                    "post" => item.post.as_ref(),
                    "delete" => item.delete.as_ref(),
                    "put" => item.put.as_ref(),
                    "patch" => item.patch.as_ref(),
                    other => panic!("unsupported method in pact: {}", other),
                };
                assert!(op.is_some(), "operation missing for {} {}", method, path);
                let op = op.unwrap();
                let has_status = op.responses.responses.keys().any(|code| match code {
                    StatusCode::Code(c) => *c == status,
                    _ => false,
                }) || (status == 200 && op.responses.default.is_some());
                assert!(
                    has_status,
                    "status {} not declared in OpenAPI for {} {}",
                    status, method, path
                );

                // Validate response shape against expected schema (minimal check by endpoint/status)
                validate_response_shape(
                    &method,
                    matched_template.as_deref().unwrap_or(path),
                    status,
                    op,
                    &interaction,
                );

                // Header checks for backpressure on 429
                if status == 429 {
                    if let Some(hdrs) = interaction["response"]["headers"].as_object() {
                        let retry = hdrs.get("Retry-After").and_then(|v| v.as_str()).unwrap_or("");
                        let backoff =
                            hdrs.get("X-Backoff-Ms").and_then(|v| v.as_str()).unwrap_or("");
                        // Correlation id must be present
                        assert!(
                            hdrs.contains_key("X-Correlation-Id"),
                            "missing X-Correlation-Id header on 429"
                        );
                        assert!(
                            retry.parse::<u64>().is_ok(),
                            "Retry-After must be seconds numeric string"
                        );
                        assert!(
                            backoff.parse::<u64>().is_ok(),
                            "X-Backoff-Ms must be integer string (ms)"
                        );
                    } else {
                        panic!("429 response missing headers");
                    }

                    // Body should include policy_label advisory field
                    if let Some(body) = interaction["response"]["body"].as_object() {
                        assert!(body.contains_key("policy_label"), "429 body missing policy_label");
                    }
                }

                // For other responses with headers, correlation id should be present
                if let Some(hdrs) = interaction["response"]["headers"].as_object() {
                    assert!(
                        hdrs.contains_key("X-Correlation-Id"),
                        "missing X-Correlation-Id header on response"
                    );
                }
            }
        }
    }
}

fn validate_response_shape(
    method: &str,
    template: &str,
    status: u16,
    _op: &Operation,
    interaction: &Value,
) {
    fn has_keys(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> bool {
        keys.iter().all(|k| obj.contains_key(*k))
    }
    match (method, template, status) {
        ("post", "/v2/tasks", 202) => {
            let body =
                interaction["response"]["body"].as_object().expect("202 body must be JSON object");
            assert!(has_keys(
                body,
                &["task_id", "queue_position", "predicted_start_ms", "backoff_ms"]
            ));
        }
        ("post", "/v2/tasks", 400 | 429 | 500 | 503) => {
            let body = interaction["response"]["body"]
                .as_object()
                .expect("error body must be JSON object");
            assert!(has_keys(body, &["code", "message", "engine"]));
        }
        ("get", "/v2/sessions/{id}", 200) => {
            let body = interaction["response"]["body"]
                .as_object()
                .expect("session body must be JSON object");
            assert!(has_keys(body, &["ttl_ms_remaining", "turns", "kv_bytes", "kv_warmth"]));
        }
        ("delete", "/v2/sessions/{id}", 204) => {
            assert!(
                interaction["response"]["body"].is_null()
                    || interaction["response"]["body"].is_object()
                        && interaction["response"]["body"].as_object().unwrap().is_empty(),
                "204 should have no body"
            );
        }
        ("get", "/v2/tasks/{id}/events", 200) => {
            // SSE responses are strings
            assert!(interaction["response"]["body"].is_string(), "SSE transcript should be string");
        }
        _ => {
            // Best-effort: nothing to validate
        }
    }
}

#[test]
fn rejects_unknown_paths_or_statuses_v2() {
    let root = repo_root();
    let oapi_path = root.join("contracts/openapi/data.yaml");
    let spec: OpenAPI = serde_yaml::from_str(&fs::read_to_string(oapi_path).unwrap()).unwrap();
    let paths = spec.paths;

    // Helper reuses the matching logic from main test
    let matches_template = |template: &str, actual: &str| {
        let t_parts: Vec<&str> = template.split('/').filter(|s| !s.is_empty()).collect();
        let a_parts: Vec<&str> = actual.split('/').filter(|s| !s.is_empty()).collect();
        if t_parts.len() != a_parts.len() {
            return false;
        }
        for (t, a) in t_parts.iter().zip(a_parts.iter()) {
            if t.starts_with('{') && t.ends_with('}') {
                if a.is_empty() {
                    return false;
                }
                continue;
            }
            if t != a {
                return false;
            }
        }
        true
    };

    // Unknown path
    let unknown_path = "/v2/does-not-exist";
    let mut found = false;
    for (p, _) in paths.iter() {
        if matches_template(p, unknown_path) {
            found = true;
            break;
        }
    }
    assert!(!found, "unknown path should not be matched by any OpenAPI path");

    // Known path but unknown status
    // Pick /v2/tasks POST, status 418
    use openapiv3::ReferenceOr as R;
    let item = match paths.paths.get("/v2/tasks").unwrap() {
        R::Item(it) => it,
        _ => panic!("unexpected ref"),
    };
    let op = item.post.as_ref().expect("post op exists");
    let has_418 = op.responses.responses.keys().any(|code| matches!(code, StatusCode::Code(418)));
    assert!(!has_418, "teapot status must not be declared for POST /v2/tasks");
}
