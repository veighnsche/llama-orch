use openapiv3::{OpenAPI, Operation, ReferenceOr, StatusCode};
use serde_json::Value;
use std::{fs, path::PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf()
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
                let method = interaction["request"]["method"]
                    .as_str()
                    .unwrap_or("")
                    .to_lowercase();
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
                        let retry = hdrs
                            .get("Retry-After")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let backoff = hdrs
                            .get("X-Backoff-Ms")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
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
        ("post", "/v1/tasks", 202) => {
            let body = interaction["response"]["body"]
                .as_object()
                .expect("202 body must be JSON object");
            assert!(has_keys(
                body,
                &[
                    "task_id",
                    "queue_position",
                    "predicted_start_ms",
                    "backoff_ms"
                ]
            ));
        }
        ("post", "/v1/tasks", 400 | 429 | 500 | 503) => {
            let body = interaction["response"]["body"]
                .as_object()
                .expect("error body must be JSON object");
            assert!(has_keys(body, &["code", "message", "engine"]));
        }
        ("get", "/v1/sessions/{id}", 200) => {
            let body = interaction["response"]["body"]
                .as_object()
                .expect("session body must be JSON object");
            assert!(has_keys(
                body,
                &["ttl_ms_remaining", "turns", "kv_bytes", "kv_warmth"]
            ));
        }
        ("delete", "/v1/sessions/{id}", 204) => {
            assert!(
                interaction["response"]["body"].is_null()
                    || interaction["response"]["body"].is_object()
                        && interaction["response"]["body"]
                            .as_object()
                            .unwrap()
                            .is_empty(),
                "204 should have no body"
            );
        }
        ("get", "/v1/tasks/{id}/stream", 200) => {
            // SSE responses are strings
            assert!(
                interaction["response"]["body"].is_string(),
                "SSE transcript should be string"
            );
        }
        _ => {
            // Best-effort: nothing to validate
        }
    }
}

#[test]
fn rejects_unknown_paths_or_statuses() {
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
    let unknown_path = "/v1/does-not-exist";
    let mut found = false;
    for (p, _) in paths.iter() {
        if matches_template(p, unknown_path) {
            found = true;
            break;
        }
    }
    assert!(
        !found,
        "unknown path should not be matched by any OpenAPI path"
    );

    // Known path but unknown status
    // Pick /v1/tasks POST, status 418
    use openapiv3::ReferenceOr as R;
    let item = match paths.paths.get("/v1/tasks").unwrap() {
        R::Item(it) => it,
        _ => panic!("unexpected ref"),
    };
    let op = item.post.as_ref().expect("post op exists");
    let has_418 = op
        .responses
        .responses
        .keys()
        .any(|code| matches!(code, StatusCode::Code(418)));
    assert!(
        !has_418,
        "teapot status must not be declared for POST /v1/tasks"
    );
}
