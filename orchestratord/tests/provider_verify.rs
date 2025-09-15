use openapiv3::{OpenAPI, ReferenceOr, StatusCode};
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
                for (p, pi_ref) in paths.iter() {
                    if matches_template(p, path) {
                        match pi_ref {
                            ReferenceOr::Item(pi) => {
                                item_opt = Some(pi);
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
            }
        }
    }
}
