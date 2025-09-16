use std::collections::BTreeSet;
use std::fs;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn parse_spec_metric_names(spec_text: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in spec_text.lines() {
        let l = line.trim();
        // Match bullet lines that look like "- metric_name" (no spaces inside the token)
        if let Some(rest) = l.strip_prefix("- ") {
            // Skip bullets that are not metric names (contain spaces or colons)
            if !rest.contains(' ')
                && !rest.contains(':')
                && rest.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                names.insert(rest.to_string());
            }
        }
    }
    names
}

fn linter_required_metric_names(lint_json: &serde_json::Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    if let Some(req) = lint_json.get("required_metrics").and_then(|v| v.as_array()) {
        for m in req {
            if let Some(name) = m.get("name").and_then(|v| v.as_str()) {
                names.insert(name.to_string());
            }
        }
    }
    names
}

#[test]
fn spec_metrics_are_in_linter_config() {
    let root = repo_root();
    let spec_path = root.join(".specs/metrics/otel-prom.md");
    let lint_path = root.join("ci/metrics.lint.json");

    let spec_text = fs::read_to_string(&spec_path).expect("read otel-prom.md");
    let lint_text = fs::read_to_string(&lint_path).expect("read metrics.lint.json");
    let lint_json: serde_json::Value =
        serde_json::from_str(&lint_text).expect("parse metrics.lint.json");

    let spec_names = parse_spec_metric_names(&spec_text);
    let lint_names = linter_required_metric_names(&lint_json);

    // Every metric enumerated in the SPEC must appear in the linter config
    for n in &spec_names {
        assert!(
            lint_names.contains(n),
            "metric {} from SPEC missing in linter config",
            n
        );
    }
}

#[test]
fn tasks_rejected_total_labels_match_spec_exception() {
    let root = repo_root();
    let lint_path = root.join("ci/metrics.lint.json");
    let lint_text = fs::read_to_string(&lint_path).expect("read metrics.lint.json");
    let lint_json: serde_json::Value =
        serde_json::from_str(&lint_text).expect("parse metrics.lint.json");

    let req = lint_json["required_metrics"]
        .as_array()
        .expect("required_metrics array");
    let mut found = false;
    for m in req {
        if m["name"].as_str() == Some("tasks_rejected_total") {
            found = true;
            let labels: Vec<String> = m["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert_eq!(
                labels,
                vec!["engine".to_string(), "reason".to_string()],
                "tasks_rejected_total labels must omit engine_version per SPEC exception"
            );
        }
    }
    assert!(
        found,
        "tasks_rejected_total must be present in linter config"
    );
}
