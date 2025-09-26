use std::{collections::BTreeSet, fs};

#[test]
fn metrics_lint_placeholders_compile() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let lint_path = root.join("ci/metrics.lint.json");
    let text = fs::read_to_string(&lint_path).expect("read metrics.lint.json");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse json");
    // Validate schema-ish keys exist
    assert!(v.get("required_metrics").is_some());
    assert!(v.get("label_cardinality_budget").is_some());

    // Placeholder: emulate a registry dump and check that required metric names are unique
    let req = v["required_metrics"].as_array().unwrap();
    let mut names = BTreeSet::new();
    for m in req {
        let name = m["name"].as_str().unwrap();
        assert!(names.insert(name.to_string()), "duplicate metric name {} in linter", name);
    }
}

#[test]
fn label_cardinality_budgets_are_reasonable() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let lint_path = root.join("ci/metrics.lint.json");
    let text = fs::read_to_string(&lint_path).expect("read metrics.lint.json");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse json");
    let budgets = v["label_cardinality_budget"].as_object().expect("budget object");
    for (k, val) in budgets {
        let n = val.as_i64().unwrap_or(0);
        assert!(n > 0, "budget for {} must be positive, got {}", k, n);
    }
}
