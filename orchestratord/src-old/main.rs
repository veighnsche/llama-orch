use axum::Router;

fn main() {
    // Initialize structured logging (JSON). In tests, double init is ignored by tracing-subscriber.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .json()
        .try_init();
    // Build the router using the library entrypoint. Do not start the server in pre-code phase.
    let app: Router<orchestratord::state::AppState> = orchestratord::build_app();
    // Avoid unused warning in pre-code phase
    let _ = app;
    println!("orchestratord routes wired (stubs)");
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn metrics_text_includes_required_names() {
        // Read linter config to get required names
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(1)
            .unwrap()
            .to_path_buf();
        let lint_path = root.join("ci/metrics.lint.json");
        let lint_text = fs::read_to_string(&lint_path).expect("read metrics.lint.json");
        let lint_json: serde_json::Value =
            serde_json::from_str(&lint_text).expect("parse metrics.lint.json");

        let text = orchestratord::metrics::gather_metrics_text();
        let names = lint_json["required_metrics"].as_array().unwrap();
        for m in names {
            let name = m["name"].as_str().unwrap();
            // Prometheus text format includes a TYPE line for each registered metric
            let needle = format!("# TYPE {} ", name);
            assert!(
                text.contains(&needle) || text.contains(name),
                "missing metric {} in /metrics text",
                name
            );
        }
    }
}
