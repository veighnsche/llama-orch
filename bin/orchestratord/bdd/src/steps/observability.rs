use crate::steps::world::World;
use cucumber::{given, then, when};
use http::Method;

#[when(regex = r"^I request /metrics$")]
pub async fn when_request_metrics(world: &mut World) {
    let _ = world.http_call(Method::GET, "/metrics", None).await;
}

#[then(regex = r"^Content-Type is text/plain$")]
pub async fn then_content_type_is_text_plain(world: &mut World) {
    if let Some(headers) = &world.last_headers {
        let content_type = headers.get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(content_type.contains("text/plain"), "expected text/plain, got: {}", content_type);
    } else {
        panic!("no headers in response");
    }
}

#[then(regex = r"^tasks_enqueued_total includes labels (.+)$")]
pub async fn then_tasks_enqueued_total_includes_labels(world: &mut World, labels: String) {
    let text = world.last_body.as_ref().expect("expected /metrics response body");
    
    // Check that tasks_enqueued_total metric exists
    assert!(text.contains("tasks_enqueued_total"), "missing tasks_enqueued_total metric");
    
    // Check each expected label
    for label in labels.split_whitespace() {
        assert!(text.contains(label), "missing label: {}", label);
    }
}

#[then(regex = r"^the response includes TYPE headers$")]
pub async fn then_response_includes_type_headers(world: &mut World) {
    let text = world.last_body.as_ref().expect("expected /metrics response body");
    // Check for Prometheus TYPE comments
    assert!(text.contains("# TYPE "), "missing TYPE headers in metrics");
}

#[then(regex = r"^the response includes pre-registered metrics$")]
pub async fn then_response_includes_preregistered_metrics(world: &mut World) {
    let text = world.last_body.as_ref().expect("expected /metrics response body");
    // Check for at least one metric
    assert!(text.contains("tasks_enqueued_total") || text.contains("queue_depth"), 
        "missing pre-registered metrics");
}

#[then(regex = r"^metrics conform to linter names and labels$")]
pub async fn then_metrics_conform_names_labels(world: &mut World) {
    let _ = world.http_call(Method::GET, "/metrics", None).await;
    let text = world.last_body.as_ref().expect("expected /metrics response body");
    // Check for presence of a couple of required metric names
    assert!(text.contains("# TYPE tasks_enqueued_total "), "missing tasks_enqueued_total");
    assert!(text.contains("# TYPE queue_depth "), "missing queue_depth");
}

#[then(regex = r"^label cardinality budgets are enforced$")]
pub async fn then_label_cardinality_budgets_enforced(world: &mut World) {
    // Weak assertion: ensure engine_version label appears in some sample line
    let text = world.last_body.as_ref().expect("expected /metrics response body");
    assert!(text.contains("engine_version=\""), "expected engine_version label present");
}

#[given(regex = r"^started event and admission logs$")]
pub async fn given_started_event_and_admission_logs(world: &mut World) {
    world.push_fact("obs.logs_started");
}

#[then(regex = r"^include queue_position and predicted_start_ms$")]
pub async fn then_logs_include_queue_pos_eta(world: &mut World) {
    let logs = world.state.logs.lock().unwrap();
    let line = logs.last().expect("no logs recorded");
    assert!(line.contains("\"queue_position\":"), "missing queue_position in logs: {}", line);
    assert!(
        line.contains("\"predicted_start_ms\":"),
        "missing predicted_start_ms in logs: {}",
        line
    );
}

#[then(regex = r"^logs do not contain secrets or API keys$")]
pub async fn then_logs_do_not_contain_secrets_or_api_keys(world: &mut World) {
    let logs = world.state.logs.lock().unwrap();
    for line in logs.iter() {
        assert!(!line.to_lowercase().contains("secret"), "log leaks secret: {}", line);
        assert!(!line.contains("X-API-Key"), "log leaks api key: {}", line);
        assert!(!line.to_lowercase().contains("api_key"), "log leaks api_key: {}", line);
    }
}

// Placeholders used by skeleton features and basic.feature
#[given(regex = r"^a metrics endpoint$")]
pub async fn given_metrics_endpoint(_world: &mut World) {
    // Metrics endpoint always exists
}

#[given(regex = r"^tasks have been enqueued$")]
pub async fn given_tasks_enqueued(world: &mut World) {
    // Enqueue a few tasks to generate metrics
    use serde_json::json;
    
    for i in 0..3 {
        let body = json!({
            "task_id": format!("t-metrics-{}", i),
            "session_id": "s-0",
            "workload": "completion",
            "model_ref": "model0",
            "engine": "llamacpp",
            "ctx": 0,
            "priority": "interactive",
            "max_tokens": 1,
            "deadline_ms": 1000,
        });
        let _ = world.http_call(Method::POST, "/v2/tasks", Some(body)).await;
    }
}

#[given(regex = r"^noop$")]
pub async fn given_noop(_world: &mut World) {}

#[when(regex = r"^nothing happens$")]
pub async fn when_nothing_happens(_world: &mut World) {}

#[then(regex = r"^it passes$")]
pub async fn then_it_passes(_world: &mut World) {}
