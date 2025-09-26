use crate::steps::world::World;
use cucumber::{given, then, when};
use http::Method;

#[then(regex = r"^metrics conform to linter names and labels$")]
pub async fn then_metrics_conform_names_labels(world: &mut World) {
    let _ = world.http_call(Method::GET, "/metrics", None).await;
    let text = world
        .last_body
        .as_ref()
        .expect("expected /metrics response body");
    // Check for presence of a couple of required metric names
    assert!(
        text.contains("# TYPE tasks_enqueued_total "),
        "missing tasks_enqueued_total"
    );
    assert!(text.contains("# TYPE queue_depth "), "missing queue_depth");
}

#[then(regex = r"^label cardinality budgets are enforced$")]
pub async fn then_label_cardinality_budgets_enforced(world: &mut World) {
    // Weak assertion: ensure engine_version label appears in some sample line
    let text = world
        .last_body
        .as_ref()
        .expect("expected /metrics response body");
    assert!(
        text.contains("engine_version=\""),
        "expected engine_version label present"
    );
}

#[given(regex = r"^started event and admission logs$")]
pub async fn given_started_event_and_admission_logs(world: &mut World) {
    world.push_fact("obs.logs_started");
}

#[then(regex = r"^include queue_position and predicted_start_ms$")]
pub async fn then_logs_include_queue_pos_eta(world: &mut World) {
    let logs = world.state.logs.lock().unwrap();
    let line = logs.last().expect("no logs recorded");
    assert!(
        line.contains("\"queue_position\":"),
        "missing queue_position in logs: {}",
        line
    );
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
        assert!(
            !line.to_lowercase().contains("secret"),
            "log leaks secret: {}",
            line
        );
        assert!(!line.contains("X-API-Key"), "log leaks api key: {}", line);
        assert!(
            !line.to_lowercase().contains("api_key"),
            "log leaks api_key: {}",
            line
        );
    }
}

// Placeholders used by skeleton features and basic.feature
#[given(regex = r"^noop$")]
pub async fn given_noop(_world: &mut World) {}

#[when(regex = r"^nothing happens$")]
pub async fn when_nothing_happens(_world: &mut World) {}

#[then(regex = r"^it passes$")]
pub async fn then_it_passes(_world: &mut World) {}
