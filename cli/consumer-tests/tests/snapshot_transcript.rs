use serde_json::json;

#[test]
fn inline_snapshot_admission() {
    let admission = json!({
        "task_id": "11111111-1111-4111-8111-111111111111",
        "queue_position": 3,
        "predicted_start_ms": 420,
        "backoff_ms": 0
    });
    insta::assert_yaml_snapshot!(admission, @r###"
---
backoff_ms: 0
predicted_start_ms: 420
queue_position: 3
task_id: 11111111-1111-4111-8111-111111111111
"###);
}
