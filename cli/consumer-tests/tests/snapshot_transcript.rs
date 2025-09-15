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

#[test]
fn inline_snapshot_session_info() {
    let session = json!({
        "ttl_ms_remaining": 600000,
        "turns": 1,
        "kv_bytes": 0,
        "kv_warmth": false
    });
    insta::assert_yaml_snapshot!(session, @r###"
---
kv_bytes: 0
kv_warmth: false
ttl_ms_remaining: 600000
turns: 1
"###);
}

#[test]
fn inline_snapshot_sse_transcript() {
    let transcript = "event: started\n\n\
                      data: {\"queue_position\":3,\"predicted_start_ms\":420}\n\n\
                      event: token\n\n\
                      data: {\"t\":\"Hello\",\"i\":0}\n\n\
                      event: end\n\n\
                      data: {\"tokens_out\":1,\"decode_ms\":100}\n\n";
    insta::assert_snapshot!(transcript, @r###"
event: started

data: {"queue_position":3,"predicted_start_ms":420}

event: token

data: {"t":"Hello","i":0}

event: end

data: {"tokens_out":1,"decode_ms":100}

"###);
}
