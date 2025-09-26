use orchestratord::services::streaming::render_sse_for_task;
use orchestratord::state::AppState;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn sse_event_order_and_transcript_persisted() {
    let state = AppState::new();

    // Render SSE
    let sse = render_sse_for_task(&state, "t-x".to_string()).await;

    // Validate event order: started < token < metrics < end
    let a = sse.find("event: started").unwrap_or(usize::MAX);
    let b = sse.find("event: token").unwrap_or(usize::MAX);
    let m = sse.find("event: metrics").unwrap_or(usize::MAX);
    let c = sse.find("event: end").unwrap_or(usize::MAX);
    assert!(a < b && b < m && m < c, "SSE order incorrect: {}", sse);

    // Validate persisted transcript artifact has all events
    let guard = state.artifacts.lock().unwrap();
    let mut found = false;
    for (_id, doc) in guard.iter() {
        if let Some(events) = doc.get("events").and_then(|e| e.as_array()) {
            let mut got_started = false;
            let mut got_token = false;
            let mut got_metrics = false;
            let mut got_end = false;
            for ev in events {
                if let Some(t) = ev.get("type").and_then(|t| t.as_str()) {
                    match t {
                        "started" => got_started = true,
                        "token" => got_token = true,
                        "metrics" => got_metrics = true,
                        "end" => got_end = true,
                        _ => {}
                    }
                }
            }
            if got_started && got_token && got_metrics && got_end {
                found = true;
                break;
            }
        }
    }
    assert!(
        found,
        "expected persisted SSE transcript artifact with all events"
    );
}

#[tokio::test]
async fn cancel_before_first_token_yields_no_tokens() {
    let state = AppState::new();
    // Mark canceled before render
    state.cancellations.lock().unwrap().insert("t-a".into());
    let sse = render_sse_for_task(&state, "t-a".to_string()).await;
    // Expect started then end, no token or metrics
    assert!(sse.contains("event: started"));
    assert!(sse.contains("event: end"));
    assert!(
        !sse.contains("event: token"),
        "unexpected token event: {}",
        sse
    );
    assert!(
        !sse.contains("event: metrics"),
        "unexpected metrics event: {}",
        sse
    );
}

#[tokio::test]
async fn cancel_between_tokens_yields_single_token_and_no_metrics() {
    let state = AppState::new();
    // Insert cancellation after 5ms (before 30ms inter-token window elapses)
    let canc = state.cancellations.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(5)).await;
        canc.lock().unwrap().insert("t-b".into());
    });
    let sse = render_sse_for_task(&state, "t-b".to_string()).await;
    let token_count = sse.matches("event: token").count();
    assert_eq!(
        token_count, 1,
        "expected exactly one token, got {}: {}",
        token_count, sse
    );
    assert!(
        !sse.contains("event: metrics"),
        "unexpected metrics event: {}",
        sse
    );
}

#[tokio::test]
async fn cancel_before_metrics_after_two_tokens_yields_no_metrics() {
    let state = AppState::new();
    // Insert cancellation after ~35ms to land between second token and metrics (30ms + small delay)
    let canc = state.cancellations.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(35)).await;
        canc.lock().unwrap().insert("t-c".into());
    });
    let sse = render_sse_for_task(&state, "t-c".to_string()).await;
    let token_count = sse.matches("event: token").count();
    assert_eq!(token_count, 2, "expected two tokens before cancel: {}", sse);
    assert!(
        !sse.contains("event: metrics"),
        "unexpected metrics event: {}",
        sse
    );
}
