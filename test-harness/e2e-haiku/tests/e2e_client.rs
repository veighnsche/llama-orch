use test_harness_e2e_haiku as hh;

#[tokio::test]
#[ignore]
async fn e2e_enqueue_and_stream_gated() {
    if std::env::var("REQUIRE_REAL_LLAMA").ok().as_deref() != Some("1") {
        eprintln!("skipping: REQUIRE_REAL_LLAMA=1 not set");
        return;
    }
    let base = std::env::var("ORCH_BASE").unwrap_or_else(|_| "http://localhost:8080".into());
    let task = hh::build_task(
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    );
    let resp = hh::enqueue(&base, &task).await.unwrap();
    assert!(resp.status().is_success());

    // SSE stream shape (placeholder)
    let sse = hh::stream(&base, &task.task_id).await.unwrap();
    assert_eq!(sse.status(), 200);
}
