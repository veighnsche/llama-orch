use axum::body::Body;
use axum::Router;
use http::Request;
use orchestratord::app::router::build_router;
use orchestratord::state::AppState;
use tower::util::ServiceExt as _; // for Router::oneshot

#[tokio::test]
async fn capabilities_requires_api_key_and_sets_corr_id() {
    let app: Router = build_router(AppState::new());

    // Missing key -> 401
    let req = Request::builder()
        .method(http::Method::GET)
        .uri("/v1/capabilities")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::UNAUTHORIZED);

    // Invalid key -> 403
    let req = Request::builder()
        .method(http::Method::GET)
        .uri("/v1/capabilities")
        .header("X-API-Key", "nope")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::FORBIDDEN);

    // Valid key -> 200 and correlation id present
    let req = Request::builder()
        .method(http::Method::GET)
        .uri("/v1/capabilities")
        .header("X-API-Key", "valid")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
    assert!(resp.headers().get("X-Correlation-Id").is_some());
}

#[tokio::test]
async fn metrics_is_public_and_has_corr_id() {
    let app: Router = build_router(AppState::new());

    // /metrics should be accessible without API key and include correlation id
    let req =
        Request::builder().method(http::Method::GET).uri("/metrics").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
    assert!(resp.headers().get("X-Correlation-Id").is_some());
    let ctype = resp.headers().get(http::header::CONTENT_TYPE).unwrap();
    assert_eq!(ctype, "text/plain; version=0.0.4");
}
