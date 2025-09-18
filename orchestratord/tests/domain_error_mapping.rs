use axum::body::to_bytes;
use axum::response::IntoResponse as _;
use http::StatusCode;
use orchestratord::domain::error::OrchestratorError as ErrO;

#[tokio::test]
async fn invalid_params_maps_to_400_with_code() {
    let err = ErrO::InvalidParams("bad".into());
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["code"], "INVALID_PARAMS");
}

#[tokio::test]
async fn deadline_unmet_maps_to_400_with_code() {
    let err = ErrO::DeadlineUnmet;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["code"], "DEADLINE_UNMET");
}

#[tokio::test]
async fn pool_unavailable_maps_to_503_with_code() {
    let err = ErrO::PoolUnavailable;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["code"], "POOL_UNAVAILABLE");
}

#[tokio::test]
async fn internal_maps_to_500_with_code() {
    let err = ErrO::Internal;
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["code"], "INTERNAL");
}

#[tokio::test]
async fn admission_reject_sets_retry_headers() {
    let err = ErrO::AdmissionReject { policy_label: "reject".into(), retry_after_ms: Some(1000) };
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let headers = resp.headers().clone();
    assert_eq!(headers.get("Retry-After").unwrap(), "1");
    assert_eq!(headers.get("X-Backoff-Ms").unwrap(), "1000");
    let body = to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["code"], "ADMISSION_REJECT");
    assert_eq!(v["policy_label"], "reject");
    assert_eq!(v["retry_after_ms"], 1000);
}
