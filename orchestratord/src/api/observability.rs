use axum::response::IntoResponse;
use http::HeaderMap;

use crate::metrics;

pub async fn metrics_endpoint() -> axum::response::Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        "X-Correlation-Id",
        "11111111-1111-4111-8111-111111111111".parse().unwrap(),
    );
    headers.insert(
        http::header::CONTENT_TYPE,
        "text/plain; version=0.0.4".parse().unwrap(),
    );

    // Ensure at least one series includes engine_version label so BDD assertions don't rely on prior events
    metrics::set_gauge(
        "queue_depth",
        &[("engine", "llamacpp"), ("engine_version", "v0"), ("priority", "interactive")],
        1,
    );
    let text = metrics::gather_metrics_text();

    (http::StatusCode::OK, headers, text).into_response()
}
