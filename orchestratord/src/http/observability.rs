use crate::metrics;
use axum::response::{IntoResponse, Response};
use http::{header::CONTENT_TYPE, HeaderMap};

pub async fn metrics_endpoint() -> Response {
    let body = metrics::gather_metrics_text();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/plain; version=0.0.4".parse().unwrap());
    (headers, body).into_response()
}
