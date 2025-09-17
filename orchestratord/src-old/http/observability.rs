use crate::metrics;
use axum::response::{IntoResponse, Response};
use http::{header::CONTENT_TYPE, HeaderMap};
use uuid::Uuid;

fn new_correlation_id() -> String { Uuid::new_v4().to_string() }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn metrics_generates_correlation_id_and_content_type() {
        let resp = metrics_endpoint().await;
        assert_eq!(resp.headers().get(CONTENT_TYPE).unwrap(), "text/plain; version=0.0.4");
        let cid = resp.headers().get("X-Correlation-Id").unwrap();
        assert!(!cid.to_str().unwrap().is_empty());
    }
}

pub async fn metrics_endpoint() -> Response {
    let body = metrics::gather_metrics_text();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/plain; version=0.0.4".parse().unwrap());
    headers.insert("X-Correlation-Id", new_correlation_id().parse().unwrap());
    (headers, body).into_response()
}
