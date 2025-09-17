#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("invalid parameters: {0}")] InvalidParams(String),
    #[error("deadline unmet")] DeadlineUnmet,
    #[error("pool unavailable")] PoolUnavailable,
    #[error("internal error")] Internal,
    #[error("admission rejected: {policy_label}")] AdmissionReject { policy_label: String, retry_after_ms: Option<i64> },
    #[error("canceled: {reason}")] Canceled { reason: String },
}

impl OrchestratorError {
    pub fn status_code(&self) -> http::StatusCode {
        match self {
            Self::InvalidParams(_) | Self::DeadlineUnmet => http::StatusCode::BAD_REQUEST,
            Self::PoolUnavailable => http::StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal => http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::AdmissionReject { .. } => http::StatusCode::TOO_MANY_REQUESTS,
            Self::Canceled { .. } => http::StatusCode::NO_CONTENT,
        }
    }
}

impl axum::response::IntoResponse for OrchestratorError {
    fn into_response(self) -> axum::response::Response {
        use axum::response::IntoResponse as _;
        use contracts_api_types as api;
        let status = self.status_code();
        let mut headers = http::HeaderMap::new();
        let (code, message, retriable, retry_after_ms, policy_label, engine) = match &self {
            OrchestratorError::InvalidParams(msg) => (
                api::ErrorKind::InvalidParams,
                Some(msg.clone()),
                None,
                None,
                None,
                None,
            ),
            OrchestratorError::DeadlineUnmet => (
                api::ErrorKind::DeadlineUnmet,
                Some("deadline_ms must be > 0".to_string()),
                None,
                None,
                None,
                None,
            ),
            OrchestratorError::PoolUnavailable => (
                api::ErrorKind::PoolUnavailable,
                Some("pool unavailable".to_string()),
                Some(true),
                Some(1000),
                Some("retry".to_string()),
                None,
            ),
            OrchestratorError::Internal => (
                api::ErrorKind::Internal,
                Some("internal error".to_string()),
                None,
                None,
                None,
                None,
            ),
            OrchestratorError::AdmissionReject { policy_label, retry_after_ms } => {
                if let Some(ms) = retry_after_ms {
                    headers.insert("Retry-After", http::HeaderValue::from_str(&format!("{}", (ms/1000).max(1))).unwrap());
                    headers.insert("X-Backoff-Ms", http::HeaderValue::from_str(&format!("{}", ms)).unwrap());
                }
                (
                    api::ErrorKind::AdmissionReject,
                    Some("queue full policies applied".to_string()),
                    Some(true),
                    *retry_after_ms,
                    Some(policy_label.clone()),
                    None,
                )
            }
            OrchestratorError::Canceled { reason } => (
                api::ErrorKind::Canceled,
                Some(reason.clone()),
                None,
                None,
                None,
                None,
            ),
        };
        let env = api::ErrorEnvelope {
            code,
            message,
            engine,
            retriable,
            retry_after_ms,
            policy_label,
        };
        (status, headers, axum::Json(env)).into_response()
    }
}
