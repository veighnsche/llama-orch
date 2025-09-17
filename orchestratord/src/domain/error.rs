#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("invalid parameters: {0}")] InvalidParams(String),
    #[error("deadline unmet")] DeadlineUnmet,
    #[error("pool unavailable")] PoolUnavailable,
    #[error("internal error")] Internal,
}

impl OrchestratorError {
    pub fn status_code(&self) -> http::StatusCode {
        match self {
            Self::InvalidParams(_) | Self::DeadlineUnmet => http::StatusCode::BAD_REQUEST,
            Self::PoolUnavailable => http::StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal => http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
