//! Middleware stubs (auth, correlation-id, error mapping).
//! These are placeholders; handlers implement header checks directly for now.

pub struct MiddlewareConfig {
    pub require_api_key: bool,
}

impl Default for MiddlewareConfig {
    fn default() -> Self { Self { require_api_key: true } }
}
