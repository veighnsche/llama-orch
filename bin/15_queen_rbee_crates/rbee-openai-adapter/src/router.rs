//! OpenAI API router
//!
//! Creates the Axum router for OpenAI-compatible endpoints.

use crate::handlers;
use axum::{routing::{get, post}, Router};

/// Create OpenAI-compatible router
///
/// Returns a router that can be mounted at `/openai` prefix in queen-rbee.
///
/// # Example
///
/// ```rust,ignore
/// use rbee_openai_adapter::create_openai_router;
///
/// let openai_router = create_openai_router(state);
/// let app = Router::new()
///     .nest("/openai", openai_router)
///     .nest("/v1", rbee_router);
/// ```
pub fn create_openai_router() -> Router {
    Router::new()
        // Chat completions
        .route("/v1/chat/completions", post(handlers::chat_completions))
        
        // Models
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/:model", get(handlers::get_model))
}
