//! OpenAI API handlers
//!
//! HTTP handlers that implement OpenAI-compatible endpoints.
//! These translate OpenAI requests to rbee Operations.

use crate::types::*;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

/// Handle POST /openai/v1/chat/completions
///
/// Translates OpenAI chat completion request to rbee Infer operation.
///
/// # Implementation TODO
///
/// 1. Extract prompt from messages array
/// 2. Map model name to rbee model ID
/// 3. Create rbee Operation::Infer
/// 4. If streaming: return SSE stream
/// 5. If non-streaming: wait for completion and return full response
pub async fn chat_completions(
    State(_state): State<()>, // TODO: Add proper state type
    Json(_request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Handle GET /openai/v1/models
///
/// Lists available models in OpenAI format.
///
/// # Implementation TODO
///
/// 1. Query rbee model catalog
/// 2. Transform to OpenAI ModelInfo format
/// 3. Return ModelListResponse
pub async fn list_models(
    State(_state): State<()>, // TODO: Add proper state type
) -> Result<Json<ModelListResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Handle GET /openai/v1/models/{model}
///
/// Get details for a specific model.
///
/// # Implementation TODO
///
/// 1. Query rbee model catalog for model ID
/// 2. Transform to OpenAI ModelInfo format
/// 3. Return 404 if not found
pub async fn get_model(
    State(_state): State<()>, // TODO: Add proper state type
    _model_id: String,
) -> Result<Json<ModelInfo>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}
