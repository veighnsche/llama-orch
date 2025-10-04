//! POST /execute endpoint - Execute inference

use crate::cuda::safe::InferenceHandle;
use crate::error::WorkerError;
use crate::http::routes::AppState;
use axum::{
    extract::State,
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

#[derive(Deserialize)]
pub struct ExecuteRequest {
    pub job_id: String,
    pub prompt: String,
    pub max_tokens: i32,
    pub temperature: f32,
    pub seed: u64,
}

#[derive(Serialize)]
struct StartedEvent {
    job_id: String,
    started_at: String,
}

#[derive(Serialize)]
struct TokenEvent {
    t: String,
    i: i32,
}

#[derive(Serialize)]
struct EndEvent {
    tokens_out: i32,
}

#[derive(Serialize)]
struct ErrorEvent {
    code: String,
    message: String,
    retriable: bool,
}

#[axum::debug_handler]
/// Handle POST /execute
pub async fn handle_execute(
    State(state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, WorkerError> {
    tracing::info!(
        job_id = %req.job_id,
        prompt_len = req.prompt.len(),
        max_tokens = req.max_tokens,
        "Starting inference"
    );

    // Start inference
    let inference = InferenceHandle::start(
        &state.model,
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.seed,
    )?;

    // Create SSE stream
    let job_id = req.job_id.clone();
    let stream = stream::unfold(
        (inference, 0, false, job_id),
        |(mut inf, count, done, job_id)| async move {
            if done {
                return None;
            }

            // First event: started
            if count == 0 {
                let event = StartedEvent {
                    job_id: job_id.clone(),
                    started_at: chrono::Utc::now().to_rfc3339(),
                };
                let sse = Event::default().event("started").json_data(&event).ok()?;
                return Some((Ok(sse), (inf, 1, false, job_id)));
            }

            // Generate next token
            match inf.next_token() {
                Ok(Some((token, token_index))) => {
                    let event = TokenEvent { t: token, i: token_index };
                    let sse = Event::default().event("token").json_data(&event).ok()?;
                    Some((Ok(sse), (inf, count + 1, false, job_id)))
                }
                Ok(None) => {
                    // Inference complete
                    let event = EndEvent { tokens_out: count - 1 };
                    let sse = Event::default().event("end").json_data(&event).ok()?;
                    Some((Ok(sse), (inf, count, true, job_id)))
                }
                Err(e) => {
                    // Error during inference
                    let event = ErrorEvent {
                        code: "INFERENCE_FAILED".to_string(),
                        message: e.to_string(),
                        retriable: false,
                    };
                    let sse = Event::default().event("error").json_data(&event).ok()?;
                    Some((Ok(sse), (inf, count, true, job_id)))
                }
            }
        },
    );

    Ok(Sse::new(stream))
}

// ---
// Built by Foundation-Alpha 🏗️
