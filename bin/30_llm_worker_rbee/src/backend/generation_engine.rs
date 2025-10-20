// TEAM-149: Created for real-time streaming implementation
//! Generation engine that processes requests in `spawn_blocking`
//!
//! This module implements the generation loop pattern from candle-vllm:
//! - Runs in `spawn_blocking` to avoid blocking async runtime
//! - Processes requests sequentially from the queue
//! - Sends tokens through channels as they're generated
//! - Locks backend only for the duration of one request
//!
//! Reference: reference/candle-vllm/src/openai/pipelines/llm_engine.rs

use super::inference::CandleInferenceBackend;
use super::request_queue::{GenerationRequest, TokenResponse};
use super::sampling;
use crate::token_output_stream::TokenOutputStream;
use anyhow::{Context, Result};
use candle_core::Tensor;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Generation engine that processes inference requests
///
/// This runs in a `spawn_blocking` task to avoid blocking the async runtime.
/// It pulls requests from the queue and generates tokens one by one,
/// sending them through channels to the HTTP handlers.
pub struct GenerationEngine {
    backend: Arc<Mutex<CandleInferenceBackend>>,
    request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
}

impl GenerationEngine {
    /// Create a new generation engine
    ///
    /// # Arguments
    /// * `backend` - The inference backend (shared with warmup, etc.)
    /// * `request_rx` - Receiver for generation requests from HTTP handlers
    pub fn new(
        backend: Arc<Mutex<CandleInferenceBackend>>,
        request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
    ) -> Self {
        Self { backend, request_rx }
    }

    /// Start the generation engine loop
    ///
    /// This spawns a blocking task that processes requests sequentially.
    /// The task runs until the request channel is closed.
    ///
    /// CRITICAL: This uses `spawn_blocking` to move CPU-intensive work
    /// off the async runtime, preventing it from blocking HTTP handlers.
    pub fn start(mut self) {
        tokio::task::spawn_blocking(move || {
            // Get tokio runtime handle for async operations within blocking context
            let rt = tokio::runtime::Handle::current();

            tracing::info!("Generation engine started");

            loop {
                // Wait for next request (blocking is OK here, we're in spawn_blocking)
                let request = if let Some(req) = rt.block_on(self.request_rx.recv()) {
                    req
                } else {
                    tracing::info!("Request channel closed, stopping generation engine");
                    break;
                };

                tracing::info!(
                    request_id = %request.request_id,
                    prompt_len = request.prompt.len(),
                    max_tokens = request.config.max_tokens,
                    "Processing generation request"
                );

                // Lock backend for this request only
                // CRITICAL: Lock is held only during generation, not while waiting for requests
                let mut backend = self.backend.lock().unwrap();

                // Generate tokens and send through channel
                if let Err(e) = Self::generate_streaming(
                    &mut backend,
                    &request.prompt,
                    &request.config,
                    request.response_tx,
                ) {
                    tracing::error!(
                        request_id = %request.request_id,
                        error = %e,
                        "Generation failed"
                    );
                }

                // Lock is released here, next request can proceed
                tracing::debug!(
                    request_id = %request.request_id,
                    "Request completed, backend lock released"
                );
            }

            tracing::info!("Generation engine stopped");
        });
    }

    /// Generate tokens and stream them through the channel
    ///
    /// This is the core generation loop that:
    /// 1. Tokenizes the prompt
    /// 2. Resets the KV cache
    /// 3. Generates tokens one by one
    /// 4. Sends each token immediately through the channel
    /// 5. Stops on EOS or `max_tokens`
    ///
    /// CRITICAL: Tokens are sent as soon as they're decoded, not batched!
    fn generate_streaming(
        backend: &mut CandleInferenceBackend,
        prompt: &str,
        config: &crate::common::SamplingConfig,
        response_tx: mpsc::UnboundedSender<TokenResponse>,
    ) -> Result<()> {
        // Tokenize prompt
        let encoding = backend
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let mut tokens = encoding.get_ids().to_vec();

        tracing::debug!(prompt_tokens = tokens.len(), "Prompt tokenized");

        // Reset cache to clear any previous state
        backend.model.reset_cache().context("Failed to reset cache before inference")?;
        tracing::debug!("Cache reset before generation");

        // Create sampling components
        let mut logits_processor = sampling::create_logits_processor(config);
        let mut token_stream = TokenOutputStream::new(backend.tokenizer.clone());

        // Generate tokens one by one
        for pos in 0..config.max_tokens {
            let pos_usize = pos as usize;

            // Prepare input tensor
            let input_ids = if pos == 0 {
                // First iteration: use all prompt tokens
                Tensor::new(&tokens[..], &backend.device)?.unsqueeze(0)? // Add batch dimension: [seq_len] -> [1, seq_len]
            } else {
                // Subsequent iterations: only last token
                Tensor::new(&[tokens[tokens.len() - 1]], &backend.device)?.unsqueeze(0)?
                // Add batch dimension: [1] -> [1, 1]
            };

            // Forward pass through model
            let logits = backend.model.forward(&input_ids, pos_usize)?;

            // Get logits for last position
            let logits = logits.squeeze(0)?;
            let logits =
                if logits.dims().len() > 1 { logits.get(logits.dims()[0] - 1)? } else { logits };

            // Sample next token
            let next_token = logits_processor.sample(&logits)?;

            // Check for EOS
            let tokenizer_eos_id = backend.tokenizer.token_to_id("</s>");
            let is_eos = tokenizer_eos_id.map_or_else(
                || next_token == backend.model.eos_token_id(),
                |eos_id| next_token == eos_id,
            );

            if is_eos {
                tracing::debug!(
                    pos = pos,
                    next_token = next_token,
                    "EOS token detected, stopping generation"
                );
                break;
            }

            // Decode token and send IMMEDIATELY through channel
            // CRITICAL: This is where real-time streaming happens!
            if let Some(token_str) = token_stream
                .next_token(next_token)
                .map_err(|e| anyhow::anyhow!("Token decode failed: {}", e))?
            {
                // Send token as soon as it's generated
                if response_tx.send(TokenResponse::Token(token_str)).is_err() {
                    // Client disconnected, stop generation
                    tracing::debug!("Client disconnected, stopping generation");
                    return Ok(());
                }
            }

            tokens.push(next_token);

            // Log progress every 10 tokens
            if (pos + 1) % 10 == 0 {
                tracing::debug!(tokens_generated = pos + 1, "Generation progress");
            }
        }

        // Send any remaining decoded bytes
        if let Some(rest) = token_stream
            .decode_rest()
            .map_err(|e| anyhow::anyhow!("Failed to decode rest: {}", e))?
        {
            let _ = response_tx.send(TokenResponse::Token(rest));
        }

        // Send done signal
        let _ = response_tx.send(TokenResponse::Done);

        tracing::debug!("Generation completed successfully");

        Ok(())
    }
}
