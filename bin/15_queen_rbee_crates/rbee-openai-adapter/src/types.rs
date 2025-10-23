//! OpenAI API types
//!
//! Type definitions matching the OpenAI API specification.
//! These will be translated to/from rbee's internal Operation types.

use serde::{Deserialize, Serialize};

/// OpenAI chat completion request
///
/// See: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,
    
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
    
    /// Sampling temperature (0-2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Whether to stream responses
    #[serde(default)]
    pub stream: bool,
    
    /// Random seed for deterministic generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: system, user, assistant
    pub role: String,
    
    /// Message content
    pub content: String,
}

/// OpenAI chat completion response (non-streaming)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique identifier
    pub id: String,
    
    /// Object type: "chat.completion"
    pub object: String,
    
    /// Unix timestamp
    pub created: u64,
    
    /// Model used
    pub model: String,
    
    /// Completion choices
    pub choices: Vec<ChatCompletionChoice>,
    
    /// Token usage statistics
    pub usage: Usage,
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    /// Choice index
    pub index: u32,
    
    /// Generated message
    pub message: ChatMessage,
    
    /// Finish reason: "stop", "length", "content_filter"
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in prompt
    pub prompt_tokens: u32,
    
    /// Tokens in completion
    pub completion_tokens: u32,
    
    /// Total tokens
    pub total_tokens: u32,
}

/// OpenAI streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier
    pub id: String,
    
    /// Object type: "chat.completion.chunk"
    pub object: String,
    
    /// Unix timestamp
    pub created: u64,
    
    /// Model used
    pub model: String,
    
    /// Streaming choices
    pub choices: Vec<ChatCompletionChunkChoice>,
}

/// Streaming choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    /// Choice index
    pub index: u32,
    
    /// Delta (partial message)
    pub delta: ChatMessageDelta,
    
    /// Finish reason (null until done)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Message delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageDelta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    
    /// Content delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Model list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// Object type: "list"
    pub object: String,
    
    /// List of models
    pub data: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    
    /// Object type: "model"
    pub object: String,
    
    /// Unix timestamp
    pub created: u64,
    
    /// Owner organization
    pub owned_by: String,
}

/// OpenAI error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,
    
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}
