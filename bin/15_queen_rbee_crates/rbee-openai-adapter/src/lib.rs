//! OpenAI-compatible API adapter for rbee
//!
//! This crate provides an OpenAI-compatible HTTP API that translates OpenAI API calls
//! to rbee's internal Operation types. This allows existing applications built for
//! OpenAI to work with rbee without modification.
//!
//! # Architecture
//!
//! ```text
//! External App → /openai/v1/chat/completions → OpenAI Adapter → rbee Operations → queen-rbee
//! ```
//!
//! # Endpoints
//!
//! - `POST /openai/v1/chat/completions` - Chat completions (streaming and non-streaming)
//! - `GET /openai/v1/models` - List available models
//! - `GET /openai/v1/models/{model}` - Get model details
//!
//! # Usage
//!
//! ```rust,ignore
//! use rbee_openai_adapter::create_openai_router;
//!
//! let router = create_openai_router(state);
//! // Mount at /openai prefix in queen-rbee
//! ```
//!
//! # Status
//!
//! **STUB CRATE** - Design phase only. Implementation requires:
//! 1. Research OpenAI API specification
//! 2. Map OpenAI request types to rbee Operations
//! 3. Implement request/response translation
//! 4. Add streaming SSE support
//! 5. Add error code mapping

pub mod error;
pub mod handlers;
pub mod router;
pub mod types;

pub use router::create_openai_router;
