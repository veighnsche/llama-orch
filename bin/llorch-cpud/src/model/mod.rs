//! Model implementations
//!
//! Modified by: TEAM-008 - Added Llama-2 support
//!
//! IMPORTS: worker-models (GPTConfig), internal crate modules
//! CHECKPOINTS: 8-12 (Full Model)

mod gpt2;
pub mod gguf_parser;  // TEAM-008: GGUF format support

pub use gpt2::GPT2Model;
pub use gguf_parser::GGUFParser;
