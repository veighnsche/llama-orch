// Tokenizer module for byte-level BPE tokenization
//
// Implements GGUF-based vocabulary and merge parsing, plus BPE encoding/decoding
// for Llama-family models (Qwen, Phi-3, Llama 2/3).
//
// Spec: M0-W-1362

pub mod vocab;
pub mod merges;
pub mod encoder;
pub mod decoder;
pub mod streaming;
pub mod error;
pub mod hf_json;
pub mod backend;

pub use vocab::{Vocabulary, VocabParser};
pub use merges::{MergeTable, MergePair, MergesParser};
pub use encoder::BPEEncoder;
pub use decoder::BPEDecoder;
pub use streaming::StreamingDecoder;
pub use error::{TokenizerError, VocabError, MergeError, EncodeError, DecodeError};
pub use hf_json::HfJsonTokenizer;
pub use backend::TokenizerBackend;
