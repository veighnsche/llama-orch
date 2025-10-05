// Tokenizer module for byte-level BPE tokenization
//
// Implements GGUF-based vocabulary and merge parsing, plus BPE encoding/decoding
// for Llama-family models (Qwen, Phi-3, Llama 2/3).
//
// Spec: M0-W-1362

pub mod decoder;
pub mod encoder;
pub mod error;
pub mod merges;
pub mod streaming;
pub mod vocab;

pub use decoder::BPEDecoder;
pub use encoder::BPEEncoder;
pub use error::{DecodeError, EncodeError, MergeError, TokenizerError, VocabError};
pub use merges::{MergePair, MergeTable, MergesParser};
pub use streaming::StreamingDecoder;
pub use vocab::{VocabParser, Vocabulary};
