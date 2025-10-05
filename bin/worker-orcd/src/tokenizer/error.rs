//
// Spec: M0-W-1362

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Vocabulary error: {0}")]
    Vocab(#[from] VocabError),
    #[error("Merge error: {0}")]
    Merge(#[from] MergeError),
    #[error("Encode error: {0}")]
    Encode(#[from] EncodeError),
    #[error("Decode error: {0}")]
    Decode(#[from] DecodeError),
    #[error("Tokenizer not found. Searched paths: {}", searched_paths.join(", "))]
    NotFound { searched_paths: Vec<String> },
    #[error("Failed to load tokenizer: {0}")]
    LoadFailed(String),
    #[error("Encoding failed: {0}")]
    EncodeFailed(String),
    #[error("Decoding failed: {0}")]
    DecodeFailed(String),
}

#[derive(Error, Debug)]
pub enum VocabError {
    #[error("Missing vocabulary metadata: {0}")]
    MissingMetadata(String),
    
    #[error("Invalid vocabulary size: expected {expected}, got {actual}")]
    InvalidSize { expected: usize, actual: usize },
    
    #[error("Invalid special token ID: {token_type} = {id}, vocab_size = {vocab_size}")]
    InvalidSpecialToken {
        token_type: String,
        id: u32,
        vocab_size: u32,
    },
    
    #[error("Duplicate token at ID {id}: {token}")]
    DuplicateToken { id: u32, token: String },
}

#[derive(Error, Debug)]
pub enum MergeError {
    #[error("Missing merges metadata")]
    MissingMetadata,
    
    #[error("Malformed merge line: {line}")]
    MalformedLine { line: String },
    
    #[error("Invalid merge count: {count}")]
    InvalidCount { count: usize },
}

#[derive(Error, Debug)]
pub enum EncodeError {
    #[error("Unknown token: {token}")]
    UnknownToken { token: String },
    
    #[error("Empty input")]
    EmptyInput,
    
    #[error("Encoding failed: {reason}")]
    EncodingFailed { reason: String },
}

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("Unknown token ID: {id}")]
    UnknownTokenId { id: u32 },
    
    #[error("Invalid UTF-8 sequence")]
    InvalidUtf8,
    
    #[error("Empty token sequence")]
    EmptySequence,
    
    #[error("Decoding failed: {reason}")]
    DecodingFailed { reason: String },
}
