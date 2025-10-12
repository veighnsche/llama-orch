// Created by: TEAM-DX-001
// Error types for DX tool

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DxError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Selector not found: {selector}")]
    SelectorNotFound { selector: String },
    
    #[error("Class not found in stylesheet: {class}")]
    ClassNotFound { class: String },
    
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
    
    #[error("Timeout after {timeout}s")]
    Timeout { timeout: u64 },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, DxError>;
