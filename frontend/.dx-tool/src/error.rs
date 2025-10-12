// Created by: TEAM-DX-001
// Error types for DX tool

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DxError {
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Selector not found: {selector}")]
    SelectorNotFound { selector: String },
    
    #[error("Class not found in stylesheet: {class}")]
    ClassNotFound { class: String },
    
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
    
    #[error("Timeout after {timeout_secs}s for URL: {url}")]
    Timeout { url: String, timeout_secs: u64 },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, DxError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    // TEAM-DX-002: Error type tests
    
    #[test]
    fn test_parse_error() {
        let err = DxError::Parse("test error".to_string());
        assert!(err.to_string().contains("Parse error"));
        assert!(err.to_string().contains("test error"));
    }
    
    #[test]
    fn test_selector_not_found_error() {
        let err = DxError::SelectorNotFound {
            selector: ".test".to_string(),
        };
        assert!(err.to_string().contains("Selector not found"));
        assert!(err.to_string().contains(".test"));
    }
    
    #[test]
    fn test_class_not_found_error() {
        let err = DxError::ClassNotFound {
            class: "cursor-pointer".to_string(),
        };
        assert!(err.to_string().contains("Class not found"));
        assert!(err.to_string().contains("cursor-pointer"));
    }
    
    #[test]
    fn test_invalid_url_error() {
        let err = DxError::InvalidUrl("not-a-url".to_string());
        assert!(err.to_string().contains("Invalid URL"));
        assert!(err.to_string().contains("not-a-url"));
    }
    
    #[test]
    fn test_timeout_error() {
        let err = DxError::Timeout { timeout: 5 };
        assert!(err.to_string().contains("Timeout"));
        assert!(err.to_string().contains("5"));
    }
    
    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
    
    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(DxError::Parse("test".to_string()));
        assert!(result.is_err());
    }
}
