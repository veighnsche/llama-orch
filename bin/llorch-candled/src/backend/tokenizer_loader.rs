//! Tokenizer loading with auto-detection
//!
//! Created by: TEAM-017

use anyhow::{bail, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Load tokenizer with auto-detection
///
/// TEAM-017: Tries multiple tokenizer formats in order:
/// 1. tokenizer.json (HuggingFace format)
/// 2. tokenizer.model (SentencePiece format - future support)
pub fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
    let parent = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or_else(|| Path::new("."))
    };

    // Try tokenizer.json (HuggingFace format)
    let hf_path = parent.join("tokenizer.json");
    if hf_path.exists() {
        tracing::debug!(path = ?hf_path, "Loading HuggingFace tokenizer");
        return Tokenizer::from_file(&hf_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {}", e));
    }

    // Try tokenizer.model (SentencePiece format)
    let sp_path = parent.join("tokenizer.model");
    if sp_path.exists() {
        tracing::warn!(
            path = ?sp_path,
            "Found tokenizer.model but SentencePiece support not yet implemented"
        );
        bail!("SentencePiece tokenizer support not yet implemented. Please convert to tokenizer.json format.");
    }

    bail!("No tokenizer found at {:?}. Expected tokenizer.json", parent);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_tokenizer_missing() {
        let temp_dir = TempDir::new().unwrap();
        let result = load_tokenizer(temp_dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No tokenizer found"));
    }

    #[test]
    fn test_load_tokenizer_sentencepiece_not_supported() {
        let temp_dir = TempDir::new().unwrap();
        let tokenizer_path = temp_dir.path().join("tokenizer.model");
        fs::write(&tokenizer_path, b"fake sentencepiece model").unwrap();

        let result = load_tokenizer(temp_dir.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("SentencePiece tokenizer support not yet implemented"));
    }
}
