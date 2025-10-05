// Tokenizer file discovery
//
// Implements logic to find tokenizer.json files relative to model files.
//
// Spec: M0-W-1361
// Story: GT-002

use std::path::{Path, PathBuf};
use super::error::TokenizerError;

/// Tokenizer file discovery
pub struct TokenizerDiscovery;

impl TokenizerDiscovery {
    /// Find tokenizer.json relative to model file
    ///
    /// Search order:
    /// 1. Same directory as model file
    /// 2. Current working directory
    /// 3. Parent directory of model file
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file (.gguf)
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to tokenizer.json if found
    /// * `Err(TokenizerError::NotFound)` - If not found in any location
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use worker_orcd::tokenizer::TokenizerDiscovery;
    ///
    /// let model_path = Path::new("/models/gpt-oss-20b/model.gguf");
    /// let tokenizer_path = TokenizerDiscovery::find_tokenizer_json(model_path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn find_tokenizer_json(model_path: &Path) -> Result<PathBuf, TokenizerError> {
        let mut searched_paths = Vec::new();
        
        // 1. Same directory as model file
        if let Some(model_dir) = model_path.parent() {
            let tokenizer_path = model_dir.join("tokenizer.json");
            searched_paths.push(tokenizer_path.display().to_string());
            
            if tokenizer_path.exists() && tokenizer_path.is_file() {
                return Ok(tokenizer_path);
            }
        }
        
        // 2. Current working directory
        let cwd_path = PathBuf::from("./tokenizer.json");
        searched_paths.push(cwd_path.display().to_string());
        
        if cwd_path.exists() && cwd_path.is_file() {
            return Ok(cwd_path);
        }
        
        // 3. Parent directory of model file
        if let Some(model_dir) = model_path.parent() {
            if let Some(parent_dir) = model_dir.parent() {
                let parent_path = parent_dir.join("tokenizer.json");
                searched_paths.push(parent_path.display().to_string());
                
                if parent_path.exists() && parent_path.is_file() {
                    return Ok(parent_path);
                }
            }
        }
        
        Err(TokenizerError::NotFound { searched_paths })
    }
    
    /// Validate tokenizer.json file is valid JSON
    ///
    /// Performs basic validation without fully parsing the tokenizer.
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    ///
    /// # Returns
    /// * `Ok(())` - If file is valid JSON
    /// * `Err(TokenizerError)` - If file is invalid or cannot be read
    pub fn validate_tokenizer_json(path: &Path) -> Result<(), TokenizerError> {
        use std::fs;
        
        // Check file exists
        if !path.exists() {
            return Err(TokenizerError::NotFound {
                searched_paths: vec![path.display().to_string()],
            });
        }
        
        // Check file is readable
        let contents = fs::read_to_string(path)
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to read file: {}", e)))?;
        
        // Check file is valid JSON
        serde_json::from_str::<serde_json::Value>(&contents)
            .map_err(|e| TokenizerError::LoadFailed(format!("Invalid JSON: {}", e)))?;
        
        Ok(())
    }
    
    /// Find and validate tokenizer.json
    ///
    /// Combines find_tokenizer_json and validate_tokenizer_json.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to valid tokenizer.json
    /// * `Err(TokenizerError)` - If not found or invalid
    pub fn find_and_validate(model_path: &Path) -> Result<PathBuf, TokenizerError> {
        let tokenizer_path = Self::find_tokenizer_json(model_path)?;
        Self::validate_tokenizer_json(&tokenizer_path)?;
        Ok(tokenizer_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;
    
    fn create_test_tokenizer_json(dir: &Path) -> PathBuf {
        let tokenizer_path = dir.join("tokenizer.json");
        let mut file = fs::File::create(&tokenizer_path).unwrap();
        
        // Minimal valid tokenizer.json
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {},
                "merges": []
            }
        }"#;
        
        file.write_all(json.as_bytes()).unwrap();
        tokenizer_path
    }
    
    #[test]
    fn test_find_in_model_directory() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        fs::File::create(&model_path).unwrap();
        
        let tokenizer_path = create_test_tokenizer_json(temp_dir.path());
        
        let found = TokenizerDiscovery::find_tokenizer_json(&model_path).unwrap();
        assert_eq!(found, tokenizer_path);
    }
    
    #[test]
    fn test_find_in_current_directory() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("subdir").join("model.gguf");
        fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        fs::File::create(&model_path).unwrap();
        
        // Create tokenizer.json in current directory (temp_dir root)
        create_test_tokenizer_json(temp_dir.path());
        
        // Change to temp_dir
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();
        
        let result = TokenizerDiscovery::find_tokenizer_json(&model_path);
        
        // Restore original directory
        std::env::set_current_dir(original_dir).unwrap();
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        fs::File::create(&model_path).unwrap();
        
        // Don't create tokenizer.json
        let result = TokenizerDiscovery::find_tokenizer_json(&model_path);
        
        assert!(result.is_err());
        match result {
            Err(TokenizerError::NotFound { searched_paths }) => {
                assert!(!searched_paths.is_empty());
                assert!(searched_paths[0].contains("tokenizer.json"));
            }
            _ => panic!("Expected NotFound error"),
        }
    }
    
    #[test]
    fn test_validate_valid_json() {
        let temp_dir = TempDir::new().unwrap();
        let tokenizer_path = create_test_tokenizer_json(temp_dir.path());
        
        let result = TokenizerDiscovery::validate_tokenizer_json(&tokenizer_path);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_invalid_json() {
        let temp_dir = TempDir::new().unwrap();
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = fs::File::create(&tokenizer_path).unwrap();
        file.write_all(b"not valid json").unwrap();
        
        let result = TokenizerDiscovery::validate_tokenizer_json(&tokenizer_path);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_find_and_validate() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        fs::File::create(&model_path).unwrap();
        
        create_test_tokenizer_json(temp_dir.path());
        
        let result = TokenizerDiscovery::find_and_validate(&model_path);
        assert!(result.is_ok());
    }
}

// ---
// Crafted by GPT-Gamma 🤖
