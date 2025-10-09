//! Multi-model support tests
//!
//! Created by: TEAM-017

use llorch_candled::backend::models::detect_architecture;
use serde_json::json;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama_architecture_from_model_type() {
        let config = json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "vocab_size": 32000
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "llama");
    }

    #[test]
    fn test_detect_llama_architecture_from_architectures_array() {
        let config = json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "vocab_size": 32000
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "llama");
    }

    #[test]
    fn test_detect_mistral_architecture() {
        let config = json!({
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "vocab_size": 32000
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "mistral");
    }

    #[test]
    fn test_detect_phi_architecture() {
        let config = json!({
            "architectures": ["PhiForCausalLM"],
            "model_type": "phi",
            "hidden_size": 2560,
            "vocab_size": 51200
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "phi");
    }

    #[test]
    fn test_detect_qwen_architecture() {
        let config = json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 3584,
            "vocab_size": 151936
        });

        let arch = detect_architecture(&config).unwrap();
        // TEAM-017: model_type takes precedence, returns "qwen2" which normalizes to "qwen"
        assert!(arch == "qwen2" || arch == "qwen");
    }

    #[test]
    fn test_detect_gemma_architecture() {
        let config = json!({
            "architectures": ["GemmaForCausalLM"],
            "model_type": "gemma",
            "hidden_size": 2048,
            "vocab_size": 256000
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "gemma");
    }

    #[test]
    fn test_detect_architecture_case_insensitive() {
        let config = json!({
            "architectures": ["LLAMAFORCAUSALLM"],
            "hidden_size": 4096
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "llama");
    }

    #[test]
    fn test_detect_architecture_missing_fields() {
        let config = json!({
            "hidden_size": 4096,
            "vocab_size": 32000
        });

        let result = detect_architecture(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Could not detect"));
    }

    #[test]
    fn test_detect_architecture_empty_architectures_array() {
        let config = json!({
            "architectures": [],
            "hidden_size": 4096
        });

        let result = detect_architecture(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_type_takes_precedence() {
        let config = json!({
            "model_type": "mistral",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096
        });

        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "mistral");
    }
}
