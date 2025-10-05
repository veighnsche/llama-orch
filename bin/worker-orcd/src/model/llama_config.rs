//! Llama Model Configuration
//!
//! Rust representation of Llama model configuration extracted from GGUF metadata.
//!
//! Spec: M0-W-1211, M0-W-1212

/// Llama model configuration
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    /// Model architecture (always "llama")
    pub architecture: String,

    /// Context window size (e.g., 32768 for Qwen, 4096 for Phi-3)
    pub context_length: u32,

    /// Embedding dimensions / hidden size (e.g., 896 for Qwen, 3072 for Phi-3)
    pub embedding_length: u32,

    /// Number of transformer layers (e.g., 24 for Qwen, 32 for Phi-3)
    pub block_count: u32,

    /// Number of attention heads (e.g., 14 for Qwen, 32 for Phi-3)
    pub attention_head_count: u32,

    /// Number of KV heads for GQA (e.g., 2 for Qwen GQA, 32 for Phi-3 MHA)
    pub attention_head_count_kv: u32,

    /// FFN intermediate size (e.g., 4864 for Qwen, 8192 for Phi-3)
    pub ffn_length: u32,

    /// RoPE dimension count (e.g., 64)
    pub rope_dimension_count: u32,

    /// RoPE frequency base (e.g., 10000.0 standard, 1000000.0 for Qwen)
    pub rope_freq_base: f32,

    /// Vocabulary size (e.g., 151936 for Qwen, 32064 for Phi-3)
    pub vocab_size: u32,

    /// Derived: dimension per attention head (embedding_length / attention_head_count)
    pub head_dim: u32,

    /// Derived: dimension per KV head (embedding_length / attention_head_count_kv)
    pub kv_head_dim: u32,
}

impl LlamaConfig {
    /// Check if model uses Grouped Query Attention (GQA)
    pub fn is_gqa(&self) -> bool {
        self.attention_head_count_kv < self.attention_head_count
    }

    /// Check if model uses Multi-Head Attention (MHA)
    pub fn is_mha(&self) -> bool {
        self.attention_head_count_kv == self.attention_head_count
    }

    /// Get GQA group size (attention_heads / kv_heads)
    pub fn gqa_group_size(&self) -> u32 {
        if self.attention_head_count_kv == 0 {
            return 0;
        }
        self.attention_head_count / self.attention_head_count_kv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_qwen_config() -> LlamaConfig {
        LlamaConfig {
            architecture: "llama".to_string(),
            context_length: 32768,
            embedding_length: 896,
            block_count: 24,
            attention_head_count: 14,
            attention_head_count_kv: 2,
            ffn_length: 4864,
            rope_dimension_count: 64,
            rope_freq_base: 1000000.0,
            vocab_size: 151936,
            head_dim: 64,
            kv_head_dim: 448,
        }
    }

    fn create_phi3_config() -> LlamaConfig {
        LlamaConfig {
            architecture: "llama".to_string(),
            context_length: 4096,
            embedding_length: 3072,
            block_count: 32,
            attention_head_count: 32,
            attention_head_count_kv: 32,
            ffn_length: 8192,
            rope_dimension_count: 96,
            rope_freq_base: 10000.0,
            vocab_size: 32064,
            head_dim: 96,
            kv_head_dim: 96,
        }
    }

    #[test]
    fn test_qwen_is_gqa() {
        let config = create_qwen_config();
        assert!(config.is_gqa());
        assert!(!config.is_mha());
        assert_eq!(config.gqa_group_size(), 7); // 14 / 2
    }

    #[test]
    fn test_phi3_is_mha() {
        let config = create_phi3_config();
        assert!(!config.is_gqa());
        assert!(config.is_mha());
        assert_eq!(config.gqa_group_size(), 1); // 32 / 32
    }

    #[test]
    fn test_derived_parameters() {
        let config = create_qwen_config();
        assert_eq!(config.head_dim, 64); // 896 / 14
        assert_eq!(config.kv_head_dim, 448); // 896 / 2
    }
}

// ---
// Implemented by Llama-Beta 🦙
