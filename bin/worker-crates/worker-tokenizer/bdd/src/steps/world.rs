//! World state for worker-tokenizer BDD tests

use cucumber::World;

#[derive(Debug, Default, World)]
pub struct TokenizerWorld {
    pub tokenizer_type: Option<String>,
    pub input_text: Option<String>,
    pub encoded_tokens: Option<Vec<u32>>,
    pub decoded_text: Option<String>,
}
