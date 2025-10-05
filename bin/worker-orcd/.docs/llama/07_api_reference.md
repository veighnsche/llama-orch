# API Reference

**Component**: Public API  
**Spec**: M0-W-1214

---

## Module Structure

```
worker_orcd
├── models
│   ├── LlamaInferenceAdapter
│   ├── AdapterForwardConfig
│   ├── ModelType
│   ├── AdapterError
│   ├── qwen
│   │   ├── QwenConfig
│   │   ├── QwenWeightLoader
│   │   ├── QwenModel
│   │   └── QwenForward
│   └── phi3
│       ├── Phi3Config
│       ├── Phi3WeightLoader
│       ├── Phi3Model
│       └── Phi3Forward
└── tokenizer
    ├── BPEEncoder
    ├── BPEDecoder
    ├── StreamingDecoder
    ├── Vocabulary
    └── MergeTable
```

---

## models::LlamaInferenceAdapter

Unified adapter for all Llama-family models.

### Constructor Methods

```rust
pub fn new_qwen(model: QwenModel) -> Self
```
Create adapter for Qwen model.

```rust
pub fn new_phi3(model: Phi3Model) -> Self
```
Create adapter for Phi-3 model.

### Information Methods

```rust
pub fn model_type(&self) -> ModelType
```
Returns the model type (Qwen2_5, Phi3, etc.).

```rust
pub fn vocab_size(&self) -> Result<usize, AdapterError>
```
Returns vocabulary size.

```rust
pub fn hidden_dim(&self) -> Result<usize, AdapterError>
```
Returns hidden dimension.

```rust
pub fn num_layers(&self) -> Result<usize, AdapterError>
```
Returns number of transformer layers.

```rust
pub fn vram_usage(&self) -> Result<usize, AdapterError>
```
Returns total VRAM usage in bytes.

### Inference Methods

```rust
pub fn prefill(
    &self,
    input_ids: &[u32],
    config: &AdapterForwardConfig,
) -> Result<Vec<u32>, AdapterError>
```
Process full prompt (prefill phase).

**Parameters**:
- `input_ids`: Input token IDs
- `config`: Forward pass configuration

**Returns**: Output token IDs

```rust
pub fn decode(
    &self,
    input_id: u32,
    config: &AdapterForwardConfig,
) -> Result<u32, AdapterError>
```
Generate single token (decode phase).

**Parameters**:
- `input_id`: Previous token ID
- `config`: Forward pass configuration

**Returns**: Next token ID

```rust
pub fn generate(
    &self,
    input_ids: &[u32],
    max_tokens: usize,
    config: &AdapterForwardConfig,
) -> Result<Vec<u32>, AdapterError>
```
Generate tokens autoregressively.

**Parameters**:
- `input_ids`: Input token IDs
- `max_tokens`: Maximum tokens to generate
- `config`: Forward pass configuration

**Returns**: Complete output (input + generated)

---

## models::AdapterForwardConfig

Configuration for forward pass.

```rust
pub struct AdapterForwardConfig {
    pub is_prefill: bool,
    pub batch_size: usize,
    pub seq_len: usize,
    pub cache_len: usize,
    pub temperature: f32,
    pub seed: u32,
}
```

**Fields**:
- `is_prefill`: True for prefill, false for decode
- `batch_size`: Batch size (currently must be 1)
- `seq_len`: Sequence length
- `cache_len`: KV cache length (for decode)
- `temperature`: Sampling temperature (0.1-2.0)
- `seed`: Random seed for reproducibility

---

## models::qwen::QwenConfig

Qwen model configuration.

```rust
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
}
```

### Methods

```rust
pub fn qwen2_5_0_5b() -> Self
```
Returns Qwen2.5-0.5B configuration.

---

## models::qwen::QwenWeightLoader

GGUF weight loader for Qwen.

### Methods

```rust
pub fn load_to_vram(
    gguf_path: &str,
    config: &QwenConfig,
) -> Result<QwenModel, WeightLoadingError>
```
Load Qwen model from GGUF to VRAM.

```rust
pub fn calculate_vram_usage(config: &QwenConfig) -> usize
```
Calculate total VRAM usage in bytes.

---

## models::phi3::Phi3Config

Phi-3 model configuration.

```rust
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
}
```

### Methods

```rust
pub fn phi3_mini_4k() -> Self
```
Returns Phi-3-mini-4k configuration.

---

## models::phi3::Phi3WeightLoader

GGUF weight loader for Phi-3.

### Methods

```rust
pub fn load_to_vram(
    gguf_path: &str,
    config: &Phi3Config,
) -> Result<Phi3Model, Phi3Error>
```
Load Phi-3 model from GGUF to VRAM.

```rust
pub fn calculate_vram_usage(config: &Phi3Config) -> usize
```
Calculate total VRAM usage in bytes.

---

## tokenizer::BPEEncoder

Byte-level BPE encoder.

### Constructor

```rust
pub fn new(vocab: Vocabulary, merges: MergeTable) -> Self
```
Create encoder from vocabulary and merges.

```rust
pub fn from_gguf(gguf_path: &str) -> Result<Self, TokenizerError>
```
Load encoder from GGUF metadata.

### Methods

```rust
pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>
```
Encode text to token IDs.

**Parameters**:
- `text`: Input text

**Returns**: Token IDs

---

## tokenizer::BPEDecoder

Byte-level BPE decoder.

### Constructor

```rust
pub fn new(vocab: Vocabulary) -> Self
```
Create decoder from vocabulary.

```rust
pub fn from_gguf(gguf_path: &str) -> Result<Self, TokenizerError>
```
Load decoder from GGUF metadata.

### Methods

```rust
pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>
```
Decode token IDs to text.

**Parameters**:
- `token_ids`: Input token IDs

**Returns**: Decoded text

---

## tokenizer::StreamingDecoder

UTF-8 safe streaming decoder.

### Constructor

```rust
pub fn new(decoder: BPEDecoder) -> Self
```
Create streaming decoder.

### Methods

```rust
pub fn decode_token(&mut self, token_id: u32) -> String
```
Decode single token (may return empty if UTF-8 incomplete).

**Parameters**:
- `token_id`: Token ID

**Returns**: Partial text (empty if buffered)

```rust
pub fn flush(&mut self) -> String
```
Flush remaining buffered bytes.

**Returns**: Remaining text

---

## tokenizer::Vocabulary

Token vocabulary.

### Constructor

```rust
pub fn new(
    tokens: Vec<String>,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: Option<u32>,
) -> Result<Self, TokenizerError>
```
Create vocabulary from token list.

### Methods

```rust
pub fn token_to_id(&self, token: &str) -> Result<u32, TokenizerError>
```
Convert token to ID.

```rust
pub fn id_to_token(&self, id: u32) -> Result<String, TokenizerError>
```
Convert ID to token.

```rust
pub fn bos_token_id(&self) -> u32
```
Returns BOS (beginning of sequence) token ID.

```rust
pub fn eos_token_id(&self) -> u32
```
Returns EOS (end of sequence) token ID.

---

## tokenizer::MergeTable

BPE merge table.

### Constructor

```rust
pub fn new(merge_list: Vec<String>) -> Result<Self, TokenizerError>
```
Create merge table from merge list.

### Methods

```rust
pub fn get_rank(&self, pair: &str) -> Option<usize>
```
Get merge rank for token pair.

---

## Error Types

### AdapterError

```rust
pub enum AdapterError {
    ModelNotLoaded,
    InvalidModelType(ModelType),
    ForwardPassFailed(String),
    UnsupportedOperation(ModelType),
}
```

### TokenizerError

```rust
pub enum TokenizerError {
    TokenNotFound(String),
    IdNotFound(u32),
    InvalidUtf8(std::string::FromUtf8Error),
    InvalidHexString,
    GGUFError(GGUFError),
}
```

### WeightLoadingError

```rust
pub enum WeightLoadingError {
    AllocationFailed(usize),
    TransferFailed(String),
    MappingError(WeightMappingError),
}
```

---

## Type Aliases

```rust
pub type Result<T> = std::result::Result<T, AdapterError>;
```

---

## Constants

```rust
// Qwen2.5-0.5B
pub const QWEN_VOCAB_SIZE: usize = 151936;
pub const QWEN_HIDDEN_DIM: usize = 896;
pub const QWEN_NUM_LAYERS: usize = 24;

// Phi-3-mini-4k
pub const PHI3_VOCAB_SIZE: usize = 32064;
pub const PHI3_HIDDEN_DIM: usize = 3072;
pub const PHI3_NUM_LAYERS: usize = 32;
```

---

## Complete Example

```rust
use worker_orcd::models::{
    LlamaInferenceAdapter, AdapterForwardConfig,
    qwen::{QwenConfig, QwenWeightLoader},
};
use worker_orcd::tokenizer::{BPEEncoder, BPEDecoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let config = QwenConfig::qwen2_5_0_5b();
    let model = QwenWeightLoader::load_to_vram("qwen2.5-0.5b.gguf", &config)?;
    let adapter = LlamaInferenceAdapter::new_qwen(model);
    
    // Load tokenizer
    let encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf")?;
    let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf")?;
    
    // Encode
    let prompt = "Hello, world!";
    let input_ids = encoder.encode(prompt)?;
    
    // Generate
    let fwd_config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len(),
        cache_len: 0,
        temperature: 0.7,
        seed: 42,
    };
    
    let output_ids = adapter.generate(&input_ids, 30, &fwd_config)?;
    
    // Decode
    let output_text = decoder.decode(&output_ids)?;
    println!("{}", output_text);
    
    Ok(())
}
```

---

**Status**: Complete  
**Version**: M0  
**Last Updated**: 2025-10-05
