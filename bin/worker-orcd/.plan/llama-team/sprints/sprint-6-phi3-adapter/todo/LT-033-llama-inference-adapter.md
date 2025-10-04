# LT-033: LlamaInferenceAdapter Implementation

**Team**: Llama-Beta  
**Sprint**: Sprint 6 - Phi-3 + Adapter  
**Size**: M (3 days)  
**Days**: 75-77  
**Spec Ref**: M0-W-1213, M0-W-1214

---

## Story Description

Implement LlamaInferenceAdapter to provide unified interface for all Llama-family models (Qwen, Phi-3). Abstract model-specific details behind common adapter interface, enabling polymorphic model loading and inference.

---

## Acceptance Criteria

- [ ] Implement InferenceAdapter trait for Llama models
- [ ] Support Qwen2.5-0.5B model via adapter
- [ ] Support Phi-3-mini-4k model via adapter
- [ ] Abstract model loading (GGUF → VRAM)
- [ ] Abstract tokenization (encode/decode)
- [ ] Abstract inference (prefill/decode)
- [ ] Support model selection by model_ref string
- [ ] Unit tests validate adapter interface
- [ ] Integration tests validate both models via adapter
- [ ] Error handling for unsupported models
- [ ] Log adapter operations at INFO level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-027: Gate 2 Checkpoint (needs Qwen working)
- LT-031: Phi-3 Forward Pass (needs Phi-3 working)
- LT-032: Tokenizer Conformance Tests Phi-3 (needs validated tokenizer)
- FT-026: InferenceAdapter Trait (needs adapter interface)

### Downstream (This Story Blocks)
- LT-034: Gate 3 Participation (needs adapter validation)
- FT-028: Adapter Registry (needs concrete adapter)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/adapters/llama_adapter.rs` - Llama adapter implementation
- `bin/worker-orcd/src/adapters/mod.rs` - Adapter module exports
- `bin/worker-orcd/cuda/src/adapters/llama_adapter.cpp` - C++ adapter wrapper

### InferenceAdapter Trait (from Foundation)
```rust
pub trait InferenceAdapter: Send + Sync {
    // Model lifecycle
    fn load(&mut self, model_path: &Path) -> Result<(), AdapterError>;
    fn unload(&mut self) -> Result<(), AdapterError>;
    
    // Tokenization
    fn encode(&self, text: &str) -> Result<Vec<u32>, AdapterError>;
    fn decode(&self, token_ids: &[u32]) -> Result<String, AdapterError>;
    
    // Inference
    fn prefill(&mut self, input_ids: &[u32]) -> Result<Vec<u32>, AdapterError>;
    fn decode_token(&mut self, input_id: u32) -> Result<u32, AdapterError>;
    
    // Metadata
    fn model_type(&self) -> &str;
    fn context_length(&self) -> usize;
    fn vocab_size(&self) -> usize;
}
```

### LlamaInferenceAdapter Implementation
```rust
pub struct LlamaInferenceAdapter {
    model_variant: LlamaVariant,
    model: Option<LlamaModel>,
    encoder: Option<BPEEncoder>,
    decoder: Option<BPEDecoder>,
    streaming_decoder: Option<StreamingDecoder>,
    kv_cache: Option<KVCache>,
    config: Option<LlamaConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlamaVariant {
    Qwen,
    Phi3,
}

pub enum LlamaModel {
    Qwen(QwenModel),
    Phi3(Phi3Model),
}

impl LlamaInferenceAdapter {
    pub fn new(variant: LlamaVariant) -> Self {
        Self {
            model_variant: variant,
            model: None,
            encoder: None,
            decoder: None,
            streaming_decoder: None,
            kv_cache: None,
            config: None,
        }
    }
    
    pub fn from_model_ref(model_ref: &str) -> Result<Self, AdapterError> {
        let variant = match model_ref {
            ref s if s.contains("qwen") => LlamaVariant::Qwen,
            ref s if s.contains("phi-3") || s.contains("phi3") => LlamaVariant::Phi3,
            _ => return Err(AdapterError::UnsupportedModel(model_ref.to_string())),
        };
        
        Ok(Self::new(variant))
    }
}

impl InferenceAdapter for LlamaInferenceAdapter {
    fn load(&mut self, model_path: &Path) -> Result<(), AdapterError> {
        tracing::info!("Loading Llama model: variant={:?}, path={}", self.model_variant, model_path.display());
        
        // 1. Load model based on variant
        match self.model_variant {
            LlamaVariant::Qwen => {
                let qwen_model = QwenLoader::load(model_path)?;
                self.config = Some(qwen_model.config.clone());
                self.model = Some(LlamaModel::Qwen(qwen_model));
            }
            LlamaVariant::Phi3 => {
                let phi3_model = Phi3Loader::load(model_path)?;
                self.config = Some(phi3_model.config.clone());
                self.model = Some(LlamaModel::Phi3(phi3_model));
            }
        }
        
        // 2. Load tokenizer
        self.encoder = Some(BPEEncoder::from_gguf(model_path)?);
        self.decoder = Some(BPEDecoder::from_gguf(model_path)?);
        self.streaming_decoder = Some(StreamingDecoder::new(self.decoder.as_ref().unwrap().clone()));
        
        // 3. Allocate KV cache
        let config = self.config.as_ref().unwrap();
        self.kv_cache = Some(KVCache::new(
            1,
            config.context_length as usize,
            config.attention_head_count_kv as usize,
            config.head_dim as usize,
        )?);
        
        tracing::info!("Llama model loaded successfully");
        Ok(())
    }
    
    fn unload(&mut self) -> Result<(), AdapterError> {
        tracing::info!("Unloading Llama model");
        
        self.model = None;
        self.encoder = None;
        self.decoder = None;
        self.streaming_decoder = None;
        self.kv_cache = None;
        self.config = None;
        
        Ok(())
    }
    
    fn encode(&self, text: &str) -> Result<Vec<u32>, AdapterError> {
        let encoder = self.encoder.as_ref()
            .ok_or(AdapterError::ModelNotLoaded)?;
        
        Ok(encoder.encode(text))
    }
    
    fn decode(&self, token_ids: &[u32]) -> Result<String, AdapterError> {
        let decoder = self.decoder.as_ref()
            .ok_or(AdapterError::ModelNotLoaded)?;
        
        Ok(decoder.decode(token_ids)?)
    }
    
    fn prefill(&mut self, input_ids: &[u32]) -> Result<Vec<u32>, AdapterError> {
        let model = self.model.as_ref()
            .ok_or(AdapterError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut()
            .ok_or(AdapterError::ModelNotLoaded)?;
        
        let config = ForwardPassConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: input_ids.len() as i32,
            cache_len: 0,
            temperature: 0.7,
            seed: 42,
        };
        
        match model {
            LlamaModel::Qwen(qwen) => {
                Ok(QwenForward::prefill(qwen, input_ids, kv_cache, &config)?)
            }
            LlamaModel::Phi3(phi3) => {
                Ok(Phi3Forward::prefill(phi3, input_ids, kv_cache, &config)?)
            }
        }
    }
    
    fn decode_token(&mut self, input_id: u32) -> Result<u32, AdapterError> {
        let model = self.model.as_ref()
            .ok_or(AdapterError::ModelNotLoaded)?;
        let kv_cache = self.kv_cache.as_mut()
            .ok_or(AdapterError::ModelNotLoaded)?;
        
        let cache_len = kv_cache.current_length();
        
        let config = ForwardPassConfig {
            is_prefill: false,
            batch_size: 1,
            seq_len: 1,
            cache_len: cache_len as i32,
            temperature: 0.7,
            seed: 42,
        };
        
        match model {
            LlamaModel::Qwen(qwen) => {
                Ok(QwenForward::decode(qwen, input_id, kv_cache, &config)?)
            }
            LlamaModel::Phi3(phi3) => {
                Ok(Phi3Forward::decode(phi3, input_id, kv_cache, &config)?)
            }
        }
    }
    
    fn model_type(&self) -> &str {
        match self.model_variant {
            LlamaVariant::Qwen => "llama/qwen",
            LlamaVariant::Phi3 => "llama/phi3",
        }
    }
    
    fn context_length(&self) -> usize {
        self.config.as_ref()
            .map(|c| c.context_length as usize)
            .unwrap_or(0)
    }
    
    fn vocab_size(&self) -> usize {
        self.config.as_ref()
            .map(|c| c.vocab_size as usize)
            .unwrap_or(0)
    }
}
```

### Usage Example
```rust
// Create adapter from model_ref
let mut adapter = LlamaInferenceAdapter::from_model_ref("qwen2.5-0.5b-instruct")?;

// Load model
adapter.load(Path::new("qwen2.5-0.5b.gguf"))?;

// Encode prompt
let input_ids = adapter.encode("Write a haiku about autumn leaves")?;

// Prefill
let prefill_output = adapter.prefill(&input_ids)?;

// Decode tokens
let mut generated_ids = vec![];
let mut current_token = *prefill_output.last().unwrap();

for _ in 0..30 {
    current_token = adapter.decode_token(current_token)?;
    generated_ids.push(current_token);
    
    if current_token == adapter.encoder.as_ref().unwrap().eos_token_id() {
        break;
    }
}

// Decode output
let output = adapter.decode(&generated_ids)?;
println!("Generated: {}", output);

// Unload
adapter.unload()?;
```

---

## Testing Strategy

### Unit Tests
- Test adapter creation from model_ref
- Test Qwen variant selection
- Test Phi-3 variant selection
- Test model loading/unloading
- Test encode/decode
- Test prefill/decode

### Integration Tests
- Test full generation with Qwen via adapter
- Test full generation with Phi-3 via adapter
- Test switching between models
- Test adapter interface compliance

### Polymorphism Tests
- Test using adapter as trait object
- Test adapter registry integration
- Test dynamic model selection

### Manual Verification
1. Create Qwen adapter
2. Load model and generate text
3. Unload and create Phi-3 adapter
4. Load model and generate text
5. Verify both work identically via interface

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing
- [ ] Both Qwen and Phi-3 work via adapter
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.10 (Adapters)
- Related Stories: LT-027, LT-031, LT-032, FT-026

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
