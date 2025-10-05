# Tokenizer Implementation Plan - Final Step to Haiku Test

**Date**: 2025-10-05  
**Status**: 📋 READY TO IMPLEMENT  
**ETA**: 1-2 hours

---

## Current Status

### ✅ What's Done
- ✅ GGUF parser extracts tokenizer metadata
- ✅ Tokenizer crate exists (`worker-tokenizer`)
- ✅ BPE encoder/decoder implemented
- ✅ Vocab and merges parsing ready
- ✅ Streaming decoder for UTF-8 safety

### ❌ What's Missing
- ❌ Wire tokenizer to GGUF metadata
- ❌ Implement `Tokenizer::from_gguf()` method
- ❌ Connect tokenizer to inference pipeline
- ❌ Add tokenizer to haiku test

---

## Problem Analysis

### Current Situation

The tokenizer crate has this stub:

```rust
// worker-tokenizer/src/backend.rs:92
pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
    // Note: GGUF BPE loading requires vocab and merges from GGUF file
    // This is handled separately in model loading
    Err(TokenizerError::EncodeFailed(
        "GGUF BPE tokenizer must be loaded from GGUF file".to_string()
    ))
}
```

**This needs to be implemented!**

### What GGUF Contains

From `RESEARCH_RESULTS.md`, GGUF has:

| Metadata Key | Format | Example |
|--------------|--------|---------|
| `tokenizer.ggml.model` | String | "gpt2" (BPE variant) |
| `tokenizer.ggml.tokens` | Array[151,643] | ["!", "\"", "#", ...] |
| `tokenizer.ggml.merges` | Array | ["Ġ t", "Ġ a", ...] |
| `tokenizer.ggml.bos_token_id` | Integer | 151,643 |
| `tokenizer.ggml.eos_token_id` | Integer | 151,643 |

### What's Available

The `worker-gguf` crate already parses this:

```rust
// worker-gguf/src/lib.rs
pub fn vocab_size(&self) -> Result<usize, GGUFError> {
    match self.metadata.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array { count, .. }) => Ok(*count as usize),
        _ => Err(GGUFError::MissingKey("tokenizer.ggml.tokens".to_string())),
    }
}
```

**But it doesn't expose the actual tokens and merges arrays!**

---

## Implementation Plan

### Phase 1: Extend GGUF Parser (30 min)

**File**: `worker-gguf/src/lib.rs`

**Add methods to extract tokenizer data**:

```rust
impl GGUFMetadata {
    /// Get tokenizer tokens array
    pub fn tokenizer_tokens(&self) -> Result<Vec<String>, GGUFError> {
        match self.metadata.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array { items, .. }) => {
                items.iter()
                    .map(|v| match v {
                        MetadataValue::String(s) => Ok(s.clone()),
                        _ => Err(GGUFError::InvalidType("Expected string".to_string())),
                    })
                    .collect()
            }
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.tokens".to_string())),
        }
    }
    
    /// Get tokenizer merges array
    pub fn tokenizer_merges(&self) -> Result<Vec<String>, GGUFError> {
        match self.metadata.get("tokenizer.ggml.merges") {
            Some(MetadataValue::Array { items, .. }) => {
                items.iter()
                    .map(|v| match v {
                        MetadataValue::String(s) => Ok(s.clone()),
                        _ => Err(GGUFError::InvalidType("Expected string".to_string())),
                    })
                    .collect()
            }
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.merges".to_string())),
        }
    }
    
    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Result<u32, GGUFError> {
        match self.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(MetadataValue::UInt32(id)) => Ok(*id),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.bos_token_id".to_string())),
        }
    }
    
    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Result<u32, GGUFError> {
        match self.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(MetadataValue::UInt32(id)) => Ok(*id),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.eos_token_id".to_string())),
        }
    }
}
```

**Test**:
```rust
#[test]
fn test_extract_tokenizer_data() {
    let metadata = GGUFMetadata::from_file("qwen2.5-0.5b.gguf").unwrap();
    
    let tokens = metadata.tokenizer_tokens().unwrap();
    assert_eq!(tokens.len(), 151936);
    
    let merges = metadata.tokenizer_merges().unwrap();
    assert!(merges.len() > 0);
    
    let bos = metadata.bos_token_id().unwrap();
    assert_eq!(bos, 151643);
}
```

---

### Phase 2: Implement Tokenizer::from_gguf() (30 min)

**File**: `worker-tokenizer/src/backend.rs`

**Replace stub with real implementation**:

```rust
impl Tokenizer {
    /// Load tokenizer from GGUF file
    ///
    /// Extracts vocabulary and merges from GGUF metadata.
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        use worker_gguf::GGUFMetadata;
        
        // Parse GGUF file
        let metadata = GGUFMetadata::from_file(path.as_ref().to_str().unwrap())
            .map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;
        
        // Extract tokens
        let tokens = metadata.tokenizer_tokens()
            .map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;
        
        // Extract merges
        let merge_strings = metadata.tokenizer_merges()
            .map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;
        
        // Build vocabulary
        let vocab = Vocabulary::from_tokens(tokens)?;
        
        // Build merge table
        let merges = MergeTable::from_strings(&merge_strings)?;
        
        // Create encoder and decoder
        let encoder = BPEEncoder::new(vocab.clone(), merges.clone());
        let decoder = BPEDecoder::new(vocab);
        
        Ok(Tokenizer::GgufBpe { encoder, decoder })
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        match self {
            Tokenizer::GgufBpe { encoder, .. } => {
                let mut tokens = encoder.encode(text)?;
                
                if add_bos {
                    tokens.insert(0, 151643);  // Qwen BOS token
                }
                
                Ok(tokens)
            }
            Tokenizer::HfJson(hf) => hf.encode(text, add_bos),
        }
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        match self {
            Tokenizer::GgufBpe { decoder, .. } => decoder.decode(tokens),
            Tokenizer::HfJson(hf) => hf.decode(tokens),
        }
    }
}
```

**Test**:
```rust
#[test]
fn test_tokenizer_from_gguf() {
    let tokenizer = Tokenizer::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    
    // Test encoding
    let tokens = tokenizer.encode("Write a haiku about", true).unwrap();
    assert!(tokens.len() > 0);
    assert_eq!(tokens[0], 151643);  // BOS token
    
    // Test decoding
    let text = tokenizer.decode(&tokens[1..]).unwrap();
    assert!(text.contains("haiku"));
}
```

---

### Phase 3: Add Dependencies (5 min)

**File**: `worker-tokenizer/Cargo.toml`

```toml
[dependencies]
worker-gguf = { path = "../worker-gguf" }
# ... existing dependencies
```

**File**: `worker-orcd/Cargo.toml`

```toml
[dependencies]
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
# ... existing dependencies
```

---

### Phase 4: Wire to Inference Pipeline (15 min)

**File**: `worker-orcd/src/inference/cuda_backend.rs` (or create if needed)

```rust
use worker_tokenizer::Tokenizer;
use worker_gguf::GGUFMetadata;

pub struct QwenInference {
    tokenizer: Tokenizer,
    inference_ctx: *mut InferenceContext,
    // ... other fields
}

impl QwenInference {
    pub fn new(model_path: &str) -> Result<Self> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_gguf(model_path)?;
        
        // Load model and init inference (existing code)
        let metadata = GGUFMetadata::from_file(model_path)?;
        let model = cuda_load_model(...)?;
        let inference_ctx = cuda_inference_init(...)?;
        
        Ok(Self {
            tokenizer,
            inference_ctx,
        })
    }
    
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Encode prompt
        let token_ids = self.tokenizer.encode(prompt, true)?;
        
        // Generate tokens
        let mut generated = Vec::new();
        let mut current_token = token_ids.last().copied().unwrap();
        
        for _ in 0..max_tokens {
            let mut error = 0;
            let next_token = unsafe {
                cuda_inference_generate_token(
                    self.inference_ctx,
                    current_token,
                    0.7,  // temperature
                    0,    // top_k
                    0.0,  // top_p
                    42,   // seed
                    &mut error,
                )
            };
            
            if error != 0 {
                return Err(Error::InferenceFailed);
            }
            
            generated.push(next_token);
            current_token = next_token;
            
            // Check for EOS
            if next_token == 151643 {
                break;
            }
        }
        
        // Decode tokens
        let text = self.tokenizer.decode(&generated)?;
        Ok(text)
    }
}
```

---

### Phase 5: Update Haiku Test (10 min)

**File**: `worker-orcd/tests/haiku_generation_anti_cheat.rs`

```rust
#[tokio::test]
async fn test_haiku_generation_real() {
    let model_path = "/path/to/qwen2.5-0.5b.gguf";
    
    // Create inference
    let mut inference = QwenInference::new(model_path).unwrap();
    
    // Generate haiku
    let haiku = inference.generate("Write a haiku about mountains:", 30).unwrap();
    
    println!("Generated haiku:\n{}", haiku);
    
    // Verify it's not empty
    assert!(!haiku.is_empty());
    
    // Verify it contains text (not just tokens)
    assert!(haiku.chars().any(|c| c.is_alphabetic()));
}
```

---

## Implementation Checklist

### Step 1: GGUF Parser Extensions ✅
- [ ] Add `tokenizer_tokens()` method
- [ ] Add `tokenizer_merges()` method
- [ ] Add `bos_token_id()` method
- [ ] Add `eos_token_id()` method
- [ ] Add tests for extraction
- [ ] Verify with real GGUF file

### Step 2: Tokenizer Implementation ✅
- [ ] Implement `Tokenizer::from_gguf()`
- [ ] Implement `encode()` method
- [ ] Implement `decode()` method
- [ ] Add BOS token handling
- [ ] Add EOS token detection
- [ ] Add tests

### Step 3: Dependencies ✅
- [ ] Add `worker-gguf` to `worker-tokenizer`
- [ ] Add `worker-tokenizer` to `worker-orcd`
- [ ] Verify builds

### Step 4: Integration ✅
- [ ] Create `QwenInference` struct
- [ ] Wire tokenizer to inference
- [ ] Implement `generate()` method
- [ ] Add error handling
- [ ] Test end-to-end

### Step 5: Testing ✅
- [ ] Update haiku test
- [ ] Run with real model
- [ ] Verify output quality
- [ ] Check UTF-8 safety

---

## Testing Strategy

### Unit Tests

```rust
// Test GGUF extraction
#[test]
fn test_gguf_tokenizer_extraction() {
    let metadata = GGUFMetadata::from_file("qwen.gguf").unwrap();
    let tokens = metadata.tokenizer_tokens().unwrap();
    let merges = metadata.tokenizer_merges().unwrap();
    assert_eq!(tokens.len(), 151936);
}

// Test tokenizer loading
#[test]
fn test_tokenizer_from_gguf() {
    let tokenizer = Tokenizer::from_gguf("qwen.gguf").unwrap();
    let tokens = tokenizer.encode("Hello", true).unwrap();
    assert_eq!(tokens[0], 151643);  // BOS
}

// Test encode/decode roundtrip
#[test]
fn test_encode_decode_roundtrip() {
    let tokenizer = Tokenizer::from_gguf("qwen.gguf").unwrap();
    let text = "Write a haiku about mountains";
    let tokens = tokenizer.encode(text, false).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(text, decoded);
}
```

### Integration Test

```rust
#[test]
fn test_full_inference_pipeline() {
    let mut inference = QwenInference::new("qwen.gguf").unwrap();
    let output = inference.generate("Write a haiku", 20).unwrap();
    
    assert!(!output.is_empty());
    assert!(output.len() > 10);
    println!("Generated: {}", output);
}
```

---

## Potential Issues & Solutions

### Issue 1: GGUF Array Parsing

**Problem**: `MetadataValue::Array` might not have `items` field

**Solution**: Check actual GGUF parser implementation and adjust

```rust
// May need to parse array differently
match self.metadata.get("tokenizer.ggml.tokens") {
    Some(MetadataValue::StringArray(items)) => Ok(items.clone()),
    // ... handle other cases
}
```

### Issue 2: Byte-level BPE Encoding

**Problem**: Qwen uses byte-level BPE, need proper byte handling

**Solution**: Use existing `BPEEncoder` which already handles this

```rust
// BPEEncoder already does:
// 1. Convert text to UTF-8 bytes
// 2. Map bytes to tokens
// 3. Apply merges
// 4. Return token IDs
```

### Issue 3: Special Token Handling

**Problem**: BOS/EOS tokens need special handling

**Solution**: Add special token logic in encode/decode

```rust
pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
    let mut tokens = self.encoder.encode(text)?;
    if add_bos {
        tokens.insert(0, self.bos_token_id);
    }
    Ok(tokens)
}
```

### Issue 4: UTF-8 Streaming

**Problem**: Token boundaries might not align with UTF-8

**Solution**: Use existing `StreamingDecoder`

```rust
// StreamingDecoder already handles:
// - Partial UTF-8 sequences
// - Buffering incomplete bytes
// - Safe streaming output
```

---

## Time Estimate

| Phase | Task | Time |
|-------|------|------|
| 1 | GGUF parser extensions | 30 min |
| 2 | Tokenizer implementation | 30 min |
| 3 | Dependencies | 5 min |
| 4 | Integration | 15 min |
| 5 | Testing | 10 min |
| **Total** | | **1.5 hours** |

**Buffer for debugging**: +30 min  
**Total with buffer**: **2 hours**

---

## Success Criteria

### Must Have ✅
- [ ] `Tokenizer::from_gguf()` works
- [ ] Can encode text to tokens
- [ ] Can decode tokens to text
- [ ] Haiku test generates actual text
- [ ] UTF-8 is valid

### Nice to Have 🎯
- [ ] Streaming decoder works
- [ ] Special tokens handled correctly
- [ ] Performance is acceptable
- [ ] Error messages are helpful

---

## Next Steps

1. **Start with Phase 1**: Extend GGUF parser
2. **Test immediately**: Verify token extraction works
3. **Move to Phase 2**: Implement tokenizer
4. **Test again**: Verify encode/decode works
5. **Phase 3-4**: Wire everything together
6. **Phase 5**: Run haiku test and celebrate! 🎉

---

## Final Note

**This is the LAST piece!** Once tokenizer is wired:

- ✅ GGUF parsing works
- ✅ Weight loading works
- ✅ Transformer works
- ✅ Sampling works
- ✅ FFI works
- ✅ **Tokenizer works** ← We're here
- ✅ **HAIKU TEST PASSES** ← Next!

**ETA to haiku test passing**: 2 hours! 🚀

---
Crafted by GPT-Gamma 🤖
