# TEAM-016 HANDOFF - Multi-Model Support

**Team:** TEAM-016  
**Date:** 2025-10-08T23:25:00Z  
**Status:** 🔄 READY TO START  

---

## Mission Summary

**Objective:** Remove Llama-2 specific assumptions and add support for all models that Candle supports.

**Current State:**
- ✅ Backend refactored into focused modules (TEAM-015)
- ✅ Worker-crates integrated into binary (TEAM-015)
- ✅ GPU warmup, proper tokenization, sampling (TEAM-014)
- ❌ **Hardcoded to Llama-2 only** - Need multi-model support

**Target State:**
- Support all Candle models: Llama, Mistral, Phi, Qwen, Gemma, etc.
- Model architecture auto-detection from config.json
- Flexible tokenizer loading (HuggingFace, SentencePiece, etc.)
- Unified inference interface across all models

---

## CRITICAL: Read the Rules First! 📋

**BEFORE YOU START, READ:**
- `/home/vince/Projects/llama-orch/.windsurf/rules/candled-rules.md`

**Key Rules to Remember:**
1. ✅ **ALWAYS add your team signature** - `// TEAM-016: <description>` or `// Modified by: TEAM-016`
2. ✅ **NEVER remove signatures from other teams** - Maintain full history
3. ✅ **Complete ALL priorities in order** - Don't skip Priority 2 and 3
4. ❌ **NO background testing** - Always use blocking commands
5. ❌ **NEVER create multiple .md files** - Update existing docs

---

## Current Architecture (TEAM-015)

### File Structure
```
src/
├── backend/
│   ├── mod.rs              (10 lines)   - Module exports
│   ├── model_loader.rs     (155 lines)  - SafeTensors/GGUF loading
│   ├── sampling.rs         (28 lines)   - LogitsProcessor creation
│   └── inference.rs        (287 lines)  - Main backend & generation
├── common/                 - Integrated from worker-common
│   ├── error.rs, inference_result.rs, sampling_config.rs, startup.rs
├── http/                   - Integrated from worker-http
│   ├── backend.rs, execute.rs, health.rs, routes.rs, server.rs, sse.rs, validation.rs
├── bin/
│   ├── cpu.rs, cuda.rs, accelerate.rs
├── device.rs, error.rs, lib.rs, main.rs, token_output_stream.rs
```

### Llama-2 Specific Code (To Remove)

**1. Hardcoded Model Type:**
```rust
// backend/inference.rs:24
pub struct CandleInferenceBackend {
    model: Llama,  // ❌ Hardcoded to Llama!
    tokenizer: Tokenizer,
    device: Device,
    config: Config,  // ❌ Llama-specific Config!
    model_size_bytes: u64,
}
```

**2. Hardcoded Model Loading:**
```rust
// backend/model_loader.rs:199
let model = Llama::load(vb, &config)  // ❌ Only loads Llama!
    .context("Failed to load Llama model from safetensors")?;
```

**3. Hardcoded Cache:**
```rust
// backend/inference.rs:144
let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)  // ❌ Llama Cache!
```

**4. Hardcoded Config Parsing:**
```rust
// backend/model_loader.rs:175-190
let config = Config {  // ❌ Llama Config struct!
    hidden_size: hidden_size as usize,
    intermediate_size: intermediate_size as usize,
    // ... Llama-specific fields
};
```

---

## Next Steps for TEAM-016

### PRIORITY 1: Add Model Architecture Detection (4-6 hours)

**Problem:** Currently hardcoded to Llama-2. Need to support all Candle models.

**Recommendation:** Create model abstraction layer

**Step 1.1: Create Model Trait (2 hours)**

Create `src/backend/model_trait.rs`:
```rust
//! Model abstraction trait
//!
//! Created by: TEAM-016

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Platform-agnostic model trait
///
/// Abstracts different model architectures (Llama, Mistral, Phi, etc.)
pub trait ModelBackend: Send + Sync {
    /// Forward pass through the model
    fn forward(
        &mut self,
        input_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor>;
    
    /// Get model configuration
    fn config(&self) -> &ModelConfig;
    
    /// Get EOS token ID
    fn eos_token_id(&self) -> u32;
}

/// Unified model configuration
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub architecture: String,  // "llama", "mistral", "phi", etc.
}
```

**Step 1.2: Implement Model Wrappers (2 hours)**

Create `src/backend/models/` directory:
```
src/backend/models/
├── mod.rs           - Model factory
├── llama.rs         - Llama wrapper
├── mistral.rs       - Mistral wrapper
├── phi.rs           - Phi wrapper
└── qwen.rs          - Qwen wrapper
```

Example `models/llama.rs`:
```rust
//! Llama model wrapper
//!
//! Created by: TEAM-016

use super::model_trait::{ModelBackend, ModelConfig};
use candle_transformers::models::llama::{Llama, Cache, Config};
use candle_core::{Device, DType, Tensor};
use anyhow::Result;

pub struct LlamaModel {
    model: Llama,
    cache: Cache,
    config: Config,
}

impl LlamaModel {
    pub fn load(vb: VarBuilder, config: Config, device: &Device) -> Result<Self> {
        let model = Llama::load(vb, &config)?;
        let cache = Cache::new(true, DType::F32, &config, device)?;
        Ok(Self { model, cache, config })
    }
}

impl ModelBackend for LlamaModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position, &mut self.cache)
    }
    
    fn config(&self) -> &ModelConfig {
        // Convert Llama Config to unified ModelConfig
        todo!()
    }
    
    fn eos_token_id(&self) -> u32 {
        // Extract from Llama config
        todo!()
    }
}
```

**Step 1.3: Create Model Factory (1 hour)**

Create `src/backend/models/mod.rs`:
```rust
//! Model factory - Auto-detect and load models
//!
//! Created by: TEAM-016

use anyhow::{Result, bail};
use serde_json::Value;
use std::path::Path;

pub mod llama;
pub mod mistral;
pub mod phi;
pub mod qwen;

pub use model_trait::ModelBackend;

/// Detect model architecture from config.json
pub fn detect_architecture(config_json: &Value) -> Result<String> {
    // Check "model_type" field
    if let Some(model_type) = config_json.get("model_type").and_then(|v| v.as_str()) {
        return Ok(model_type.to_string());
    }
    
    // Check "architectures" array
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            return Ok(arch.to_lowercase());
        }
    }
    
    bail!("Could not detect model architecture from config.json");
}

/// Load model based on detected architecture
pub fn load_model(
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn ModelBackend>> {
    let config_json = load_config_json(model_path)?;
    let architecture = detect_architecture(&config_json)?;
    
    match architecture.as_str() {
        "llama" | "llamaforcausallm" => {
            let model = llama::LlamaModel::load(model_path, device)?;
            Ok(Box::new(model))
        }
        "mistral" | "mistralforcausallm" => {
            let model = mistral::MistralModel::load(model_path, device)?;
            Ok(Box::new(model))
        }
        "phi" | "phiforcausallm" => {
            let model = phi::PhiModel::load(model_path, device)?;
            Ok(Box::new(model))
        }
        "qwen" | "qwenforcausallm" => {
            let model = qwen::QwenModel::load(model_path, device)?;
            Ok(Box::new(model))
        }
        _ => bail!("Unsupported model architecture: {}", architecture),
    }
}
```

**Step 1.4: Update CandleInferenceBackend (1 hour)**

Update `src/backend/inference.rs`:
```rust
// TEAM-016: Changed from Llama to generic ModelBackend
pub struct CandleInferenceBackend {
    model: Box<dyn ModelBackend>,  // ✅ Generic model!
    tokenizer: Tokenizer,
    device: Device,
    model_size_bytes: u64,
}
```

**Files to Modify:**
- `src/backend/inference.rs` - Use `Box<dyn ModelBackend>` instead of `Llama`
- `src/backend/model_loader.rs` - Use model factory instead of hardcoded Llama
- `src/backend/mod.rs` - Export models module

---

### PRIORITY 2: Add Flexible Tokenizer Loading (2-3 hours)

**Problem:** Currently assumes HuggingFace tokenizer.json. Some models use SentencePiece (.model files).

**Recommendation:** Auto-detect tokenizer format

**Step 2.1: Create Tokenizer Factory (1 hour)**

Create `src/backend/tokenizer_loader.rs`:
```rust
//! Tokenizer loading with auto-detection
//!
//! Created by: TEAM-016

use anyhow::{Result, bail};
use tokenizers::Tokenizer;
use std::path::Path;

/// Load tokenizer with auto-detection
pub fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
    let parent = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or_else(|| Path::new("."))
    };
    
    // Try tokenizer.json (HuggingFace format)
    let hf_path = parent.join("tokenizer.json");
    if hf_path.exists() {
        return Tokenizer::from_file(&hf_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {}", e));
    }
    
    // Try tokenizer.model (SentencePiece format)
    let sp_path = parent.join("tokenizer.model");
    if sp_path.exists() {
        // Load SentencePiece tokenizer
        // Note: May need to use tokenizers::models::bpe or external crate
        bail!("SentencePiece tokenizer support not yet implemented");
    }
    
    bail!("No tokenizer found at {:?}", parent);
}
```

**Step 2.2: Update Model Loading (30 min)**

Update `src/backend/inference.rs`:
```rust
// TEAM-016: Use tokenizer_loader instead of hardcoded path
let tokenizer = tokenizer_loader::load_tokenizer(&path)?;
```

**Files to Modify:**
- `src/backend/inference.rs` - Use tokenizer factory
- `src/backend/mod.rs` - Export tokenizer_loader module

---

### PRIORITY 3: Add Integration Tests for Multiple Models (2 hours)

**Problem:** No tests for non-Llama models.

**Recommendation:** Add tests for each supported architecture

**Step 3.1: Create Model Test Fixtures (1 hour)**

Create `tests/fixtures/` directory with sample configs:
```
tests/fixtures/
├── llama_config.json
├── mistral_config.json
├── phi_config.json
└── qwen_config.json
```

**Step 3.2: Add Multi-Model Tests (1 hour)**

Create `tests/multi_model_support.rs`:
```rust
//! Multi-model support tests
//!
//! Created by: TEAM-016

#[cfg(test)]
mod tests {
    use llorch_candled::backend::models::detect_architecture;
    
    #[test]
    fn test_detect_llama_architecture() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama"
        });
        
        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "llama");
    }
    
    #[test]
    fn test_detect_mistral_architecture() {
        let config = serde_json::json!({
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral"
        });
        
        let arch = detect_architecture(&config).unwrap();
        assert_eq!(arch, "mistral");
    }
    
    // Add tests for Phi, Qwen, Gemma, etc.
}
```

**Files to Create:**
- `tests/multi_model_support.rs` - Architecture detection tests
- `tests/fixtures/*.json` - Sample model configs

---

## Technical Debt Status

### ✅ Resolved by TEAM-015
1. [x] Backend refactoring - Split into focused modules
2. [x] Worker-crates integration - Removed unnecessary split

### ⏳ Remaining Technical Debt (M0)
1. **Multi-model support** - Currently Llama-2 only (PRIORITY 1 for TEAM-016)
2. **GGUF support** - Currently only SafeTensors works
3. **Error handling** - No graceful OOM, timeout handling
4. **Metrics** - No Prometheus metrics for token rate, latency
5. **SSE streaming** - Still returns complete result (deferred to M1/M2)

---

## Success Criteria for TEAM-016

### Must Complete (Priority 1)
1. [ ] Model trait abstraction created
2. [ ] Llama, Mistral, Phi, Qwen wrappers implemented
3. [ ] Model factory with auto-detection
4. [ ] CandleInferenceBackend uses generic ModelBackend
5. [ ] All existing tests pass
6. [ ] CPU and CUDA builds succeed

### Should Complete (Priority 2)
1. [ ] Tokenizer auto-detection (HuggingFace + SentencePiece)
2. [ ] Flexible tokenizer loading

### Nice to Have (Priority 3)
1. [ ] Integration tests for multiple models
2. [ ] Documentation of supported models

---

## Reference: Candle Supported Models

**Candle supports these models (as of 0.9):**
- ✅ Llama (1, 2, 3, 3.1, 3.2)
- ✅ Mistral (7B, 8x7B MoE)
- ✅ Phi (1, 1.5, 2, 3, 3.5)
- ✅ Qwen (1.5, 2, 2.5)
- ✅ Gemma (2B, 7B)
- ✅ Falcon
- ✅ StableLM
- ✅ Yi

**Location:** `reference/candle/candle-transformers/src/models/`

**Strategy:** Start with Llama, Mistral, Phi, Qwen (most popular). Add others incrementally.

---

## Code Quality Guidelines

### File Size Limits
- Keep files under 300 lines (per project standards)
- If a file grows too large, split it into focused modules

### Team Signatures
```rust
// Created by: TEAM-016
// Modified by: TEAM-016 (added multi-model support)
// TEAM-016: Changed from Llama to generic ModelBackend
```

### Testing
- Add tests for each new model wrapper
- Verify architecture detection works
- Test tokenizer auto-detection

---

## Known Issues & Gotchas

### 1. Candle Model Differences
- **Llama**: Uses `Cache` struct for KV cache
- **Mistral**: Similar to Llama but different attention
- **Phi**: Different architecture, may need different cache
- **Qwen**: Different tokenizer format

**Solution:** Abstract cache management in ModelBackend trait

### 2. Tokenizer Compatibility
- **HuggingFace**: tokenizer.json (most common)
- **SentencePiece**: tokenizer.model (Llama-1, some others)
- **Custom**: Some models have custom tokenizers

**Solution:** Implement fallback chain in tokenizer_loader

### 3. Config Field Variations
- Different models use different config field names
- Some use `num_hidden_layers`, others use `n_layers`
- Some use `hidden_size`, others use `d_model`

**Solution:** Normalize in model factory

---

## Verification Checklist

After completing all priorities:
- [ ] CPU build succeeds: `cargo build --release --features cpu --bin llorch-cpu-candled`
- [ ] CUDA build succeeds: `cargo build --release --features cuda --bin llorch-cuda-candled`
- [ ] All tests pass: `cargo test --features cpu`
- [ ] Architecture detection works for Llama, Mistral, Phi, Qwen
- [ ] Model loading works for at least 2 different architectures
- [ ] Tokenizer auto-detection works
- [ ] No Llama-specific hardcoding remains
- [ ] All team signatures added

---

## TEAM-015 Signing Off

**Status:** ✅ **BACKEND REFACTORED, WORKER-CRATES INTEGRATED**

**Key Achievements:**
- ✅ Refactored backend into 3 focused modules (287, 155, 28 lines)
- ✅ Integrated worker-common and worker-http into binary
- ✅ Removed unnecessary crate split
- ✅ Both CPU and CUDA builds verified
- ✅ Cleaner architecture, faster compilation

**Code Quality:**
- All files under 300 lines ✅
- Team signatures preserved ✅
- Both builds succeed ✅

**Handoff Notes:**
- The backend is now well-organized and ready for multi-model support
- Model loading is in `backend/model_loader.rs` (155 lines)
- Inference loop is in `backend/inference.rs` (287 lines)
- All Llama-specific code is clearly marked in this handoff

**Recommendation:** **Start with Priority 1 (model abstraction), then Priority 2 (tokenizer), then Priority 3 (tests).** The architecture is clean, now make it flexible.

---

*"One model is a prototype, many models is a platform."*  
— TEAM-015, 2025-10-08T23:25:00Z

**To TEAM-016: Remove the Llama-2 assumptions, add model abstraction, support all Candle models. The foundation is solid, now make it universal. 🚀**

**END HANDOFF**
