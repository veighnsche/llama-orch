# GT-002: tokenizer.json Loading

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 - HF Tokenizer  
**Size**: M (2 days)  
**Days**: 16 - 17  
**Spec Ref**: M0-W-1361

---

## Story Description

Implement robust tokenizer.json file discovery and loading logic for GPT-OSS-20B. Support loading from model directory, handle missing files gracefully, and integrate with model loading pipeline.

---

## Acceptance Criteria

- [ ] Tokenizer.json discovered in model directory (same dir as .gguf file)
- [ ] Fallback search paths implemented (./tokenizer.json, ../tokenizer.json)
- [ ] Error handling for missing tokenizer.json with clear error message
- [ ] Tokenizer loaded during model initialization (before CUDA operations)
- [ ] Tokenizer instance stored in model state for inference
- [ ] Unit tests validate file discovery logic
- [ ] Integration test validates loading with GPT-OSS-20B model directory
- [ ] Error messages include expected file path and search locations
- [ ] Logging at INFO level when tokenizer loaded successfully

---

## Dependencies

### Upstream (Blocks This Story)
- GT-001: HF Tokenizers Crate Integration (needs tokenizer module)

### Downstream (This Story Blocks)
- GT-003: Tokenizer Metadata Exposure (needs loaded tokenizer)
- GT-005: GPT GGUF Metadata Parsing (parallel work)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/model/loader.rs` - Add tokenizer loading logic
- `bin/worker-orcd/src/model/mod.rs` - Model struct holds tokenizer
- `bin/worker-orcd/src/tokenizer/discovery.rs` - File discovery logic

### Key Interfaces
```rust
use std::path::{Path, PathBuf};

pub struct TokenizerDiscovery;

impl TokenizerDiscovery {
    /// Find tokenizer.json relative to model file
    pub fn find_tokenizer_json(model_path: &Path) -> Result<PathBuf, TokenizerError> {
        // 1. Same directory as model file
        let model_dir = model_path.parent().unwrap_or(Path::new("."));
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            return Ok(tokenizer_path);
        }
        
        // 2. Current directory
        let cwd_path = PathBuf::from("./tokenizer.json");
        if cwd_path.exists() {
            return Ok(cwd_path);
        }
        
        // 3. Parent directory
        let parent_path = PathBuf::from("../tokenizer.json");
        if parent_path.exists() {
            return Ok(parent_path);
        }
        
        Err(TokenizerError::NotFound {
            searched_paths: vec![
                tokenizer_path.display().to_string(),
                cwd_path.display().to_string(),
                parent_path.display().to_string(),
            ],
        })
    }
}

pub struct Model {
    weights: DeviceMemory,
    tokenizer: Option<HfJsonTokenizer>,  // Some for GPT, None for Llama
    metadata: ModelMetadata,
}

impl Model {
    pub fn load(ctx: &Context, model_path: &Path) -> Result<Self, ModelError> {
        // 1. Parse GGUF metadata
        let metadata = parse_gguf_metadata(model_path)?;
        
        // 2. Load tokenizer if GPT architecture
        let tokenizer = if metadata.architecture == "gpt2" || metadata.architecture == "gpt" {
            let tokenizer_path = TokenizerDiscovery::find_tokenizer_json(model_path)?;
            info!("Loading tokenizer from: {}", tokenizer_path.display());
            Some(HfJsonTokenizer::from_file(tokenizer_path)?)
        } else {
            None
        };
        
        // 3. Load weights to VRAM
        let weights = load_weights_to_vram(ctx, model_path, &metadata)?;
        
        Ok(Model { weights, tokenizer, metadata })
    }
}
```

### Implementation Notes
- Search order: model directory → current directory → parent directory
- Log all searched paths in error message for debugging
- Load tokenizer before CUDA operations (fail fast)
- Store tokenizer in Model struct (Option<HfJsonTokenizer>)
- Only load for GPT architectures (gpt2/gpt)
- Validate tokenizer.json is valid JSON before loading

---

## Testing Strategy

### Unit Tests
- Test find_tokenizer_json with file in model directory
- Test find_tokenizer_json with file in current directory
- Test find_tokenizer_json with file in parent directory
- Test error handling when file not found
- Test error message includes all searched paths

### Integration Tests
- Test Model::load with GPT-OSS-20B directory structure
- Test Model::load fails gracefully when tokenizer.json missing
- Test Model::load skips tokenizer for Llama models
- Test tokenizer accessible after model load

### Manual Verification
1. Create test directory with model.gguf and tokenizer.json
2. Run: `cargo test tokenizer_loading`
3. Verify tokenizer loads from correct location
4. Test with missing tokenizer.json
5. Verify error message shows searched paths

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (5+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §8.2 HF-JSON Backend (M0-W-1361)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §6.4 Test Models (GPT-OSS-20B)
- Related Stories: GT-001 (crate integration), GT-003 (metadata)

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
