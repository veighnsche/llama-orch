# GT-051: GGUF Config Parsing from Real File

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: S (2-3 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: M0-W-1211, M0-W-1212

---

## Story Description

Parse actual GGUF file metadata to extract model configuration instead of returning hardcoded values. This enables architecture detection and proper model initialization.

---

## Current State (STUB)

**File**: `cuda/src/model/gpt_weights.cpp` line 335

```cpp
GPTConfig GPTWeightLoader::parse_config_from_gguf(const std::string& path) {
    // TODO: Parse actual GGUF file
    // For now, return GPT-OSS-20B config as stub
    GPTConfig config;
    config.vocab_size = 50257;      // Hardcoded!
    config.hidden_dim = 2048;       // Hardcoded!
    config.num_layers = 44;         // Hardcoded!
    config.num_heads = 64;          // Hardcoded!
    config.head_dim = 32;           // Hardcoded!
    config.ffn_dim = 8192;          // Hardcoded!
    config.max_seq_len = 2048;      // Hardcoded!
    config.context_length = 8192;   // Hardcoded!
    config.quant_kind = "Q4_K_M";   // Hardcoded!
    
    return config;
}
```

**Problem**: Returns hardcoded GPT-OSS-20B config regardless of actual model file.

---

- [ ] Parse GGUF header using existing `parse_gguf_header()` function
- [ ] Extract config from GGUF metadata using existing `extract_llama_config()` helper
- [ ] Map GGUF metadata keys to `GPTConfig` struct fields
- [ ] Detect architecture from `general.architecture` metadata key
- [ ] Validate extracted config (all required fields present)
- [ ] Works for Qwen2.5-0.5B (llama architecture)
- [ ] Works for GPT-OSS-20B (gpt2 architecture)
- [ ] Remove hardcoded config values
- [ ] Unit test verifies correct config extraction

## Technical Details

### Implementation

**File**: `cuda/src/model/gpt_weights.cpp`

**IMPORTANT**: Research shows Qwen2.5-0.5B uses architecture="qwen2" with metadata keys prefixed `qwen2.*`

```cpp
GPTConfig GPTWeightLoader::parse_config_from_gguf(const std::string& path) {
    // 1. Use existing mmap and parser
    auto mmap = io::MmapFile::open(path);
    auto header = gguf::parse_gguf_header(mmap.data(), mmap.size());
    
    // 2. Extract architecture
    std::string arch = "unknown";
    for (const auto& kv : header.metadata) {
        if (kv.key == "general.architecture" && kv.value_type == gguf::GGUFValueType::STRING) {
            arch = kv.string_value;
            break;
        }
    }
    
    // 3. For Qwen2 models (Qwen2.5-0.5B)
    // Research: architecture="qwen2", keys prefixed with "qwen2."
    if (arch == "qwen2") {
        GPTConfig config;
        
        // Extract required fields (from RESEARCH_RESULTS.md Table 2)
        for (const auto& kv : header.metadata) {
            if (kv.key == "qwen2.vocab_size") {
                config.vocab_size = kv.uint_value;  // 151,643
            } else if (kv.key == "qwen2.embedding_length") {
                config.hidden_dim = kv.uint_value;  // 896
            } else if (kv.key == "qwen2.block_count") {
                config.num_layers = kv.uint_value;  // 24
            } else if (kv.key == "qwen2.attention.head_count") {
                config.num_heads = kv.uint_value;  // 14
            } else if (kv.key == "qwen2.attention.head_count_kv") {
                config.num_kv_heads = kv.uint_value;  // 2 (GQA)
            } else if (kv.key == "qwen2.feed_forward_length") {
                config.ffn_dim = kv.uint_value;  // 4,864
            } else if (kv.key == "qwen2.context_length") {
                config.max_seq_len = kv.uint_value;  // 32,768
                config.context_length = kv.uint_value;
            } else if (kv.key == "qwen2.rope.dimension_count") {
                config.rope_dim = kv.uint_value;  // 64
            } else if (kv.key == "qwen2.rope.freq_base") {
                config.rope_freq_base = kv.float_value;  // 1,000,000.0
            } else if (kv.key == "qwen2.attention.layer_norm_rms_epsilon") {
                config.rms_norm_eps = kv.float_value;  // 1e-6
            }
        }
        
        config.head_dim = config.hidden_dim / config.num_heads;  // 896 / 14 = 64
        
        // Extract quantization
        if (!header.tensors.empty()) {
            config.quant_kind = gguf::tensor_type_to_string(header.tensors[0].type);
        }
        
        return config;
    }
    
    // 4. For pure Llama models (if needed later)
    if (arch == "llama") {
        auto llama_config = gguf::extract_llama_config(header.metadata);
        // ... (existing code)
    }
    
    // 5. For GPT-style models (if needed later)
    if (arch == "gpt2" || arch == "gpt") {
        // ... (existing code)
    }
    
    throw CudaError::model_load_failed("Unsupported architecture: " + arch);
}
```

### Existing Code to Use

**Already implemented**:
- ✅ `gguf::parse_gguf_header()` - Parses GGUF file
- ✅ `gguf::extract_llama_config()` - Extracts Llama metadata
- ✅ `io::MmapFile::open()` - Memory-maps file
- ✅ `GGUFHeader` struct with metadata vector
- ✅ `GGUFMetadata` struct with key/value pairs

**Files to reference**:
- `cuda/src/gguf/header_parser.cpp` - GGUF parsing
- `cuda/src/gguf/llama_metadata.cpp` - Metadata extraction
- `cuda/src/io/mmap_file.cpp` - File I/O

---

## Testing Strategy

### Unit Test

**File**: `cuda/tests/test_gpt_weights.cpp`

```cpp
TEST(GPTWeightLoader, ParseConfigFromGGUF_Qwen) {
    // Test with real Qwen model
    std::string path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    GPTConfig config = GPTWeightLoader::parse_config_from_gguf(path);
    
    // Verify config extracted correctly
    EXPECT_GT(config.vocab_size, 0);
    EXPECT_GT(config.hidden_dim, 0);
    EXPECT_GT(config.num_layers, 0);
    EXPECT_GT(config.num_heads, 0);
    EXPECT_EQ(config.head_dim, config.hidden_dim / config.num_heads);
    EXPECT_GT(config.ffn_dim, 0);
    EXPECT_GT(config.max_seq_len, 0);
    EXPECT_FALSE(config.quant_kind.empty());
    
    // Verify it's not the hardcoded GPT-OSS-20B config
    EXPECT_NE(config.vocab_size, 50257);  // GPT-OSS-20B value
}

TEST(GPTWeightLoader, ParseConfigFromGGUF_InvalidFile) {
    EXPECT_THROW(
        GPTWeightLoader::parse_config_from_gguf("/nonexistent/file.gguf"),
        CudaError
    );
}
```

### Manual Verification

```bash
# Build and run test
cd bin/worker-orcd
cargo build --release

# Verify config extraction
./target/release/worker-orcd \
  --model /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --gpu-device 0 \
  --port 8080 \
  --worker-id test-$(uuidgen)

# Should see in logs:
# - Correct vocab size (not 50257)
# - Correct hidden dim (not 2048)
# - Correct architecture detected
```

---

## Dependencies

**Upstream**: None (uses existing code)  
**Downstream**: GT-052 (needs real config for weight loading)

---

## Definition of Done

- [ ] Code implemented and compiles
- [ ] Hardcoded values removed
- [ ] Unit tests pass
- [ ] Manual verification with Qwen model shows correct config
- [ ] No TODOs remain in `parse_config_from_gguf()`
- [ ] Story marked complete

---

## Estimated Time

**Optimistic**: 2 hours  
**Realistic**: 2-3 hours  
**Pessimistic**: 4 hours (if metadata extraction is tricky)

---

## Notes

### Why This is Quick

- ✅ GGUF parser already exists and works
- ✅ Metadata extraction helpers already exist
- ✅ Just need to wire them together
- ✅ No new code, just use existing functions

### Risks

- ⚠️ Different GGUF versions might have different metadata keys
- ⚠️ Some models might be missing required metadata

**Mitigation**: Add clear error messages for missing metadata

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO  
**Related Fine**: FINE-001-20251005
