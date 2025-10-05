# llama.cpp Reference Guide: What We Need to Know

**Date**: 2025-10-05  
**Purpose**: Document what llama.cpp does so we can implement dynamically instead of hardcoding  
**Philosophy**: Learn their patterns, don't hardcode our implementation

---

## Executive Summary

Instead of hardcoding Qwen2.5-specific logic, we should understand **how llama.cpp handles multiple architectures dynamically**. This document identifies what to learn from their codebase.

---

## Key Questions to Answer from llama.cpp

### 1. How do they detect and handle different architectures?

**What to look for in llama.cpp**:
- Where do they check `general.architecture`?
- How do they map architecture string to loading logic?
- Do they use a factory pattern or switch statement?
- How do they handle unknown architectures?

**Files to examine**:
```
llama.cpp/llama.cpp
  - Search for: "general.architecture"
  - Search for: "llm_load_hparams"
  - Search for: "enum llm_arch"
```

**What we want to learn**:
```cpp
// Example pattern we might find:
enum class Architecture {
    LLAMA,
    QWEN2,
    GPT2,
    PHI3,
    UNKNOWN
};

Architecture detect_architecture(const std::string& arch_string) {
    if (arch_string == "llama") return Architecture::LLAMA;
    if (arch_string == "qwen2") return Architecture::QWEN2;
    if (arch_string == "gpt2") return Architecture::GPT2;
    // ...
    return Architecture::UNKNOWN;
}
```

**Why this matters**: We can support multiple models without hardcoding each one.

---

### 2. How do they map metadata keys dynamically?

**What to look for in llama.cpp**:
- Do they have a mapping table for metadata keys?
- How do they handle architecture-specific prefixes (`llama.*` vs `qwen2.*`)?
- Do they fall back to generic keys?
- How do they handle missing metadata?

**Files to examine**:
```
llama.cpp/llama.cpp
  - Search for: "llm_kv"
  - Search for: "gguf_get_val"
  - Look for metadata key construction
```

**What we want to learn**:
```cpp
// Example pattern we might find:
class MetadataKeyMapper {
public:
    std::string get_key(Architecture arch, const std::string& param) {
        std::string prefix = get_prefix(arch);
        return prefix + "." + param;
    }
    
private:
    std::string get_prefix(Architecture arch) {
        switch (arch) {
            case Architecture::LLAMA: return "llama";
            case Architecture::QWEN2: return "qwen2";
            case Architecture::GPT2: return "gpt2";
            default: return "unknown";
        }
    }
};

// Usage:
std::string vocab_key = mapper.get_key(arch, "vocab_size");
// Returns "qwen2.vocab_size" for Qwen2, "llama.vocab_size" for Llama, etc.
```

**Why this matters**: We can extract config for any architecture without hardcoding keys.

---

### 3. How do they map tensor names dynamically?

**What to look for in llama.cpp**:
- Do they have a tensor name mapping table?
- How do they handle architecture-specific tensor names?
- Do they use a naming convention or lookup table?
- How do they handle optional tensors?

**Files to examine**:
```
llama.cpp/llama.cpp
  - Search for: "llm_load_tensors"
  - Search for: "tensor_map"
  - Look for tensor name construction
```

**What we want to learn**:
```cpp
// Example pattern we might find:
struct TensorMapping {
    std::string gguf_name;     // Name in GGUF file
    std::string internal_name;  // Name in our structure
    bool required;              // Is this tensor required?
};

class TensorMapper {
public:
    std::vector<TensorMapping> get_mappings(Architecture arch) {
        switch (arch) {
            case Architecture::QWEN2:
                return {
                    {"token_embd.weight", "embeddings", true},
                    {"blk.{L}.attn_q.weight", "layers[{L}].attn_q", true},
                    {"blk.{L}.attn_q.bias", "layers[{L}].attn_q_bias", true},  // Qwen-specific
                    {"blk.{L}.attn_k.weight", "layers[{L}].attn_k", true},
                    {"blk.{L}.attn_k.bias", "layers[{L}].attn_k_bias", true},  // Qwen-specific
                    // ...
                };
            case Architecture::LLAMA:
                return {
                    {"token_embd.weight", "embeddings", true},
                    {"blk.{L}.attn_q.weight", "layers[{L}].attn_q", true},
                    // Note: No bias for pure Llama
                    // ...
                };
        }
    }
};
```

**Why this matters**: We can load any model without hardcoding tensor names.

---

### 4. How do they handle architecture-specific features?

**What to look for in llama.cpp**:
- How do they detect if a model uses RoPE vs learned positions?
- How do they detect if QKV are fused or separate?
- How do they detect if bias terms exist?
- How do they handle GQA vs MHA vs MQA?

**Files to examine**:
```
llama.cpp/llama.cpp
  - Search for: "rope"
  - Search for: "n_head_kv"
  - Search for: "qkv"
  - Look for feature detection logic
```

**What we want to learn**:
```cpp
// Example pattern we might find:
struct ArchitectureFeatures {
    bool has_rope;
    bool has_learned_positions;
    bool has_qkv_bias;
    bool qkv_fused;
    AttentionType attention_type;  // MHA, GQA, MQA
    ActivationType ffn_activation;  // GELU, SwiGLU, etc.
    NormType norm_type;  // LayerNorm, RMSNorm
};

ArchitectureFeatures detect_features(Architecture arch, const GGUFMetadata& metadata) {
    ArchitectureFeatures features;
    
    // Detect from metadata
    features.has_rope = metadata.has_key(prefix + ".rope.freq_base");
    features.has_learned_positions = !features.has_rope;
    
    // Detect attention type from head counts
    uint32_t n_head = metadata.get_uint(prefix + ".attention.head_count");
    uint32_t n_head_kv = metadata.get_uint(prefix + ".attention.head_count_kv", n_head);
    
    if (n_head_kv == 1) {
        features.attention_type = AttentionType::MQA;
    } else if (n_head_kv < n_head) {
        features.attention_type = AttentionType::GQA;
    } else {
        features.attention_type = AttentionType::MHA;
    }
    
    // Detect from architecture
    if (arch == Architecture::QWEN2) {
        features.has_qkv_bias = true;
        features.qkv_fused = false;
        features.ffn_activation = ActivationType::SwiGLU;
        features.norm_type = NormType::RMSNorm;
    } else if (arch == Architecture::GPT2) {
        features.has_qkv_bias = false;
        features.qkv_fused = true;
        features.ffn_activation = ActivationType::GELU;
        features.norm_type = NormType::LayerNorm;
    }
    
    return features;
}
```

**Why this matters**: We can adapt to different model architectures automatically.

---

### 5. How do they handle tokenizer variations?

**What to look for in llama.cpp**:
- How do they detect tokenizer type (BPE, Unigram, etc.)?
- How do they extract vocab and merges?
- How do they handle different tokenizer formats?
- Do they have fallbacks?

**Files to examine**:
```
llama.cpp/llama.cpp
  - Search for: "llama_vocab"
  - Search for: "tokenizer.ggml.model"
  - Search for: "vocab_load"
```

**What we want to learn**:
```cpp
// Example pattern we might find:
enum class TokenizerType {
    BPE,
    UNIGRAM,
    WORD_PIECE,
    UNKNOWN
};

TokenizerType detect_tokenizer_type(const GGUFMetadata& metadata) {
    std::string model = metadata.get_string("tokenizer.ggml.model", "");
    
    if (model == "gpt2" || model == "bpe") return TokenizerType::BPE;
    if (model == "unigram") return TokenizerType::UNIGRAM;
    if (model == "wordpiece") return TokenizerType::WORD_PIECE;
    
    return TokenizerType::UNKNOWN;
}

std::unique_ptr<Tokenizer> create_tokenizer(TokenizerType type, const GGUFMetadata& metadata) {
    switch (type) {
        case TokenizerType::BPE:
            return std::make_unique<BPETokenizer>(
                metadata.get_array_string("tokenizer.ggml.tokens"),
                metadata.get_array_string("tokenizer.ggml.merges"),
                metadata.get_uint("tokenizer.ggml.bos_token_id"),
                metadata.get_uint("tokenizer.ggml.eos_token_id")
            );
        // ...
    }
}
```

**Why this matters**: We can support different tokenizers without hardcoding.

---

### 6. How do they handle quantization formats?

**What to look for in llama.cpp**:
- How do they detect quantization type from tensor metadata?
- Do they have a registry of quantization formats?
- How do they handle mixed quantization (different tensors with different quants)?

**Files to examine**:
```
llama.cpp/ggml-quants.c
llama.cpp/llama.cpp
  - Search for: "ggml_type"
  - Search for: "quantize"
  - Look for quantization type enum
```

**What we want to learn**:
```cpp
// Example pattern we might find:
enum class QuantType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4_K_M,
    Q8_0,
    MXFP4,
    // ...
};

QuantType get_tensor_quant_type(const GGUFTensor& tensor) {
    switch (tensor.type) {
        case GGUF_TYPE_F32: return QuantType::F32;
        case GGUF_TYPE_F16: return QuantType::F16;
        case GGUF_TYPE_Q4_K: return QuantType::Q4_K_M;
        // ...
    }
}

size_t get_tensor_size(const GGUFTensor& tensor) {
    QuantType quant = get_tensor_quant_type(tensor);
    size_t elements = calculate_elements(tensor.dimensions);
    return elements * get_bytes_per_element(quant);
}
```

**Why this matters**: We can handle any quantization format dynamically.

---

## Implementation Strategy

### Phase 1: Learn the Patterns (Research)

**Tasks**:
1. Clone llama.cpp repository
2. Read the files listed above
3. Extract the patterns they use
4. Document their approach

**Deliverable**: Document showing their patterns

### Phase 2: Implement Generic System (GT-051, GT-052)

**Instead of**:
```cpp
// Hardcoded for Qwen2
if (arch == "qwen2") {
    config.vocab_size = metadata.get_uint("qwen2.vocab_size");
    config.hidden_dim = metadata.get_uint("qwen2.embedding_length");
    // ...
}
```

**Do this**:
```cpp
// Generic for any architecture
std::string prefix = get_architecture_prefix(arch);
config.vocab_size = metadata.get_uint(prefix + ".vocab_size");
config.hidden_dim = metadata.get_uint(prefix + ".embedding_length");
// ...
```

### Phase 3: Add Architecture Configs (Data-Driven)

**Create**: `cuda/src/model/architecture_registry.cpp`

```cpp
struct ArchitectureConfig {
    std::string name;
    std::string metadata_prefix;
    bool has_qkv_bias;
    bool qkv_fused;
    bool has_rope;
    // ...
};

// Registry of known architectures
static const std::map<std::string, ArchitectureConfig> ARCHITECTURES = {
    {"qwen2", {
        .name = "qwen2",
        .metadata_prefix = "qwen2",
        .has_qkv_bias = true,
        .qkv_fused = false,
        .has_rope = true,
        // ...
    }},
    {"llama", {
        .name = "llama",
        .metadata_prefix = "llama",
        .has_qkv_bias = false,
        .qkv_fused = false,
        .has_rope = true,
        // ...
    }},
    {"gpt2", {
        .name = "gpt2",
        .metadata_prefix = "gpt2",
        .has_qkv_bias = false,
        .qkv_fused = true,
        .has_rope = false,
        // ...
    }},
};
```

**Benefits**:
- Add new architectures by adding config, not code
- No hardcoded if/else chains
- Easy to maintain

---

## Specific llama.cpp Files to Study

### Priority 1: Core Loading Logic

1. **`llama.cpp/llama.cpp`** (lines 1000-3000)
   - Function: `llm_load_hparams()`
   - Function: `llm_load_tensors()`
   - Enum: `enum llm_arch`
   - Shows: How they handle multiple architectures

2. **`llama.cpp/llama.cpp`** (lines 5000-7000)
   - Function: `llama_model_loader::load_all_data()`
   - Shows: How they load tensors dynamically

### Priority 2: Tokenizer Logic

3. **`llama.cpp/llama.cpp`** (lines 8000-10000)
   - Struct: `llama_vocab`
   - Function: `llama_vocab_init()`
   - Shows: How they handle different tokenizers

### Priority 3: Architecture Detection

4. **`llama.cpp/llama.cpp`** (search for "LLM_ARCH_")
   - Shows: All supported architectures
   - Shows: How they map arch string to enum

---

## Questions to Answer

### For GT-051 (Config Parsing)

1. How does llama.cpp construct metadata keys for different architectures?
2. Do they have a fallback mechanism for missing keys?
3. How do they validate extracted config?

### For GT-052 (Weight Loading)

1. How does llama.cpp map tensor names for different architectures?
2. How do they handle optional tensors?
3. How do they detect if QKV are fused or separate?
4. How do they detect if bias terms exist?

### For GT-053 (Tokenizer)

1. How does llama.cpp detect tokenizer type?
2. How do they handle byte-level vs character-level BPE?
3. Do they have a generic tokenizer interface?

---

## Expected Outcomes

After studying llama.cpp, we should be able to:

1. **Support multiple architectures** without hardcoding each one
2. **Detect features automatically** from metadata
3. **Handle variations gracefully** (optional tensors, different formats)
4. **Add new models easily** by adding config, not code

---

## Action Items

### Immediate (Before GT-051)

1. ⬜ Clone llama.cpp repository
2. ⬜ Read `llm_load_hparams()` function
3. ⬜ Document their architecture detection pattern
4. ⬜ Document their metadata key construction

### Before GT-052

5. ⬜ Read `llm_load_tensors()` function
6. ⬜ Document their tensor name mapping
7. ⬜ Document how they detect optional tensors

### Before GT-053

8. ⬜ Read `llama_vocab_init()` function
9. ⬜ Document their tokenizer detection
10. ⬜ Document their BPE implementation

---

## Success Criteria

We've learned enough from llama.cpp when we can answer:

1. ✅ "How do they support 20+ architectures without 20+ if statements?"
2. ✅ "How do they construct metadata keys dynamically?"
3. ✅ "How do they detect which tensors are optional?"
4. ✅ "How do they handle different tokenizer types?"
5. ✅ "What patterns can we copy to avoid hardcoding?"

---

**Created by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Purpose**: Learn patterns from llama.cpp to avoid hardcoding  
**Philosophy**: Understand their approach, implement generically
