# GT-052-V2: Architecture Registry + Weight Loading

**Story ID**: GT-052-V2  
**Title**: Implement Architecture Registry and GGUF Weight Loading  
**Size**: L (Large)  
**Estimate**: 8-10 hours  
**Priority**: P0 (Critical Path)  
**Dependencies**: GT-051 (Config Parsing)  
**Blocks**: GT-053, GT-054, GT-055, GT-056

---

## User Story

**As a** model loader  
**I want** a data-driven architecture registry and weight loading system  
**So that** I can support multiple model architectures without hardcoding

---

## Context

GT-051 implemented basic config parsing but hardcoded architecture-specific logic. This story refactors to use llama.cpp's proven patterns:
- Architecture registry (enum + data tables)
- Metadata key mapper (dynamic key construction)
- Tensor mapper (pattern expansion)
- Weight iterator (clean loading)

This is a **one-time investment** that makes adding new models trivial.

---

## Current State

**File**: `cuda/src/model/gpt_weights.cpp`

```cpp
// Hardcoded architecture handling
if (arch == "qwen2") {
    config.vocab_size = gguf::get_array_length(header.metadata, "tokenizer.ggml.tokens");
    config.hidden_dim = gguf::get_required_uint32(header.metadata, "qwen2.embedding_length");
    // ... more hardcoded keys
} else if (arch == "llama") {
    // ... different hardcoded keys
}
```

**Problems**:
- ❌ Adding Phi-3, GPT-OSS-20B requires code changes
- ❌ Tensor names are hardcoded
- ❌ Feature detection (QKV bias, fused QKV) is manual
- ❌ No clear separation of concerns

---

## Acceptance Criteria

### Architecture Registry
- [ ] `ArchitectureRegistry` class with enum and configs
- [ ] Support for Qwen2, Llama, GPT2 architectures
- [ ] Each config includes: metadata_prefix, feature defaults, tensor map
- [ ] Easy to add new architectures (just add config entry)

### Metadata Key Mapper
- [ ] `MetadataKeyMapper` class for dynamic key construction
- [ ] `get_uint32(key_suffix)` builds `"<prefix>.<suffix>"`
- [ ] `get_uint32(key_suffix, default)` for optional keys
- [ ] `get_vocab_size()` special case for tokenizer array

### Tensor Mapper
- [ ] `TensorMapper` class with pattern expansion
- [ ] `TensorMapping` struct: `{gguf_pattern, internal_name, required, per_layer}`
- [ ] Expand `{L}` placeholders for all layers
- [ ] `detect_features()` probes tensors for QKV bias, fused QKV

### Weight Iterator
- [ ] `GGUFWeightIterator` class for clean iteration
- [ ] `WeightEntry` struct: `{name, shape, quant_type, data, size}`
- [ ] `has_next()` and `next()` methods
- [ ] Integrates with mmap for zero-copy

### Weight Loading
- [ ] Load all tensors using iterator + tensor map
- [ ] Allocate GPU memory for each tensor
- [ ] Copy to GPU using chunked transfer
- [ ] Track VRAM usage
- [ ] Handle missing optional tensors gracefully

### Feature Detection
- [ ] Detect `has_qkv_bias` by probing `blk.0.attn_q.bias`
- [ ] Detect `qkv_fused` by probing `blk.0.attn_qkv.weight`
- [ ] Use arch defaults for `has_rope`, `norm_type`, `activation`
- [ ] Store in `ModelFeatures` struct

### Testing
- [ ] Unit tests for ArchitectureRegistry
- [ ] Unit tests for MetadataKeyMapper
- [ ] Unit tests for TensorMapper pattern expansion
- [ ] Unit tests for feature detection
- [ ] Integration test: Load Qwen2.5-0.5B weights to GPU
- [ ] Verify VRAM usage with `nvidia-smi`

---

## Technical Design

### 1. Architecture Registry

**File**: `cuda/src/model/architecture_registry.h`

```cpp
#ifndef WORKER_MODEL_ARCHITECTURE_REGISTRY_H
#define WORKER_MODEL_ARCHITECTURE_REGISTRY_H

#include <string>
#include <vector>
#include <map>

namespace worker {
namespace model {

enum class Architecture {
    QWEN2,
    LLAMA,
    GPT2,
    PHI3,
    UNKNOWN
};

enum class NormType {
    LAYER_NORM,
    RMS_NORM
};

enum class ActivationType {
    GELU,
    SWIGLU,
    RELU
};

struct TensorMapping {
    std::string gguf_pattern;      // "blk.{L}.attn_q.weight"
    std::string internal_name;     // "layers[{L}].attn_q"
    bool required;                 // true = error if missing
    bool per_layer;                // true = expand for each layer
};

struct ArchitectureConfig {
    std::string name;              // "qwen2"
    std::string metadata_prefix;   // "qwen2"
    
    // Feature defaults
    bool has_qkv_bias;             // Qwen2: true, Llama: false
    bool qkv_fused;                // GPT2: true, others: false
    bool has_rope;                 // Llama/Qwen2: true, GPT2: false
    NormType norm_type;            // RMSNorm vs LayerNorm
    ActivationType activation;     // SwiGLU vs GELU
    
    // Tensor name patterns
    std::vector<TensorMapping> tensor_map;
};

class ArchitectureRegistry {
public:
    static Architecture from_string(const std::string& arch_string);
    static std::string to_string(Architecture arch);
    static const ArchitectureConfig& get_config(Architecture arch);
    static bool is_supported(const std::string& arch_string);
    
private:
    static const std::map<Architecture, ArchitectureConfig> configs_;
};

} // namespace model
} // namespace worker

#endif
```

**File**: `cuda/src/model/architecture_registry.cpp`

```cpp
#include "architecture_registry.h"
#include <stdexcept>

namespace worker {
namespace model {

const std::map<Architecture, ArchitectureConfig> ArchitectureRegistry::configs_ = {
    {Architecture::QWEN2, {
        .name = "qwen2",
        .metadata_prefix = "qwen2",
        .has_qkv_bias = true,
        .qkv_fused = false,
        .has_rope = true,
        .norm_type = NormType::RMS_NORM,
        .activation = ActivationType::SWIGLU,
        .tensor_map = {
            // Embeddings
            {"token_embd.weight", "embeddings", true, false},
            
            // Attention (per layer)
            {"blk.{L}.attn_norm.weight", "layers[{L}].attn_norm", true, true},
            {"blk.{L}.attn_q.weight", "layers[{L}].attn_q", true, true},
            {"blk.{L}.attn_q.bias", "layers[{L}].attn_q_bias", true, true},
            {"blk.{L}.attn_k.weight", "layers[{L}].attn_k", true, true},
            {"blk.{L}.attn_k.bias", "layers[{L}].attn_k_bias", true, true},
            {"blk.{L}.attn_v.weight", "layers[{L}].attn_v", true, true},
            {"blk.{L}.attn_v.bias", "layers[{L}].attn_v_bias", true, true},
            {"blk.{L}.attn_output.weight", "layers[{L}].attn_out", true, true},
            
            // FFN (per layer)
            {"blk.{L}.ffn_norm.weight", "layers[{L}].ffn_norm", true, true},
            {"blk.{L}.ffn_gate.weight", "layers[{L}].ffn_gate", true, true},
            {"blk.{L}.ffn_up.weight", "layers[{L}].ffn_up", true, true},
            {"blk.{L}.ffn_down.weight", "layers[{L}].ffn_down", true, true},
            
            // Output
            {"output_norm.weight", "output_norm", true, false},
            {"output.weight", "lm_head", true, false},
        }
    }},
    
    {Architecture::LLAMA, {
        .name = "llama",
        .metadata_prefix = "llama",
        .has_qkv_bias = false,
        .qkv_fused = false,
        .has_rope = true,
        .norm_type = NormType::RMS_NORM,
        .activation = ActivationType::SWIGLU,
        .tensor_map = {
            // Similar to Qwen2 but no bias terms
            {"token_embd.weight", "embeddings", true, false},
            {"blk.{L}.attn_norm.weight", "layers[{L}].attn_norm", true, true},
            {"blk.{L}.attn_q.weight", "layers[{L}].attn_q", true, true},
            {"blk.{L}.attn_k.weight", "layers[{L}].attn_k", true, true},
            {"blk.{L}.attn_v.weight", "layers[{L}].attn_v", true, true},
            {"blk.{L}.attn_output.weight", "layers[{L}].attn_out", true, true},
            {"blk.{L}.ffn_norm.weight", "layers[{L}].ffn_norm", true, true},
            {"blk.{L}.ffn_gate.weight", "layers[{L}].ffn_gate", true, true},
            {"blk.{L}.ffn_up.weight", "layers[{L}].ffn_up", true, true},
            {"blk.{L}.ffn_down.weight", "layers[{L}].ffn_down", true, true},
            {"output_norm.weight", "output_norm", true, false},
            {"output.weight", "lm_head", true, false},
        }
    }},
    
    {Architecture::GPT2, {
        .name = "gpt2",
        .metadata_prefix = "gpt2",
        .has_qkv_bias = false,
        .qkv_fused = true,
        .has_rope = false,
        .norm_type = NormType::LAYER_NORM,
        .activation = ActivationType::GELU,
        .tensor_map = {
            {"token_embd.weight", "embeddings", true, false},
            {"position_embd.weight", "pos_embeddings", true, false},
            {"blk.{L}.attn_norm.weight", "layers[{L}].attn_norm", true, true},
            {"blk.{L}.attn_norm.bias", "layers[{L}].attn_norm_bias", true, true},
            {"blk.{L}.attn_qkv.weight", "layers[{L}].attn_qkv", true, true},
            {"blk.{L}.attn_output.weight", "layers[{L}].attn_out", true, true},
            {"blk.{L}.ffn_norm.weight", "layers[{L}].ffn_norm", true, true},
            {"blk.{L}.ffn_norm.bias", "layers[{L}].ffn_norm_bias", true, true},
            {"blk.{L}.ffn_up.weight", "layers[{L}].ffn_up", true, true},
            {"blk.{L}.ffn_down.weight", "layers[{L}].ffn_down", true, true},
            {"output_norm.weight", "output_norm", true, false},
            {"output_norm.bias", "output_norm_bias", true, false},
            {"output.weight", "lm_head", true, false},
        }
    }},
};

Architecture ArchitectureRegistry::from_string(const std::string& arch_string) {
    if (arch_string == "qwen2") return Architecture::QWEN2;
    if (arch_string == "llama") return Architecture::LLAMA;
    if (arch_string == "gpt2" || arch_string == "gpt") return Architecture::GPT2;
    if (arch_string == "phi3") return Architecture::PHI3;
    return Architecture::UNKNOWN;
}

std::string ArchitectureRegistry::to_string(Architecture arch) {
    auto it = configs_.find(arch);
    if (it != configs_.end()) {
        return it->second.name;
    }
    return "unknown";
}

const ArchitectureConfig& ArchitectureRegistry::get_config(Architecture arch) {
    auto it = configs_.find(arch);
    if (it == configs_.end()) {
        throw std::runtime_error("Unsupported architecture: " + to_string(arch));
    }
    return it->second;
}

bool ArchitectureRegistry::is_supported(const std::string& arch_string) {
    return from_string(arch_string) != Architecture::UNKNOWN;
}

} // namespace model
} // namespace worker
```

### 2. Metadata Key Mapper

**File**: `cuda/src/model/metadata_key_mapper.h`

```cpp
#ifndef WORKER_MODEL_METADATA_KEY_MAPPER_H
#define WORKER_MODEL_METADATA_KEY_MAPPER_H

#include "architecture_registry.h"
#include "../gguf/header_parser.h"
#include "../gguf/llama_metadata.h"
#include <string>
#include <vector>

namespace worker {
namespace model {

class MetadataKeyMapper {
public:
    MetadataKeyMapper(
        Architecture arch,
        const std::vector<gguf::GGUFMetadata>& metadata
    );
    
    // Required keys
    uint32_t get_uint32(const std::string& key_suffix);
    float get_float(const std::string& key_suffix);
    std::string get_string(const std::string& key_suffix);
    
    // Optional keys with defaults
    uint32_t get_uint32(const std::string& key_suffix, uint32_t default_value);
    float get_float(const std::string& key_suffix, float default_value);
    
    // Special: vocab_size from tokenizer
    uint32_t get_vocab_size();
    
    // Get architecture prefix
    const std::string& get_prefix() const { return prefix_; }
    
private:
    Architecture arch_;
    const std::vector<gguf::GGUFMetadata>& metadata_;
    std::string prefix_;
    const ArchitectureConfig* config_;
};

} // namespace model
} // namespace worker

#endif
```

**File**: `cuda/src/model/metadata_key_mapper.cpp`

```cpp
#include "metadata_key_mapper.h"

namespace worker {
namespace model {

MetadataKeyMapper::MetadataKeyMapper(
    Architecture arch,
    const std::vector<gguf::GGUFMetadata>& metadata
) : arch_(arch), metadata_(metadata) {
    config_ = &ArchitectureRegistry::get_config(arch);
    prefix_ = config_->metadata_prefix;
}

uint32_t MetadataKeyMapper::get_uint32(const std::string& key_suffix) {
    std::string full_key = prefix_ + "." + key_suffix;
    return gguf::get_required_uint32(metadata_, full_key);
}

float MetadataKeyMapper::get_float(const std::string& key_suffix) {
    std::string full_key = prefix_ + "." + key_suffix;
    return gguf::get_required_float(metadata_, full_key);
}

std::string MetadataKeyMapper::get_string(const std::string& key_suffix) {
    std::string full_key = prefix_ + "." + key_suffix;
    return gguf::get_required_string(metadata_, full_key);
}

uint32_t MetadataKeyMapper::get_uint32(const std::string& key_suffix, uint32_t default_value) {
    std::string full_key = prefix_ + "." + key_suffix;
    return gguf::get_optional_uint32(metadata_, full_key, default_value);
}

float MetadataKeyMapper::get_float(const std::string& key_suffix, float default_value) {
    std::string full_key = prefix_ + "." + key_suffix;
    return gguf::get_optional_float(metadata_, full_key, default_value);
}

uint32_t MetadataKeyMapper::get_vocab_size() {
    return gguf::get_array_length(metadata_, "tokenizer.ggml.tokens");
}

} // namespace model
} // namespace worker
```

### 3. Tensor Mapper

**File**: `cuda/src/model/tensor_mapper.h`

```cpp
#ifndef WORKER_MODEL_TENSOR_MAPPER_H
#define WORKER_MODEL_TENSOR_MAPPER_H

#include "architecture_registry.h"
#include "../gguf/header_parser.h"
#include <string>
#include <vector>

namespace worker {
namespace model {

struct ModelFeatures {
    bool has_qkv_bias;
    bool qkv_fused;
    bool has_rope;
    NormType norm_type;
    ActivationType activation;
};

class TensorMapper {
public:
    TensorMapper(Architecture arch, int num_layers);
    
    // Expand patterns for all layers
    std::vector<TensorMapping> get_expanded_mappings();
    
    // Detect features by probing tensors
    ModelFeatures detect_features(const std::vector<gguf::GGUFTensor>& tensors);
    
private:
    Architecture arch_;
    int num_layers_;
    const ArchitectureConfig* config_;
    
    std::string replace_placeholder(const std::string& pattern, const std::string& placeholder, const std::string& value);
    const gguf::GGUFTensor* find_tensor(const std::vector<gguf::GGUFTensor>& tensors, const std::string& name);
};

} // namespace model
} // namespace worker

#endif
```

**File**: `cuda/src/model/tensor_mapper.cpp`

```cpp
#include "tensor_mapper.h"
#include <algorithm>

namespace worker {
namespace model {

TensorMapper::TensorMapper(Architecture arch, int num_layers)
    : arch_(arch), num_layers_(num_layers) {
    config_ = &ArchitectureRegistry::get_config(arch);
}

std::vector<TensorMapping> TensorMapper::get_expanded_mappings() {
    std::vector<TensorMapping> result;
    
    for (const auto& mapping : config_->tensor_map) {
        if (mapping.per_layer) {
            // Expand {L} for each layer
            for (int L = 0; L < num_layers_; ++L) {
                result.push_back({
                    .gguf_pattern = replace_placeholder(mapping.gguf_pattern, "{L}", std::to_string(L)),
                    .internal_name = replace_placeholder(mapping.internal_name, "{L}", std::to_string(L)),
                    .required = mapping.required,
                    .per_layer = false
                });
            }
        } else {
            result.push_back(mapping);
        }
    }
    
    return result;
}

ModelFeatures TensorMapper::detect_features(const std::vector<gguf::GGUFTensor>& tensors) {
    ModelFeatures features;
    
    // Probe for QKV bias
    features.has_qkv_bias = find_tensor(tensors, "blk.0.attn_q.bias") != nullptr;
    
    // Probe for fused QKV
    features.qkv_fused = find_tensor(tensors, "blk.0.attn_qkv.weight") != nullptr;
    
    // Use arch defaults for others
    features.has_rope = config_->has_rope;
    features.norm_type = config_->norm_type;
    features.activation = config_->activation;
    
    return features;
}

std::string TensorMapper::replace_placeholder(
    const std::string& pattern,
    const std::string& placeholder,
    const std::string& value
) {
    std::string result = pattern;
    size_t pos = result.find(placeholder);
    if (pos != std::string::npos) {
        result.replace(pos, placeholder.length(), value);
    }
    return result;
}

const gguf::GGUFTensor* TensorMapper::find_tensor(
    const std::vector<gguf::GGUFTensor>& tensors,
    const std::string& name
) {
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [&name](const gguf::GGUFTensor& t) { return t.name == name; });
    return (it != tensors.end()) ? &(*it) : nullptr;
}

} // namespace model
} // namespace worker
```

### 4. Weight Iterator

**File**: `cuda/src/model/weight_iterator.h`

```cpp
#ifndef WORKER_MODEL_WEIGHT_ITERATOR_H
#define WORKER_MODEL_WEIGHT_ITERATOR_H

#include "../gguf/header_parser.h"
#include "../io/mmap_file.h"
#include <string>
#include <vector>

namespace worker {
namespace model {

enum class QuantType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4_K_M,
    Q8_0,
    MXFP4,
    UNKNOWN
};

struct WeightEntry {
    std::string name;
    std::vector<uint64_t> shape;
    QuantType quant_type;
    const void* data;
    size_t size_bytes;
};

class WeightIterator {
public:
    virtual ~WeightIterator() = default;
    virtual bool has_next() = 0;
    virtual WeightEntry next() = 0;
};

class GGUFWeightIterator : public WeightIterator {
public:
    GGUFWeightIterator(const io::MmapFile& mmap, const gguf::GGUFHeader& header);
    
    bool has_next() override;
    WeightEntry next() override;
    
private:
    const io::MmapFile& mmap_;
    const std::vector<gguf::GGUFTensor>& tensors_;
    size_t current_;
    size_t data_start_;
    
    QuantType map_ggml_type(uint32_t ggml_type);
};

} // namespace model
} // namespace worker

#endif
```

### 5. Updated GPTWeightLoader

**File**: `cuda/src/model/gpt_weights.cpp` (updated)

```cpp
GPTConfig GPTWeightLoader::parse_config_from_gguf(const std::string& path) {
    // Open and memory-map the GGUF file
    auto mmap = io::MmapFile::open(path);
    
    // Parse GGUF header
    auto header = gguf::parse_gguf_header(mmap.data(), mmap.size());
    
    // Detect architecture
    std::string arch_string = gguf::get_required_string(header.metadata, "general.architecture");
    Architecture arch = ArchitectureRegistry::from_string(arch_string);
    
    if (arch == Architecture::UNKNOWN) {
        throw std::runtime_error("Unsupported architecture: " + arch_string);
    }
    
    // Create metadata key mapper
    MetadataKeyMapper mapper(arch, header.metadata);
    
    // Extract config using mapper
    GPTConfig config;
    config.vocab_size = mapper.get_vocab_size();
    config.hidden_dim = mapper.get_uint32("embedding_length");
    config.num_layers = mapper.get_uint32("block_count");
    config.num_heads = mapper.get_uint32("attention.head_count");
    config.ffn_dim = mapper.get_uint32("feed_forward_length");
    config.context_length = mapper.get_uint32("context_length");
    config.max_seq_len = config.context_length;
    config.head_dim = config.hidden_dim / config.num_heads;
    
    // Extract quantization type from first tensor
    if (!header.tensors.empty()) {
        config.quant_kind = map_quant_type(header.tensors[0].type);
    }
    
    return config;
}

std::unique_ptr<GPTModelWeights> GPTWeightLoader::load_from_gguf(const std::string& path) {
    // Parse config
    GPTConfig config = parse_config_from_gguf(path);
    
    // Open and memory-map
    auto mmap = io::MmapFile::open(path);
    auto header = gguf::parse_gguf_header(mmap.data(), mmap.size());
    
    // Detect architecture
    std::string arch_string = gguf::get_required_string(header.metadata, "general.architecture");
    Architecture arch = ArchitectureRegistry::from_string(arch_string);
    
    // Create tensor mapper
    TensorMapper tensor_mapper(arch, config.num_layers);
    auto mappings = tensor_mapper.get_expanded_mappings();
    auto features = tensor_mapper.detect_features(header.tensors);
    
    // Create weight iterator
    GGUFWeightIterator weight_iter(mmap, header);
    
    // Allocate model
    auto model = std::make_unique<GPTModelWeights>();
    model->config = config;
    
    // Load weights using iterator
    while (weight_iter.has_next()) {
        auto entry = weight_iter.next();
        
        // Find mapping for this tensor
        auto it = std::find_if(mappings.begin(), mappings.end(),
            [&entry](const TensorMapping& m) { return m.gguf_pattern == entry.name; });
        
        if (it != mappings.end()) {
            // Allocate and copy to GPU
            void* gpu_ptr = allocate_and_copy(entry.data, entry.size_bytes, entry.name);
            
            // Store in model (simplified - actual code would route to correct field)
            // ... routing logic based on it->internal_name
        } else if (/* tensor is required */) {
            throw std::runtime_error("Required tensor missing: " + entry.name);
        }
    }
    
    return model;
}
```

---

## Testing Strategy

### Unit Tests

**File**: `cuda/tests/test_architecture_registry.cpp`

```cpp
TEST(ArchitectureRegistry, FromString) {
    EXPECT_EQ(ArchitectureRegistry::from_string("qwen2"), Architecture::QWEN2);
    EXPECT_EQ(ArchitectureRegistry::from_string("llama"), Architecture::LLAMA);
    EXPECT_EQ(ArchitectureRegistry::from_string("gpt2"), Architecture::GPT2);
    EXPECT_EQ(ArchitectureRegistry::from_string("unknown"), Architecture::UNKNOWN);
}

TEST(ArchitectureRegistry, GetConfig) {
    auto& config = ArchitectureRegistry::get_config(Architecture::QWEN2);
    EXPECT_EQ(config.name, "qwen2");
    EXPECT_EQ(config.metadata_prefix, "qwen2");
    EXPECT_TRUE(config.has_qkv_bias);
    EXPECT_FALSE(config.qkv_fused);
    EXPECT_TRUE(config.has_rope);
}
```

**File**: `cuda/tests/test_tensor_mapper.cpp`

```cpp
TEST(TensorMapper, PatternExpansion) {
    TensorMapper mapper(Architecture::QWEN2, 2);
    auto mappings = mapper.get_expanded_mappings();
    
    // Should have expanded layer patterns
    bool found_layer_0 = false;
    bool found_layer_1 = false;
    
    for (const auto& m : mappings) {
        if (m.gguf_pattern == "blk.0.attn_q.weight") found_layer_0 = true;
        if (m.gguf_pattern == "blk.1.attn_q.weight") found_layer_1 = true;
    }
    
    EXPECT_TRUE(found_layer_0);
    EXPECT_TRUE(found_layer_1);
}

TEST(TensorMapper, FeatureDetection) {
    TensorMapper mapper(Architecture::QWEN2, 24);
    
    std::vector<gguf::GGUFTensor> tensors = {
        {.name = "blk.0.attn_q.bias", /* ... */},
        {.name = "blk.0.attn_k.bias", /* ... */},
    };
    
    auto features = mapper.detect_features(tensors);
    
    EXPECT_TRUE(features.has_qkv_bias);  // Detected from tensors
    EXPECT_FALSE(features.qkv_fused);    // Not present
    EXPECT_TRUE(features.has_rope);      // From arch default
}
```

### Integration Test

**File**: `cuda/tests/test_weight_loading_integration.cpp`

```cpp
TEST(WeightLoading, LoadQwen25ToGPU) {
    const std::string model_path = 
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Load weights
    auto model = GPTWeightLoader::load_from_gguf(model_path);
    
    // Verify config
    EXPECT_EQ(model->config.vocab_size, 151936);
    EXPECT_EQ(model->config.hidden_dim, 896);
    EXPECT_EQ(model->config.num_layers, 24);
    
    // Verify weights loaded
    EXPECT_NE(model->token_embeddings, nullptr);
    EXPECT_EQ(model->layers.size(), 24);
    
    for (int i = 0; i < 24; ++i) {
        EXPECT_NE(model->layers[i]->attn_q_weight, nullptr);
        EXPECT_NE(model->layers[i]->attn_q_bias, nullptr);  // Qwen2 has bias
    }
    
    // Verify VRAM usage
    size_t vram_used = get_current_vram_usage();
    EXPECT_GT(vram_used, 0);
    
    fprintf(stderr, "VRAM used: %.2f MB\n", vram_used / 1024.0 / 1024.0);
}
```

---

## Implementation Steps

1. **Create ArchitectureRegistry** (2 hours)
   - Define enum and structs
   - Add Qwen2, Llama, GPT2 configs
   - Write unit tests

2. **Create MetadataKeyMapper** (1 hour)
   - Implement key construction
   - Add required/optional methods
   - Write unit tests

3. **Create TensorMapper** (2 hours)
   - Implement pattern expansion
   - Implement feature detection
   - Write unit tests

4. **Create WeightIterator** (1 hour)
   - Implement GGUF iterator
   - Add quant type mapping
   - Write unit tests

5. **Update GPTWeightLoader** (3 hours)
   - Refactor parse_config_from_gguf()
   - Implement load_from_gguf() with iterator
   - Handle tensor routing

6. **Integration Testing** (1 hour)
   - Test with Qwen2.5-0.5B
   - Verify VRAM usage
   - Verify all weights loaded

---

## Definition of Done

- [x] All classes implemented and compile
- [x] Unit tests pass (100% coverage for new code)
- [x] Integration test loads Qwen2.5-0.5B to GPU
- [x] VRAM usage verified with `nvidia-smi`
- [x] No hardcoded architecture logic remains
- [x] Easy to add new architecture (just add config entry)
- [x] Code reviewed and approved
- [x] Documentation updated

---

## Dependencies

**Requires**:
- GT-051 (Config parsing - DONE)
- Existing GGUF parser
- Existing mmap utilities

**Blocks**:
- GT-053 (Tokenizer)
- GT-054 (Transformer execution)
- GT-055 (LM head)
- GT-056 (Wire inference)

---

## Time Estimate

**Optimistic**: 8 hours  
**Realistic**: 8-10 hours  
**Pessimistic**: 12 hours (if debugging tensor routing is complex)

---

## Notes

### Why This Approach?

1. **Data-driven**: Add new models by config, not code
2. **Proven**: llama.cpp uses this for 20+ architectures
3. **Maintainable**: Clear separation of concerns
4. **Testable**: Each component tested independently
5. **Future-proof**: Easy to extend

### Risks

- ⚠️ Tensor routing logic can be complex
- ⚠️ Need to handle optional tensors carefully
- ⚠️ Feature detection needs thorough testing

### Mitigation

- Start with Qwen2 only, add others incrementally
- Write comprehensive unit tests
- Test with actual model file early

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO  
**Priority**: P0 (Critical Path)

---
Test opportunities identified by Testing Team 🔍
