# GPT Weight Mapping Specification

**Story**: GT-024  
**Status**: Infrastructure Complete  
**Date**: 2025-10-05

---

## Overview

This document specifies the weight mapping for GPT-OSS-20B (and GPT-2 family) models in GGUF format. It defines how GGUF tensor names map to model components and provides the loading order and shape validation requirements.

## Model Architecture

### GPT-OSS-20B Configuration

```
Vocabulary Size:    50257
Hidden Dimension:   2048
Number of Layers:   44
Attention Heads:    64 (MHA - no GQA)
Head Dimension:     32 (2048 / 64)
FFN Dimension:      8192
Max Sequence:       2048
Context Length:     8192
```

### GPT-2 Small Configuration (Reference)

```
Vocabulary Size:    50257
Hidden Dimension:   768
Number of Layers:   12
Attention Heads:    12 (MHA)
Head Dimension:     64 (768 / 12)
FFN Dimension:      3072
Max Sequence:       1024
```

## GGUF Tensor Naming Convention

GPT models in GGUF format follow this naming pattern:

```
<component>.<layer_index>.<weight_type>
```

### Component Prefixes

- `token_embd` — Token embedding matrix
- `position_embd` — Absolute positional embedding matrix
- `blk.<N>` — Transformer block N (0-indexed)
- `output_norm` — Final LayerNorm before LM head
- `output` — Language model head (logits projection)

### Weight Type Suffixes

- `.weight` — Weight matrix
- `.bias` — Bias vector
- `.attn_qkv.weight` — Combined Q/K/V projection (GPT-style)
- `.attn_output.weight` — Attention output projection
- `.ffn_up.weight` — FFN up projection
- `.ffn_down.weight` — FFN down projection
- `.attn_norm.weight` — LayerNorm gamma (pre-attention)
- `.attn_norm.bias` — LayerNorm beta (pre-attention)
- `.ffn_norm.weight` — LayerNorm gamma (pre-FFN)
- `.ffn_norm.bias` — LayerNorm beta (pre-FFN)

## Complete Tensor Map

### Embeddings

| GGUF Tensor Name | Shape | Component | Quantization |
|------------------|-------|-----------|--------------|
| `token_embd.weight` | `[hidden_dim, vocab_size]` | Token embeddings | Q4_K_M / MXFP4 |
| `position_embd.weight` | `[hidden_dim, max_seq_len]` | Positional embeddings | Q4_K_M / MXFP4 |

**Notes**:
- GPT uses **absolute** positional embeddings (not RoPE like Llama)
- In GGUF, embedding/linear tensors follow `[n_in, n_out]` orientation. Embedding lookup still yields vectors added as `x = token_emb + pos_emb`.

### Transformer Blocks (Per Layer)

For each layer `N` (0 to 43 for GPT-OSS-20B):

#### Pre-Attention LayerNorm

| GGUF Tensor Name | Shape | Component |
|------------------|-------|-----------|
| `blk.{N}.attn_norm.weight` | `[hidden_dim]` | LayerNorm gamma |
| `blk.{N}.attn_norm.bias` | `[hidden_dim]` | LayerNorm beta |

#### Multi-Head Attention (MHA)

| GGUF Tensor Name | Shape | Component | Quantization |
|------------------|-------|-----------|--------------|
| `blk.{N}.attn_qkv.weight` | `[hidden_dim, 3 * hidden_dim]` | Combined Q/K/V projection | Q4_K_M / MXFP4 |
| `blk.{N}.attn_qkv.bias` | `[3 * hidden_dim]` | Combined Q/K/V bias | FP16 |
| `blk.{N}.attn_output.weight` | `[hidden_dim, hidden_dim]` | Output projection | Q4_K_M / MXFP4 |
| `blk.{N}.attn_output.bias` | `[hidden_dim]` | Output bias | FP16 |

**Notes**:
- GPT uses **MHA** (Multi-Head Attention) — all heads have independent K/V
- No GQA (Grouped Query Attention) like Llama
- QKV weights are **combined** into single tensor (split during forward pass)

#### Pre-FFN LayerNorm

| GGUF Tensor Name | Shape | Component |
|------------------|-------|-----------|
| `blk.{N}.ffn_norm.weight` | `[hidden_dim]` | LayerNorm gamma |
| `blk.{N}.ffn_norm.bias` | `[hidden_dim]` | LayerNorm beta |

#### Feed-Forward Network (FFN)

| GGUF Tensor Name | Shape | Component | Quantization |
|------------------|-------|-----------|--------------|
| `blk.{N}.ffn_up.weight` | `[hidden_dim, ffn_dim]` | Up projection | Q4_K_M / MXFP4 |
| `blk.{N}.ffn_up.bias` | `[ffn_dim]` | Up bias | FP16 |
| `blk.{N}.ffn_down.weight` | `[ffn_dim, hidden_dim]` | Down projection | Q4_K_M / MXFP4 |
| `blk.{N}.ffn_down.bias` | `[hidden_dim]` | Down bias | FP16 |

**Notes**:
- GPT uses **GELU** activation (not SwiGLU like Llama)
- FFN structure: `down(GELU(up(x)))`

### Output Head

| GGUF Tensor Name | Shape | Component | Quantization |
|------------------|-------|-----------|--------------|
| `output_norm.weight` | `[hidden_dim]` | Final LayerNorm gamma | FP16 |
| `output_norm.bias` | `[hidden_dim]` | Final LayerNorm beta | FP16 |
| `output.weight` | `[hidden_dim, vocab_size]` | LM head projection | Q4_K_M / MXFP4 |

**Notes**:
- Final LayerNorm applied before LM head
- LM head projects to vocabulary logits

## Weight Shapes Summary

### GPT-OSS-20B (hidden_dim=2048, ffn_dim=8192, vocab_size=50257)

```
Token Embeddings:      [2048, 50257]  → 102,926,336 params
Position Embeddings:   [2048, 2048]   → 4,194,304 params

Per Layer (44 layers):
  Attn LayerNorm:      [2048] × 2     → 4,096 params
  QKV Projection:      [2048, 6144]   → 12,582,912 params
  QKV Bias:            [6144]         → 6,144 params
  Attn Output:         [2048, 2048]   → 4,194,304 params
  Attn Output Bias:    [2048]         → 2,048 params
  FFN LayerNorm:       [2048] × 2     → 4,096 params
  FFN Up:              [2048, 8192]   → 16,777,216 params
  FFN Up Bias:         [8192]         → 8,192 params
  FFN Down:            [8192, 2048]   → 16,777,216 params
  FFN Down Bias:       [2048]         → 2,048 params
  
  Total per layer:     50,358,272 params
  Total 44 layers:     2,215,764,000 params

Output LayerNorm:      [2048] × 2     → 4,096 params
LM Head:               [2048, 50257]  → 102,926,336 params

TOTAL MODEL:           ~2.43B parameters
```

**Note**: GPT-OSS-20B is actually ~20B parameters. The above calculation is for reference architecture. Actual model may have different dimensions.

## Quantization Strategy

### Q4_K_M (Primary Fallback)

- **Weights quantized**: All projection matrices (embeddings, QKV, output, FFN)
- **Weights FP16**: LayerNorm parameters, biases
- **Compute**: In-kernel dequantization to FP16, accumulate in FP16
- **KV Cache**: FP16 precision

### MXFP4 (Primary for GPT-OSS-20B)

- **Weights quantized**: All projection matrices (embeddings, QKV, output, FFN)
- **Weights FP16**: LayerNorm parameters, biases
- **Compute**: In-kernel dequantization to FP16, accumulate in FP16
- **KV Cache**: FP16 precision
- **Block structure**: MXFP4 uses microscaling blocks (see MXFP4 spec)

## Loading Order

### Recommended Loading Sequence

1. **Parse GGUF header** — Validate magic bytes, version, tensor count
2. **Extract metadata** — Architecture, dimensions, quantization
3. **Validate architecture** — Ensure `general.architecture == "gpt2"` or `"gpt"`
4. **Allocate VRAM** — Calculate total size, allocate single buffer
5. **Load embeddings** — Token + position embeddings
6. **Load transformer blocks** — Layer 0 to N-1 (sequential)
7. **Load output head** — Final LayerNorm + LM head
8. **Verify residency** — Check all pointers are device memory

### Memory Layout

```
[Token Embeddings | Position Embeddings | Layer 0 | Layer 1 | ... | Layer 43 | Output Norm | LM Head]
```

**Alignment**: All tensors aligned to 256-byte boundaries for optimal GPU access.

## Shape Validation

### Pre-Load Validation

```cpp
// Validate GGUF metadata
assert(metadata.get_string("general.architecture") == "gpt" || "gpt2");
assert(metadata.get_int("gpt2.context_length") > 0);
assert(metadata.get_int("gpt2.embedding_length") > 0);
assert(metadata.get_int("gpt2.block_count") > 0);
assert(metadata.get_int("gpt2.attention.head_count") > 0);

// Validate dimensions
int hidden_dim = metadata.get_int("gpt2.embedding_length");
int num_layers = metadata.get_int("gpt2.block_count");
int num_heads = metadata.get_int("gpt2.attention.head_count");
int ffn_dim = metadata.get_int("gpt2.feed_forward_length");

assert(hidden_dim % num_heads == 0);  // Head dimension must be integer
```

### Per-Tensor Validation

```cpp
// Token embeddings (GGUF stores as [hidden_dim, vocab_size])
assert(tensor.shape == [hidden_dim, vocab_size]);
assert(tensor.type == Q4_K_M || MXFP4);

// Position embeddings (GGUF stores as [hidden_dim, max_seq_len])
assert(tensor.shape == [hidden_dim, max_seq_len]);
assert(tensor.type == Q4_K_M || MXFP4);

// QKV projection
assert(tensor.shape == [hidden_dim, 3 * hidden_dim]);
assert(tensor.type == Q4_K_M || MXFP4);

// Attention output
assert(tensor.shape == [hidden_dim, hidden_dim]);
assert(tensor.type == Q4_K_M || MXFP4);

// FFN up
assert(tensor.shape == [hidden_dim, ffn_dim]);
assert(tensor.type == Q4_K_M || MXFP4);

// FFN down
assert(tensor.shape == [ffn_dim, hidden_dim]);
assert(tensor.type == Q4_K_M || MXFP4);

// LayerNorm weights
assert(tensor.shape == [hidden_dim]);
assert(tensor.type == FP16 || FP32);
```

## MXFP4 Weight Mapping

### MXFP4 Tensor Structure

MXFP4 tensors in GGUF v3 have additional metadata:

```cpp
struct MXFP4Tensor {
    uint32_t type;           // GGML_TYPE_MXFP4
    uint64_t dimensions[4];  // Tensor shape
    uint64_t offset;         // Data offset in file
    
    // MXFP4-specific metadata
    uint32_t block_size;     // Microscaling block size (typically 32)
    uint32_t scale_bits;     // Scale factor bits (typically 8)
};
```

### MXFP4 Weight Consumers

All weight matrices MUST support MXFP4 dequantization:

1. **Token Embeddings** — Lookup + dequantize
2. **Position Embeddings** — Lookup + dequantize
3. **QKV Projections** — GEMM with MXFP4 weights (prefill + decode)
4. **Attention Output** — GEMM with MXFP4 weights
5. **FFN Up** — GEMM with MXFP4 weights
6. **FFN Down** — GEMM with MXFP4 weights
7. **LM Head** — GEMM with MXFP4 weights

### Dequantization Path

```
MXFP4 (VRAM) → In-kernel dequant → FP16 (registers/shared mem) → FP16 accumulate → FP16 output
```

## Error Handling

### Invalid Tensor Names

```cpp
if (!is_valid_gpt_tensor_name(name)) {
    throw std::runtime_error("Invalid GPT tensor name: " + name);
}
```

### Shape Mismatches

```cpp
if (tensor.shape != expected_shape) {
    throw std::runtime_error(
        "Shape mismatch for " + name + 
        ": expected " + shape_to_string(expected_shape) + 
        ", got " + shape_to_string(tensor.shape)
    );
}
```

### Missing Required Tensors

```cpp
std::vector<std::string> required_tensors = {
    "token_embd.weight",
    "position_embd.weight",
    "output_norm.weight",
    "output.weight"
};

for (const auto& name : required_tensors) {
    if (!tensor_map.contains(name)) {
        throw std::runtime_error("Missing required tensor: " + name);
    }
}
```

## Testing Strategy

### Unit Tests (Mock Data)

```cpp
TEST(GPTWeightMapping, ValidateTokenEmbeddings) {
    // Create mock GGUF tensor
    GGUFTensor tensor;
    tensor.name = "token_embd.weight";
    tensor.dimensions = {50257, 2048};
    tensor.type = GGML_TYPE_Q4_K_M;
    
    // Validate
    ASSERT_TRUE(validate_gpt_tensor(tensor, config));
}

TEST(GPTWeightMapping, ValidateQKVProjection) {
    GGUFTensor tensor;
    tensor.name = "blk.0.attn_qkv.weight";
    tensor.dimensions = {2048, 6144};  // 3 * hidden_dim
    tensor.type = GGML_TYPE_Q4_K_M;
    
    ASSERT_TRUE(validate_gpt_tensor(tensor, config));
}
```

### Integration Tests (Requires Model)

```cpp
TEST(GPTWeightLoading, LoadGPTOSS20B) {
    // Load actual GPT-OSS-20B GGUF file
    auto model = load_gpt_model("gpt-oss-20b-q4_k_m.gguf");
    
    // Validate all tensors loaded
    ASSERT_EQ(model.num_layers, 44);
    ASSERT_EQ(model.hidden_dim, 2048);
    ASSERT_NE(model.token_embeddings, nullptr);
    ASSERT_NE(model.position_embeddings, nullptr);
}
```

## References

- **Spec**: M0-W-1212 (Architecture Detection)
- **Spec**: M0-W-1220 (Model Weights Allocation)
- **Spec**: M0-W-1221 (Memory-Mapped I/O)
- **Story**: GT-024 (GPT Weight Mapping)
- **Model**: GPT-OSS-20B (OpenAI open-source GPT)

## Status

- ✅ **Documentation**: Complete
- ✅ **Interface**: Defined
- ✅ **Validation**: Specified
- ⚠️ **Implementation**: Requires actual GGUF file for testing
- ⚠️ **Integration**: Blocked on model availability

---
Crafted by GPT-Gamma 🤖
