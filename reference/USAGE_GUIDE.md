# Reference Code Usage Guide

**Updated**: 2025-10-06

## Quick Start

The llama.cpp submodule is now available for code inspection:

```bash
# Already initialized - just browse the code
cd reference/llama.cpp

# Update to latest (optional)
git submodule update --remote reference/llama.cpp
```

## Key Files for Debugging Our Inference Issues

### GGUF Parsing & Weight Loading

```
reference/llama.cpp/
├── ggml.h                    # Core GGML types and structures
├── ggml.c                    # GGML implementation
├── ggml-cuda.cu              # CUDA kernels (compare to ours)
├── llama.cpp                 # Main inference engine
└── llama.h                   # Public API
```

### Specific Areas to Reference

#### 1. **GGUF Weight Loading** (for our garbage output issue)
- `llama.cpp` - Search for `llm_load_tensors()`
- How they map tensor names to model layers
- Weight alignment and padding

#### 2. **Tokenizer Integration**
- `llama.cpp` - Search for `llama_tokenize()`
- How they handle BOS/EOS tokens
- Vocab loading from GGUF metadata

#### 3. **Attention Kernels**
- `ggml-cuda.cu` - Search for `flash_attn` or `ggml_cuda_op_mul_mat`
- Compare to our `bin/worker-orcd/cuda/kernels/attention.cu`

#### 4. **Sampling**
- `common/sampling.cpp` - Temperature, top-k, top-p implementation
- Compare to our `bin/worker-orcd/cuda/kernels/sampling.cu`

#### 5. **Q4_K Dequantization**
- `ggml-cuda.cu` - Search for `dequantize_q4_K`
- Compare to our `bin/worker-orcd/cuda/kernels/dequantize.cu`

## Debugging Workflow

### Current Issue: Garbage Token Output

**Our test**: `bin/worker-orcd/tests/haiku_generation_anti_cheat.rs`
**Problem**: Model generates garbage tokens (e.g., `ðŁĽ´×Ļ×Ľ×ķ×Ļ`)

**Reference these llama.cpp files**:

1. **Check weight loading order**:
   ```bash
   cd reference/llama.cpp
   grep -n "token_embd" llama.cpp
   grep -n "output.weight" llama.cpp
   ```

2. **Compare attention implementation**:
   ```bash
   grep -n "llm_build_kv" llama.cpp
   ```

3. **Check tokenizer integration**:
   ```bash
   grep -n "llama_token_to_piece" llama.cpp
   ```

4. **Verify sampling logic**:
   ```bash
   cd common
   grep -n "sample_top_k" sampling.cpp
   ```

## Comparison Checklist

When debugging, compare:

- [ ] Tensor name mapping (theirs vs ours)
- [ ] Weight loading order
- [ ] Attention mask generation
- [ ] RoPE frequency calculation
- [ ] KV cache management
- [ ] Sampling temperature application
- [ ] Token decoding logic

## Example: Debugging Weight Loading

```bash
# In llama.cpp reference
cd reference/llama.cpp
grep -A 20 "llm_load_tensors" llama.cpp > /tmp/llamacpp_weight_loading.txt

# Compare to our implementation
cd bin/worker-orcd
grep -A 20 "load_weights_to_gpu" src/cuda/weight_loader.rs > /tmp/our_weight_loading.txt

# Diff them
diff /tmp/llamacpp_weight_loading.txt /tmp/our_weight_loading.txt
```

## Remember

- ✅ **Read** their code to understand the algorithm
- ✅ **Compare** their approach to ours
- ✅ **Learn** from their implementation choices
- ❌ **Don't** copy-paste without understanding
- ❌ **Don't** link to their libraries
- ❌ **Don't** import their headers

## Useful grep Patterns

```bash
cd reference/llama.cpp

# Find all CUDA kernels
grep -r "^__global__" *.cu

# Find quantization functions
grep -r "dequantize_q4" *.cu

# Find attention implementations
grep -r "flash_attn\|scaled_dot_product" *.cu *.cpp

# Find sampling logic
grep -r "sample_top_k\|sample_temperature" common/

# Find GGUF parsing
grep -r "gguf_get_tensor" *.cpp
```

## Next Steps for Our Garbage Output Issue

Based on the test output showing repeated garbage tokens:

1. **Check tokenizer**: Are we decoding token IDs correctly?
   - Reference: `llama.cpp` - `llama_token_to_piece()`
   - Our code: `bin/worker-orcd/src/inference/`

2. **Verify logits**: Are we computing correct output probabilities?
   - Reference: `ggml-cuda.cu` - final layer matmul
   - Our code: `bin/worker-orcd/cuda/kernels/matmul.cu`

3. **Check weight alignment**: Are tensors loaded at correct offsets?
   - Reference: `llama.cpp` - `llm_load_tensors()`
   - Our code: `bin/worker-orcd/src/cuda/weight_loader.rs`

---

**Remember**: We are building a competitor, not a clone. Learn from them, but implement it our way.
