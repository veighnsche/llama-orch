# Research Assignment: llama.cpp GQA Attention Implementation

**Date**: 2025-10-05 22:58  
**Assignee**: AI Research Agent  
**Priority**: CRITICAL  
**Estimated Time**: 2-3 hours  

## Objective

Investigate the llama.cpp codebase to understand the correct implementation of GQA (Grouped Query Attention) for Qwen2.5 models, specifically focusing on how they handle attention computation, KV caching, and RoPE.

## Background

We have implemented GQA attention for Qwen2.5-0.5B but the model generates garbage output. The implementation includes:
- Attention score computation (Q·K^T)
- Softmax normalization
- Weighted sum of values
- KV caching per layer
- QKV bias addition
- RoPE application

However, the output is nonsensical, suggesting a fundamental bug in our implementation.

## Research Questions

### 1. GQA Attention Implementation

**Primary Questions**:
- How does llama.cpp implement GQA attention for Qwen2.5?
- What is the exact formula and order of operations?
- How do they handle the head grouping (14 Q heads → 2 KV heads)?

**Files to Investigate**:
- `ggml-cuda/` - CUDA kernels for attention
- `ggml.c` or `ggml.cpp` - Core attention operations
- Look for functions like:
  - `ggml_mul_mat` (matrix multiplication)
  - `ggml_soft_max` (softmax)
  - `ggml_rope` (rotary embeddings)
  - `ggml_flash_attn` or similar

**What to Document**:
1. Exact sequence of operations in attention
2. How Q, K, V are projected and reshaped
3. How attention scores are computed and scaled
4. How GQA head grouping is implemented
5. Memory layout of Q, K, V tensors (shape, strides)

### 2. KV Cache Management

**Primary Questions**:
- How is the KV cache structured in llama.cpp?
- How do they index into the cache for different layers?
- How is the cache updated during generation?

**Files to Investigate**:
- `llama.cpp` - Main inference loop
- Look for cache-related structures and functions
- Search for "kv_cache", "cache", or "past_key_values"

**What to Document**:
1. KV cache data structure (shape, layout)
2. How cache is allocated (per layer? per head?)
3. How cache position is tracked
4. How cache is indexed during decode
5. Any special handling for GQA vs MHA

### 3. RoPE (Rotary Position Embeddings)

**Primary Questions**:
- How does llama.cpp implement RoPE for Qwen2.5?
- What are the frequency base and scaling parameters?
- Is RoPE applied before or after KV caching?

**Files to Investigate**:
- `ggml-cuda/rope.cu` or similar
- `ggml.c` - RoPE CPU implementation
- Look for "rope", "rotary", or "position"

**What to Document**:
1. RoPE formula and implementation
2. Frequency base for Qwen2.5 (we use 1000000.0)
3. When RoPE is applied (before/after cache write)
4. How RoPE handles different head dimensions
5. Any special scaling or normalization

### 4. Qwen2.5-Specific Details

**Primary Questions**:
- Are there any Qwen2.5-specific quirks or modifications?
- How do they handle the QKV biases?
- Any differences from standard Llama architecture?

**Files to Investigate**:
- `llama.cpp` - Model loading and architecture detection
- Look for "qwen" or "qwen2" references
- Check model configuration parsing

**What to Document**:
1. Qwen2.5 architecture differences from Llama
2. How biases are handled (if at all)
3. Any special attention patterns or masks
4. Configuration parameters specific to Qwen2.5

### 5. Tensor Layouts and GEMM Operations

**Primary Questions**:
- What are the exact tensor shapes at each step?
- How are matrix multiplications configured (transpose flags)?
- What is the memory layout (row-major vs column-major)?

**Files to Investigate**:
- CUDA GEMM wrappers
- cuBLAS integration code
- Tensor reshape/transpose operations

**What to Document**:
1. Input tensor shapes for each operation
2. GEMM transpose flags (CUBLAS_OP_T vs CUBLAS_OP_N)
3. Leading dimensions (lda, ldb, ldc)
4. Memory layout assumptions

### 6. Numerical Stability and Precision

**Primary Questions**:
- Do they use FP16 or FP32 for attention computation?
- How do they handle numerical stability in softmax?
- Any special scaling or normalization?

**What to Document**:
1. Data types used for intermediate computations
2. Softmax numerical stability techniques
3. Attention score scaling factors
4. Any gradient clipping or normalization

## Deliverables

### 1. Implementation Comparison Document

Create a markdown file: `LLAMACPP_VS_OURS_COMPARISON.md`

**Structure**:
```markdown
# llama.cpp vs Our Implementation

## Attention Flow Comparison

### llama.cpp
1. Step 1: ...
2. Step 2: ...

### Our Implementation
1. Step 1: ...
2. Step 2: ...

### Differences
- [ ] Difference 1
- [ ] Difference 2

## Code Snippets

### llama.cpp Attention (Simplified)
```cpp
// Paste relevant code
```

### Our Attention
```cpp
// Paste our code
```

## Identified Issues
1. **Issue 1**: ...
2. **Issue 2**: ...
```

### 2. Bug Report

Create: `ATTENTION_BUGS_FOUND.md`

List specific bugs found by comparing implementations:
- Wrong tensor shapes
- Incorrect transpose flags
- Missing operations
- Wrong scaling factors
- Cache indexing errors

### 3. Fix Recommendations

Create: `ATTENTION_FIX_PLAN.md`

Prioritized list of fixes:
1. **Critical**: Fix X (will cause garbage output)
2. **High**: Fix Y (will cause numerical instability)
3. **Medium**: Fix Z (optimization)

## Research Methodology

### Step 1: Clone and Build llama.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make LLAMA_CUBLAS=1
```

### Step 2: Find Qwen2.5 Test Case
- Look for Qwen2.5 examples or tests
- Find minimal reproduction case
- Identify which functions are called

### Step 3: Trace Attention Path
- Start from main inference loop
- Follow function calls into attention
- Document each step with code snippets

### Step 4: Extract CUDA Kernels
- Find CUDA attention kernels
- Compare with our implementation
- Note differences in:
  - Thread/block configuration
  - Memory access patterns
  - Computation order

### Step 5: Verify with Debugger (Optional)
If possible:
- Run llama.cpp with Qwen2.5 model
- Set breakpoints in attention code
- Inspect tensor values at each step
- Compare with our values

## Key Files to Examine

Based on llama.cpp structure (as of 2024):

### Core Files
- `llama.cpp` - Main inference, model loading
- `ggml.c` or `ggml.cpp` - Core tensor operations
- `ggml-cuda.cu` - CUDA backend
- `ggml-cuda/` - CUDA kernels directory

### Attention-Specific
- `ggml-cuda/fattn.cu` - Flash attention (if used)
- `ggml-cuda/mmq.cu` - Matrix multiplication
- `ggml-cuda/rope.cu` - RoPE implementation

### Model-Specific
- Look for Qwen2/Qwen2.5 architecture code
- Check model type detection
- Find GQA-specific handling

## Expected Findings

We expect to find one or more of these issues:

### Likely Bugs in Our Implementation

1. **Cache Indexing**:
   - We might be indexing cache incorrectly
   - Layer offset calculation might be wrong
   - Batch/head dimensions might be swapped

2. **Attention Score Computation**:
   - Missing or wrong scaling factor
   - Incorrect Q·K^T computation
   - Wrong softmax implementation

3. **RoPE Application**:
   - Applied at wrong time
   - Wrong frequency calculation
   - Missing position offset

4. **Tensor Reshaping**:
   - Q/K/V not reshaped correctly for multi-head
   - Output not reshaped back correctly
   - Wrong leading dimensions in GEMM

5. **GQA Head Grouping**:
   - Incorrect mapping of Q heads to KV heads
   - Wrong replication of KV heads
   - Cache not properly shared across Q head groups

## Success Criteria

Research is complete when:

1. ✅ We understand llama.cpp's attention flow completely
2. ✅ We have identified specific differences from our implementation
3. ✅ We have at least 3 concrete bug hypotheses
4. ✅ We have code snippets showing the correct implementation
5. ✅ We have a prioritized fix plan

## Timeline

- **Hour 1**: Clone repo, find Qwen2.5 code, trace main path
- **Hour 2**: Deep dive into attention and cache implementation
- **Hour 3**: Document findings, create comparison, write bug report

## Notes for Researcher

- Focus on **CUDA implementation** (not CPU) since we're using CUDA
- Prioritize **decode path** (single token generation) over prefill
- Look for **comments or documentation** explaining design decisions
- Check **git history** for Qwen2.5-specific commits
- Search for **issues or PRs** related to Qwen2.5 or GQA

## Questions to Answer

At the end of research, you should be able to answer:

1. What is the exact order of operations in llama.cpp's attention?
2. How do they compute attention scores for GQA?
3. How is the KV cache structured and indexed?
4. When and how is RoPE applied?
5. What are the tensor shapes at each step?
6. What is different between their implementation and ours?
7. Which differences are likely causing our garbage output?

## Deliverable Format

Submit findings as:
1. Three markdown files (comparison, bugs, fixes)
2. Code snippets extracted from llama.cpp
3. Diagram (optional) showing attention flow
4. Summary of top 3 most likely bugs

---

**Status**: 🔴 NOT STARTED  
**Due**: ASAP (blocking haiku test)  
**Contact**: Report findings in `.plan/` directory
