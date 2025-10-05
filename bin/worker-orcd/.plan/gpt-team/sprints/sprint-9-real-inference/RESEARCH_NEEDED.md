# Sprint 9: Research Needed for Real Inference Implementation

**Date**: 2025-10-05  
**Sprint**: Sprint 9 - Real Inference  
**Purpose**: Identify specific implementation details requiring internet research

---

## Executive Summary

We have:
- ✅ M0 spec with requirements
- ✅ Working GGUF parser
- ✅ Working CUDA kernels
- ✅ Model structure defined

We need **specific implementation details** that aren't in our codebase or spec.

---

## Research Questions (Prioritized)

### 🔴 CRITICAL: GT-051 & GT-052 (GGUF Weight Loading)

#### Q1: What are the exact GGUF tensor names for Qwen2.5-0.5B?

**Why we need this**:
- Our code has structure but doesn't know which GGUF tensor maps to which weight
- Example: Is token embedding at `token_embd.weight` or `model.embed_tokens.weight`?

**What to research**:
```bash
# Option 1: Inspect actual file
python3 -m gguf_dump /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf | grep "tensor.*name"

# Option 2: Check llama.cpp source
# Search for "qwen" or "llama" tensor loading code
```

**Expected output**:
```
Tensor names for Qwen2.5-0.5B (llama architecture):
- token_embd.weight          -> model->token_embeddings
- blk.0.attn_norm.weight     -> layer[0]->attn_norm_weight
- blk.0.attn_q.weight        -> layer[0]->attn_q_weight
- blk.0.attn_k.weight        -> layer[0]->attn_k_weight
- blk.0.attn_v.weight        -> layer[0]->attn_v_weight
- blk.0.attn_output.weight   -> layer[0]->attn_out_weight
- blk.0.ffn_norm.weight      -> layer[0]->ffn_norm_weight
- blk.0.ffn_gate.weight      -> layer[0]->ffn_gate_weight
- blk.0.ffn_up.weight        -> layer[0]->ffn_up_weight
- blk.0.ffn_down.weight      -> layer[0]->ffn_down_weight
- output_norm.weight         -> model->output_norm_weight
- output.weight              -> model->lm_head_weight
```

**Where to look**:
1. Run `gguf_dump.py` on actual Qwen model
2. llama.cpp source: `llama.cpp` file, search for `llm_load_tensors`
3. HuggingFace model card for Qwen2.5-0.5B-Instruct-GGUF

**Story impact**: GT-052 (can't load weights without knowing names)

---

#### Q2: What are the exact GGUF metadata keys for model config?

**Why we need this**:
- GT-051 needs to extract config from metadata
- Spec says `llama.context_length` but is that the actual key?

**What to research**:
```bash
python3 -m gguf_dump /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf | grep "metadata"
```

**Expected output**:
```
Metadata keys for Qwen (llama architecture):
- general.architecture       = "llama"
- llama.vocab_size           = 151936
- llama.embedding_length     = 896
- llama.block_count          = 24
- llama.attention.head_count = 14
- llama.attention.head_count_kv = 2  (GQA)
- llama.feed_forward_length  = 4864
- llama.context_length       = 32768
- llama.rope.dimension_count = 64
- llama.rope.freq_base       = 1000000.0
```

**Where to look**:
1. Run `gguf_dump.py` on actual Qwen model
2. llama.cpp source: `llama_model_loader::get_hparams()`
3. GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

**Story impact**: GT-051 (can't parse config without knowing keys)

---

#### Q3: Are some tensors optional or fused?

**Why we need this**:
- Some models don't have bias terms
- Some models fuse QKV into one tensor
- Need to know which tensors are required vs optional

**What to research**:
- Check if Qwen has bias terms (probably not - Llama-style models typically don't)
- Check if QKV are separate or fused
- Check if position embeddings exist (RoPE models don't have learned pos embeddings)

**Expected output**:
```
Qwen2.5-0.5B tensor requirements:
- ✅ REQUIRED: token_embd.weight
- ❌ OPTIONAL: token_embd.bias (doesn't exist - no bias)
- ✅ REQUIRED: blk.{N}.attn_q.weight, attn_k.weight, attn_v.weight (separate)
- ❌ OPTIONAL: blk.{N}.attn_qkv.weight (fused - doesn't exist for Qwen)
- ❌ NOT PRESENT: position_embd.weight (uses RoPE instead)
- ✅ REQUIRED: output.weight
- ❌ OPTIONAL: output.bias (doesn't exist)
```

**Where to look**:
1. Inspect actual GGUF file tensor list
2. Qwen model architecture documentation
3. llama.cpp tensor loading code (shows which tensors are optional)

**Story impact**: GT-052 (need to handle missing tensors gracefully)

---

### 🟡 IMPORTANT: GT-053 (Tokenizer)

#### Q4: How to extract tokenizer vocab and merges from GGUF?

**Why we need this**:
- Need to implement BPE tokenizer
- Vocab and merge rules are in GGUF metadata
- Don't know the exact format

**What to research**:
```bash
python3 -m gguf_dump /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf | grep "tokenizer"
```

**Expected output**:
```
Tokenizer metadata keys:
- tokenizer.ggml.model           = "gpt2"  (BPE variant)
- tokenizer.ggml.tokens          = [array of 151936 strings]
- tokenizer.ggml.merges          = [array of merge rules]
- tokenizer.ggml.bos_token_id    = 151643
- tokenizer.ggml.eos_token_id    = 151643
- tokenizer.ggml.unknown_token_id = 0
- tokenizer.ggml.padding_token_id = 151643

Format of tokens array: ["!", "\"", "#", "$", ...]
Format of merges array: ["Ġ t", "Ġ a", "h e", ...]
```

**Where to look**:
1. Run `gguf_dump.py` on actual Qwen model
2. llama.cpp source: `llama_vocab::load()` function
3. GGUF spec section on tokenizer metadata

**Story impact**: GT-053 (can't implement tokenizer without vocab/merges)

---

#### Q5: What is the BPE merge algorithm?

**Why we need this**:
- Need to implement `encode(text) -> token_ids`
- BPE applies merge rules iteratively
- Don't have the exact algorithm

**What to research**:
- How BPE merge algorithm works step-by-step
- How to handle byte-level BPE (used by GPT-2/Qwen)
- How to handle special tokens

**Expected output**:
```python
# Pseudocode for BPE encoding
def encode(text, vocab, merges):
    # 1. Convert text to bytes
    bytes = text.encode('utf-8')
    
    # 2. Convert bytes to initial tokens (byte-level)
    tokens = [byte_to_token(b) for b in bytes]
    
    # 3. Apply merges iteratively
    while True:
        # Find most frequent adjacent pair
        pairs = get_adjacent_pairs(tokens)
        if not pairs:
            break
        
        # Find highest priority merge
        best_pair = find_best_merge(pairs, merges)
        if not best_pair:
            break
        
        # Merge the pair
        tokens = merge_pair(tokens, best_pair)
    
    # 4. Convert tokens to IDs
    return [vocab[token] for token in tokens]
```

**Where to look**:
1. HuggingFace tokenizers source code (Rust implementation)
2. Original BPE paper: https://arxiv.org/abs/1508.07909
3. GPT-2 BPE implementation (byte-level variant)
4. llama.cpp tokenizer implementation

**Story impact**: GT-053 (core algorithm for tokenizer)

---

### 🟢 NICE-TO-HAVE: Optimization Details

#### Q6: What are the optimal cuBLAS settings for LM head?

**Why we need this**:
- GT-055 uses cuBLAS for LM head projection
- Want to use Tensor Cores if available
- Don't know optimal settings

**What to research**:
```cpp
// What should these be?
cublasGemmEx(
    handle,
    CUBLAS_OP_T,  // or CUBLAS_OP_N?
    CUBLAS_OP_N,
    M, N, K,
    &alpha,
    A, CUDA_R_16F,  // or CUDA_R_32F?
    lda,
    B, CUDA_R_16F,
    ldb,
    &beta,
    C, CUDA_R_16F,
    ldc,
    CUBLAS_COMPUTE_16F,  // or CUBLAS_COMPUTE_32F_FAST_16F?
    CUBLAS_GEMM_DEFAULT  // or CUBLAS_GEMM_DEFAULT_TENSOR_OP?
);
```

**Expected output**:
- Use `CUBLAS_COMPUTE_32F_FAST_16F` for FP16 with FP32 accumulation
- Use `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to enable Tensor Cores
- Transpose settings depend on weight layout

**Where to look**:
1. cuBLAS documentation
2. NVIDIA blog posts on GEMM optimization
3. llama.cpp cuBLAS usage

**Story impact**: GT-055 (optimization, not blocking)

---

## Research Priority

### Must Have (Blocking)

1. **Q1: GGUF tensor names** - Can't load weights without this
2. **Q2: GGUF metadata keys** - Can't parse config without this
3. **Q4: Tokenizer vocab/merges extraction** - Can't implement tokenizer without this
4. **Q5: BPE algorithm** - Can't encode/decode without this

### Should Have (Important)

3. **Q3: Optional tensors** - Helps with robustness

### Nice to Have (Optimization)

6. **Q6: cuBLAS settings** - Works without this, just slower

---

## How to Research

### Method 1: Inspect Actual GGUF File (BEST)

```bash
# Download gguf-py if not installed
pip install gguf

# Dump full file info
python3 -m gguf_dump /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf > qwen_dump.txt

# Extract specific info
grep "metadata" qwen_dump.txt
grep "tensor.*name" qwen_dump.txt
grep "tokenizer" qwen_dump.txt
```

**Advantage**: Gives exact answers for our specific model

### Method 2: Read llama.cpp Source (GOOD)

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Search for relevant code
grep -r "llm_load_tensors" .
grep -r "llama_vocab" .
grep -r "general.architecture" .
```

**Advantage**: Shows working implementation we can reference

### Method 3: Read GGUF Spec (OK)

https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

**Advantage**: Official spec, but may be incomplete

### Method 4: HuggingFace Model Cards (OK)

https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF

**Advantage**: Model-specific info, but may not have implementation details

---

## Expected Research Output

### Document: `GGUF_TENSOR_MAPPING.md`

```markdown
# GGUF Tensor Mapping for Qwen2.5-0.5B

## Metadata Keys
- general.architecture = "llama"
- llama.vocab_size = 151936
- llama.embedding_length = 896
- ...

## Tensor Names
- token_embd.weight -> GPTModelWeights::token_embeddings
- blk.0.attn_q.weight -> GPTLayerWeights::attn_q_weight
- ...

## Optional Tensors
- ❌ No bias terms (Llama-style)
- ❌ No learned position embeddings (uses RoPE)
- ✅ Separate Q/K/V (not fused)
```

### Document: `BPE_IMPLEMENTATION.md`

```markdown
# BPE Tokenizer Implementation Guide

## Algorithm
[Step-by-step pseudocode]

## Vocab Format
[How vocab is stored in GGUF]

## Merge Rules Format
[How merges are stored in GGUF]

## Special Tokens
[BOS, EOS, UNK handling]
```

---

## Timeline

**Before starting GT-051**: Research Q1, Q2 (2-3 hours)  
**Before starting GT-052**: Research Q3 (1 hour)  
**Before starting GT-053**: Research Q4, Q5 (3-4 hours)  
**Before starting GT-055**: Research Q6 (1 hour) - optional

**Total research time**: 6-8 hours

---

## Success Criteria

Research is complete when we can answer:

1. ✅ "What GGUF tensor name maps to `token_embeddings`?"
2. ✅ "What metadata key gives us `vocab_size`?"
3. ✅ "Does Qwen have bias terms?"
4. ✅ "How do I extract tokenizer vocab from GGUF?"
5. ✅ "What is the BPE merge algorithm pseudocode?"

**Then we can implement GT-051 to GT-057 without guessing.**

---

**Created by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Purpose**: Guide focused research before implementation
