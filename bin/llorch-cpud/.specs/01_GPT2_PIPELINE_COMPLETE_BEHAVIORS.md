# GPT-2 Inference Pipeline - Complete Behavioral Specification

**Reference:** `tinygrad/examples/gpt2.py`  
**Date:** 2025-10-08  
**Model:** GPT-2 base (124M parameters)  
**Configuration:** CPU, FP32, Temperature=0 (deterministic argmax)

This document specifies ALL behaviors in the GPT-2 inference pipeline in code flow order.

**CRITICAL CLARIFICATIONS FOR ENGINEERS:**

1. **Autoregressive Generation**: This is a sequential token generation process where each new token depends on all previously generated tokens. The model processes the entire prompt once, then generates one token at a time.

2. **KV Cache**: Key-Value cache is a performance optimization that stores previously computed attention keys and values to avoid recomputing them for past tokens during autoregressive generation.

3. **Causal Masking**: A triangular attention mask that prevents tokens from attending to future positions, ensuring the model can only use information from the current and past tokens.

4. **Pre-Norm Architecture**: Layer normalization is applied BEFORE the attention/FFN sublayers (not after), which differs from the original Transformer paper's post-norm design.

5. **Weight Tying**: The token embedding matrix and the final output projection (LM head) share the same weights to reduce parameters and improve generalization.

---

## RFC 2119 Keywords

- **MUST**: Absolute requirement
- **SHOULD**: Recommended but not absolute  
- **COULD**: Optional, implementation choice

---

## PHASE 1: Model Initialization and Weight Loading

### 1.1 Model Parameters (Lines 119-124)

**MUST Requirements:**
- The model MUST use exactly these parameters for GPT-2 base:
  - `n_layers`: 12 transformer blocks (sequential processing stages)
  - `n_heads`: 12 attention heads per layer (parallel attention computations)
  - `dim`: 768 (embedding dimension, also called `hidden_size` or `d_model` in literature)
  - `norm_eps`: 1e-5 (layer normalization epsilon for numerical stability)
  - `vocab_size`: 50257 tokens (BPE vocabulary size for GPT-2)
  - `max_seq_len`: 1024 positions (maximum sequence length, also called `context_length`)

**ENGINEER CLARIFICATION - Embedding Dimension:**
The `dim` parameter (768) represents the dimensionality of the continuous vector space where tokens are represented. Every token and position is mapped to a 768-dimensional vector. This is the fundamental "width" of the model.

**ENGINEER CLARIFICATION - Attention Heads:**
Multi-head attention splits the embedding dimension into `n_heads` parallel attention operations. Each head operates on a subspace of size `head_dim = dim / n_heads`. This allows the model to attend to different aspects of the input simultaneously.

**Derived Values:**
- `head_dim` MUST equal `dim / n_heads` = 768 / 12 = 64
  - **CRITICAL**: This division MUST be exact (no remainder). If `dim` is not divisible by `n_heads`, the architecture is invalid.
- `hidden_dim` (FFN intermediate) MUST equal `4 * dim` = 3072
  - **RATIONALE**: The 4x expansion is a standard transformer design pattern that provides capacity for the feedforward network to learn complex transformations.

**Tensor Shape Reference:**
- Token embedding weights: `[50257, 768]` (vocab_size × dim)
- Position embedding weights: `[1024, 768]` (max_seq_len × dim)

**POTENTIAL ISSUE - MAX_CONTEXT vs max_seq_len:**
The tinygrad code uses `MAX_CONTEXT` environment variable (default 128) for KV cache allocation, but `max_seq_len` defaults to 1024 for position embeddings. This is a DISCREPANCY that needs clarification:
- Position embeddings support up to 1024 tokens
- KV cache may be allocated for only 128 tokens (configurable via environment)
- **ENGINEER ACTION REQUIRED**: Decide whether to use a single configurable value or maintain separate limits for cache vs embeddings.

### 1.2 Weight Loading from PyTorch (Lines 132-142)

**MUST Requirements:**
- The implementation MUST load weights from PyTorch checkpoint format
- The implementation MUST transpose these specific Conv1D weights:
  - `attn.c_attn.weight` - QKV projection weights
  - `attn.c_proj.weight` - attention output projection
  - `mlp.c_fc.weight` - FFN up projection (also called `c_fc` for "fully connected")
  - `mlp.c_proj.weight` - FFN down projection

**ENGINEER CLARIFICATION - Conv1D in GPT-2:**
GPT-2's original implementation uses `Conv1D` layers, which is confusing naming. These are NOT 1D convolutions but rather linear (fully connected) layers with transposed weight storage:
- PyTorch Conv1D stores weights as `[out_features, in_features]` (transposed)
- Standard Linear/Dense layers expect `[in_features, out_features]`
- The transpose operation: `weights[k] = weights[k].T` converts the format

**CRITICAL - Weight Transpose Detection:**
The tinygrad code checks if key ENDS WITH the transpose patterns (line 136):
```python
if k.endswith(transposed):
```
This means the check is: `k.endswith(('attn.c_attn.weight', 'attn.c_proj.weight', ...))`

**ENGINEER WARNING**: This endswith check will match keys like:
- `h.0.attn.c_attn.weight` ✓
- `h.11.attn.c_proj.weight` ✓
- `transformer.h.5.mlp.c_fc.weight` ✓

But the actual PyTorch checkpoint keys have a prefix structure. Verify the exact key format in your checkpoint.

**Weight Tying Requirement:**
- The implementation MUST tie LM head weights with token embeddings:
  - `lm_head.weight` = `wte.weight` (same tensor reference, not copy)
  - **CRITICAL**: This must be a reference/pointer, not a deep copy. Changes to one must affect the other.
  - **MEMORY IMPLICATION**: This saves 50257 × 768 × 4 bytes ≈ 154MB in FP32

**Rationale:**
- PyTorch Conv1D stores weights as `[out_features, in_features]`
- Standard Linear layers expect `[in_features, out_features]`
- Transpose converts Conv1D format to Linear format
- Weight tying reduces total parameters and improves performance

**CONTRADICTION WITH RUST IMPLEMENTATIONS:**
Candle and mistral.rs typically use separate Linear layers without Conv1D legacy. If implementing from scratch in Rust:
- Option 1: Load PyTorch weights and transpose during loading
- Option 2: Use standard Linear layer format and transpose when loading PyTorch checkpoints
- **RECOMMENDATION**: Use Option 2 for cleaner code

### 1.3 Component Initialization Order (Lines 70-76)

**MUST Requirements:**
The implementation MUST initialize components in this exact order:
1. Token embedding layer (`wte`: Embedding(50257, 768))
2. Position embedding layer (`wpe`: Embedding(1024, 768))
3. All 12 transformer blocks (`h[0]` through `h[11]`)
4. Final layer normalization (`ln_f`: LayerNorm(768))
5. Language model head (`lm_head`: Linear(768, 50257, bias=False))

**Per-Block Initialization (Lines 59-63):**
Each TransformerBlock MUST initialize:
1. Attention sublayer (`attn`)
2. Feedforward sublayer (`mlp`)
3. First layer norm (`ln_1`: LayerNorm(768))
4. Second layer norm (`ln_2`: LayerNorm(768))

---

## PHASE 2: Input Processing and Embeddings

### 2.1 Tokenization (Lines 185, 129)

**MUST Requirements:**
- The implementation MUST use tiktoken with "gpt2" encoding
  - **ENGINEER NOTE**: tiktoken is OpenAI's BPE tokenizer library (Byte Pair Encoding)
  - **ALTERNATIVE FOR RUST**: Use `tokenizers` crate with GPT-2 tokenizer or `tiktoken-rs`
- The implementation MUST encode input string to list of token IDs
- Token IDs MUST be integers in range `[0, 50256]`

**SHOULD Requirements:**
- The implementation SHOULD support special token (endoftext marker in angle brackets)

**Example:**
- Input: "Hello world" → Token IDs: [15496, 995]
- Batch size 1, prompt length N → Shape: `[1, N]`

### 2.2 Forward Pass Entry (Lines 79-92, 196-200)

**MUST Requirements - First Token (Prompt Processing):**
- When `start_pos` = 0, the implementation MUST process entire prompt at once
- Input tokens MUST be converted to Tensor of shape `[batch_size, seq_len]`
- The implementation MUST create position indices from 0 to seq_len

**MUST Requirements - Subsequent Tokens (Autoregressive):**
- When `start_pos` > 0, the implementation MUST process single token
- Input MUST be single token ID (can be Variable or Tensor)
- The implementation MUST track current position in sequence

**Position Tracking:**
- `start_pos` variable MUST track how many tokens already generated
- First call: `start_pos` = 0 (processing prompt)
- Subsequent calls: `start_pos` = length of tokens generated so far

### 2.3 Token Embedding Lookup (Lines 83-86)

**MUST Requirements:**
- The implementation MUST lookup token embeddings from `wte` weight matrix
- For prompt (multiple tokens): Use standard embedding lookup
- For single token: COULD use weight shrinking optimization

**Tensor Shapes:**
- Input: `[batch_size, seq_len]` token IDs
- Output: `[batch_size, seq_len, 768]` token embeddings

**Operation:**
- Each token ID `t` → row `t` from embedding matrix `[50257, 768]`

### 2.4 Position Embedding Lookup (Lines 80, 89-90)

**MUST Requirements:**
- The implementation MUST create position indices tensor once and reuse
- Position indices MUST be `[0, 1, 2, ..., MAX_CONTEXT-1]` reshaped to `[1, MAX_CONTEXT]`
- The implementation MUST select positions based on `start_pos`

**Position Selection Logic:**
- If `start_pos` = 0 (prompt): Select positions `[0:seq_len]`
- If `start_pos` > 0 (generation): Select position `[start_pos:start_pos+1]`

**Tensor Shapes:**
- Selected positions: `[1, seq_len]`
- Position embeddings: `[1, seq_len, 768]`

### 2.5 Embedding Addition (Line 92)

**MUST Requirements:**
- The implementation MUST add token embeddings and position embeddings element-wise
- Result MUST be shape `[batch_size, seq_len, 768]`

**Operation:**
- `h = tok_emb + pos_emb`
- Broadcasting applies: `[batch_size, seq_len, 768]` + `[1, seq_len, 768]`

---

## PHASE 3: Attention Mask Creation

### 3.1 Causal Mask Generation (Line 96)

**MUST Requirements for Multi-Token (seq_len > 1):**
- The implementation MUST create causal attention mask
- Mask MUST be upper triangular matrix filled with `-inf`
- Mask shape MUST be `[1, 1, seq_len, start_pos+seq_len]`
- Diagonal offset MUST be `start_pos + 1`

**MUST Requirements for Single Token (seq_len = 1):**
- The implementation MUST set mask to None
- No masking needed for single token generation

**Mask Purpose:**
- Prevents attending to future positions
- Position i can only attend to positions 0 through i

**Example Mask (seq_len=3, start_pos=0):**
```
[[[[ 0.0, -inf, -inf],
   [ 0.0,  0.0, -inf],
   [ 0.0,  0.0,  0.0]]]]
```

---

## PHASE 4: Transformer Blocks (12 Layers)

### 4.1 Block Iteration (Line 98)

**MUST Requirements:**
- The implementation MUST process all 12 transformer blocks sequentially
- The implementation MUST pass output of block i as input to block i+1
- Each block MUST receive: hidden states `h`, `start_pos`, and `mask`

**Data Flow:**
- Input to block 0: `h` from embeddings `[batch_size, seq_len, 768]`
- Output from block 11: `h` ready for final layer norm

### 4.2 Per-Block Processing (Lines 65-67)

**MUST Requirements - Pre-Norm Architecture:**
1. Apply first layer norm to input
2. Pass through attention sublayer
3. Add residual connection (input + attention output)
4. Apply second layer norm
5. Pass through feedforward sublayer
6. Add residual connection (previous + FFN output)

**Formula:**
```
h = x + attention(layer_norm_1(x))
h = h + feedforward(layer_norm_2(h))
```

**Tensor Shapes Throughout Block:**
- Input: `[batch_size, seq_len, 768]`
- After each sublayer: `[batch_size, seq_len, 768]`
- Output: `[batch_size, seq_len, 768]`

---

## PHASE 5: Attention Mechanism

### 5.1 Layer Normalization (Line 62, 66)

**MUST Requirements:**
- The implementation MUST normalize across the embedding dimension (dim=768)
- The implementation MUST use epsilon = 1e-5 for numerical stability
- The implementation MUST apply learned scale and bias parameters

**Operation:**
```
mean = mean(x, dim=-1, keepdim=True)
var = var(x, dim=-1, keepdim=True)
normalized = (x - mean) / sqrt(var + eps)
output = normalized * weight + bias
```

**Tensor Shapes:**
- Input: `[batch_size, seq_len, 768]`
- Output: `[batch_size, seq_len, 768]`

### 5.2 QKV Projection (Lines 17, 29-30)

**MUST Requirements:**
- The implementation MUST use single linear layer for combined QKV projection
- Weight shape MUST be `[768, 2304]` (768 → 3*768)
- Bias MUST be included (shape `[2304]`)
- The implementation MUST reshape output to separate Q, K, V

**Operation:**
```
xqkv = Linear(768, 2304)(x)  # [batch, seq, 2304]
xqkv = reshape(xqkv, [batch, seq, 3, 12, 64])  # Split into Q,K,V and heads
xq = xqkv[:, :, 0, :, :]  # [batch, seq, 12, 64]
xk = xqkv[:, :, 1, :, :]  # [batch, seq, 12, 64]
xv = xqkv[:, :, 2, :, :]  # [batch, seq, 12, 64]
```

**Tensor Shapes:**
- Input `x`: `[batch_size, seq_len, 768]`
- After projection: `[batch_size, seq_len, 2304]`
- After reshape: `[batch_size, seq_len, 3, 12, 64]`
- Q, K, V each: `[batch_size, seq_len, 12, 64]`

### 5.3 KV Cache Management (Lines 34-45)

**MUST Requirements - Cache Initialization:**
- The implementation MUST create KV cache on first use
- Cache shape MUST be `[2, batch_size, MAX_CONTEXT, 12, 64]`
- First dimension: 0=keys, 1=values
- The implementation MUST initialize with zeros
- The implementation MUST realize/allocate the cache tensor

**MUST Requirements - Cache Update:**
- The implementation MUST update cache at positions `[start_pos:start_pos+seq_len]`
- The implementation MUST stack new K and V tensors
- The implementation MUST assign to cache slice
- The implementation MUST realize the assignment

**Cache Retrieval Logic:**
- If `start_pos` = 0 (prompt): Use current K, V directly (no cache retrieval)
- If `start_pos` > 0 (generation): Retrieve all cached keys/values up to current position

**Tensor Shapes:**
- Cache: `[2, batch_size, MAX_CONTEXT, 12, 64]`
- Update slice: `[2, batch_size, seq_len, 12, 64]`
- Retrieved keys: `[batch_size, start_pos+seq_len, 12, 64]`
- Retrieved values: `[batch_size, start_pos+seq_len, 12, 64]`

### 5.4 Attention Computation (Lines 47-48)

**MUST Requirements - Tensor Transposition:**
- The implementation MUST transpose Q, K, V from `[batch, seq, heads, head_dim]` to `[batch, heads, seq, head_dim]`
- This enables parallel computation across attention heads

**MUST Requirements - Scaled Dot-Product Attention:**
- The implementation MUST compute: `softmax((Q @ K.T) / sqrt(head_dim) + mask) @ V`
- Scale factor MUST be `sqrt(64)` = 8.0
- The implementation MUST apply causal mask (if provided)
- The implementation MUST apply softmax across key dimension

**Detailed Steps:**
1. Transpose: Q, K, V → `[batch, 12, seq, 64]`
2. Compute scores: `Q @ K.transpose(-2, -1)` → `[batch, 12, seq_q, seq_k]`
3. Scale: `scores / 8.0`
4. Add mask: `scores + mask` (broadcasts to same shape)
5. Softmax: `softmax(scores, dim=-1)`
6. Apply to values: `attention_weights @ V` → `[batch, 12, seq, 64]`

**Tensor Shapes:**
- Q: `[batch, 12, seq_q, 64]`
- K: `[batch, 12, seq_k, 64]`
- V: `[batch, 12, seq_k, 64]`
- Attention scores: `[batch, 12, seq_q, seq_k]`
- Output: `[batch, 12, seq_q, 64]`

### 5.5 Output Projection (Lines 18, 48)

**MUST Requirements:**
- The implementation MUST transpose attention output back to `[batch, seq, heads, head_dim]`
- The implementation MUST reshape to `[batch, seq, 768]` (merge heads)
- The implementation MUST apply output projection linear layer
- Output projection: Linear(768, 768) with bias

**Operation:**
```
attn_out = transpose(attn_out, [batch, seq, 12, 64])
attn_out = reshape(attn_out, [batch, seq, 768])
output = Linear(768, 768)(attn_out)
```

**Tensor Shapes:**
- Input: `[batch, 12, seq, 64]`
- After transpose: `[batch, seq, 12, 64]`
- After reshape: `[batch, seq, 768]`
- After projection: `[batch, seq, 768]`

---

## PHASE 6: Feedforward Network

### 6.1 FFN Structure (Lines 51-56)

**MUST Requirements:**
- The implementation MUST use two-layer MLP with GELU activation
- First layer (up projection): Linear(768, 3072) with bias
- Activation: GELU (Gaussian Error Linear Unit)
- Second layer (down projection): Linear(3072, 768) with bias

**Operation:**
```
hidden = Linear(768, 3072)(x)
activated = GELU(hidden)
output = Linear(3072, 768)(activated)
```

**Tensor Shapes:**
- Input: `[batch, seq, 768]`
- After up projection: `[batch, seq, 3072]`
- After GELU: `[batch, seq, 3072]`
- After down projection: `[batch, seq, 768]`

### 6.2 GELU Activation (Line 56)

**MUST Requirements:**
- The implementation MUST use GELU activation function
- GELU formula: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- Or use error function approximation if available

**Properties:**
- Smooth, non-monotonic activation
- Allows small negative values (unlike ReLU)
- Used in GPT-2, BERT, and other transformers

---

## PHASE 7: Final Layer Norm and LM Head

### 7.1 Final Layer Normalization (Lines 75, 100)

**MUST Requirements:**
- The implementation MUST apply layer normalization after all transformer blocks
- Same operation as per-block layer norms
- Epsilon = 1e-5
- Learned scale and bias parameters

**Tensor Shapes:**
- Input: `[batch, seq, 768]`
- Output: `[batch, seq, 768]`

### 7.2 Language Model Head (Lines 76, 100)

**MUST Requirements:**
- The implementation MUST apply linear projection to vocabulary size
- Weight shape: `[768, 50257]`
- NO bias term (bias=False)
- Weights MUST be tied with token embeddings

**Operation:**
```
logits = Linear(768, 50257, bias=False)(h)
```

**Tensor Shapes:**
- Input: `[batch, seq, 768]`
- Output: `[batch, seq, 50257]` (logits for each position)

### 7.3 Logit Selection (Lines 102-106)

**MUST Requirements:**
- The implementation MUST select logits for LAST token only
- If sequence length > 0: Take `logits[:, -1, :]`
- If sequence length = 0 (edge case): Return ones tensor

**Tensor Shapes:**
- Full logits: `[batch, seq, 50257]`
- Selected logits: `[batch, 50257]`

---

## PHASE 8: Sampling and Token Generation

### 8.1 Temperature=0 Sampling (Lines 108-109)

**MUST Requirements for temperature < 1e-6:**
- The implementation MUST use argmax (greedy decoding)
- The implementation MUST select token with highest logit value
- This produces deterministic output

**Operation:**
```
if temperature < 1e-6:
    next_token = argmax(logits, dim=-1)
```

**Tensor Shapes:**
- Input logits: `[batch, 50257]`
- Output token: `[batch]` (single token ID per batch item)

### 8.2 Temperature>0 Sampling (Lines 110-111)

**SHOULD Requirements for temperature >= 1e-6:**
- The implementation SHOULD divide logits by temperature
- The implementation SHOULD apply softmax to get probabilities
- The implementation SHOULD sample from multinomial distribution

**Operation:**
```
probs = softmax(logits / temperature, dim=-1)
next_token = multinomial(probs, num_samples=1)
```

**Effect of Temperature:**
- temperature < 1.0: Sharper distribution (more confident)
- temperature = 1.0: Unchanged distribution
- temperature > 1.0: Flatter distribution (more random)

### 8.3 Output Finalization (Line 112)

**MUST Requirements:**
- The implementation MUST flatten output to 1D
- The implementation MUST realize/execute the computation
- Output MUST be list of token IDs

**Tensor Shapes:**
- Before flatten: `[batch]` or `[batch, 1]`
- After flatten: `[batch]`

---

## PHASE 9: Autoregressive Generation Loop

### 9.1 Generation Loop (Lines 188-203)

**MUST Requirements:**
- The implementation MUST maintain list of generated tokens per batch item
- The implementation MUST update `start_pos` after each iteration
- The implementation MUST append new token to token list
- The implementation MUST continue until max_length reached

**Loop Structure:**
```
start_pos = 0
toks = [prompt_tokens for each batch item]

for step in range(max_length):
    # Get tokens to process
    if start_pos == 0:
        tokens = entire_prompt
    else:
        tokens = last_token_only
    
    # Run model
    next_token = model(tokens, start_pos, temperature)
    
    # Update state
    toks.append(next_token)
    start_pos = len(toks)
```

### 9.2 Token Decoding (Line 208)

**MUST Requirements:**
- The implementation MUST decode token IDs back to string
- The implementation MUST use same tokenizer (tiktoken gpt2)
- The implementation MUST handle batch of sequences

**Operation:**
```
output_text = tokenizer.decode(token_ids)
```

---

## PHASE 10: Validation and Testing

### 10.1 Deterministic Output Validation (Lines 245-254)

**SHOULD Requirements:**
- The implementation SHOULD validate output for known inputs when temperature=0
- The implementation SHOULD compare against expected outputs
- The implementation SHOULD use GPT-2 medium for validation

**Known Test Cases:**
- Prompt: "What is the answer to life, the universe, and everything?"
- Expected (10 tokens, temp=0, gpt2-medium): "What is the answer to life, the universe, and everything?\n\nThe answer is that we are all one"

---

## Summary: Complete Data Flow

### Single Forward Pass (Temperature=0, Prompt="Hello")

1. **Tokenize**: "Hello" → `[15496]`
2. **Embed**: `[15496]` → `[1, 1, 768]`
3. **Add Position**: `[1, 1, 768]` + pos_emb → `[1, 1, 768]`
4. **Mask**: None (single token)
5. **12 Transformer Blocks**:
   - Layer norm → Attention → Residual
   - Layer norm → FFN → Residual
6. **Final Layer Norm**: `[1, 1, 768]`
7. **LM Head**: `[1, 1, 768]` → `[1, 1, 50257]`
8. **Select Last**: `[1, 50257]`
9. **Argmax**: `[1]` (e.g., token 995 = " world")
10. **Decode**: `995` → " world"

### Key Tensor Shapes Reference

| Stage | Tensor | Shape |
|-------|--------|-------|
| Input tokens | tokens | `[batch, seq]` |
| Token embeddings | tok_emb | `[batch, seq, 768]` |
| Position embeddings | pos_emb | `[1, seq, 768]` |
| Combined embeddings | h | `[batch, seq, 768]` |
| Attention mask | mask | `[1, 1, seq, seq]` or None |
| QKV projection | xqkv | `[batch, seq, 2304]` |
| Q, K, V (each) | xq/xk/xv | `[batch, seq, 12, 64]` |
| KV cache | cache_kv | `[2, batch, MAX_CONTEXT, 12, 64]` |
| Attention output | attn_out | `[batch, seq, 768]` |
| FFN hidden | ffn_hidden | `[batch, seq, 3072]` |
| FFN output | ffn_out | `[batch, seq, 768]` |
| Final hidden | h_final | `[batch, seq, 768]` |
| Logits | logits | `[batch, seq, 50257]` |
| Selected logits | logits_last | `[batch, 50257]` |
| Next token | next_token | `[batch]` |

---

## End of Specification

This document captures ALL behaviors in the tinygrad GPT-2 inference pipeline for temperature=0, CPU, FP32 execution.
