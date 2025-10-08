# Llama-2 Checkpoint Update Plan

**Created by:** TEAM-008  
**Date:** 2025-10-08  
**Purpose:** Comprehensive analysis of all checkpoints and required updates for Llama-2

---

## Executive Summary

After analyzing all 13 checkpoints (0-12), here's what needs to change for Llama-2:

**Keep as-is:** Checkpoint 0 (Foundation)  
**Update:** Checkpoints 1, 2, 6, 7, 8, 9, 10, 11, 12  
**Add new:** Checkpoint 1B (RoPE)  
**Remove:** Checkpoint 3 (KV Cache structure changes)  
**Major changes:** Checkpoints 4, 5 (Attention mechanism)

---

## Checkpoint-by-Checkpoint Analysis

### ✅ CHECKPOINT 0: Foundation Setup
**Status:** Keep as-is (95% reusable)  
**Changes needed:** Minor

**What stays the same:**
- HTTP server architecture
- Worker crates integration
- Project structure
- CLI arguments
- Startup flow

**What changes:**
- Model loading: GGUF instead of PyTorch
- File references: Update from GPT-2 to Llama-2
- Memory calculation: Different model size

**Action:** Update documentation references only

---

### 🔄 CHECKPOINT 1: RMSNorm (was LayerNorm)
**Status:** REPLACE - Already created  
**File:** `CHECKPOINT_01_RMS_NORM.md` ✅ Done

**Major changes:**
- Component: LayerNorm → RMSNorm
- Formula: `(x - mean) / sqrt(var + eps) * weight + bias` → `x / sqrt(mean(x²) + eps) * weight`
- No mean subtraction
- No bias term
- Simpler implementation

**Implementation:**
```rust
// OLD (GPT-2):
pub struct LayerNorm {
    weight: Array1<f32>,
    bias: Array1<f32>,    // ❌ Remove
    eps: f32,
}

// NEW (Llama-2):
pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,             // ✅ No bias
}
```

**Test changes:**
- Input shape: `[2, 1024]` → `[2, 4096]`
- Model: GPT-2 Medium → Llama-2 7B
- Prompt tokens: `[15496, 13]` → `[1, 15043]` (BOS + "Hello")

**Action:** ✅ Already created, ready to use

---

### 🔄 CHECKPOINT 2: QKV Projection
**Status:** UPDATE - Major changes

**What changes:**
- **Separate Q, K, V:** Llama-2 has 3 separate weight matrices, not combined
- **No bias:** All projections have no bias terms
- **Dimensions:** `[1024, 3072]` → `[4096, 4096]` per projection
- **Heads:** 16 → 32
- **Head dim:** 64 → 128

**OLD (GPT-2):**
```rust
// Single combined QKV projection
c_attn: Linear(1024, 3072)  // Combined Q, K, V
// Split after projection
```

**NEW (Llama-2):**
```rust
// Three separate projections
wq: Linear(4096, 4096)  // Query only, no bias
wk: Linear(4096, 4096)  // Key only, no bias
wv: Linear(4096, 4096)  // Value only, no bias
```

**Implementation changes:**
```rust
// OLD:
let qkv = x.dot(&self.c_attn_weight) + &self.c_attn_bias;
let qkv = qkv.reshape([batch, seq, 3, n_heads, head_dim]);
let (q, k, v) = split_qkv(qkv);

// NEW:
let q = x.dot(&self.wq);  // No bias
let k = x.dot(&self.wk);  // No bias
let v = x.dot(&self.wv);  // No bias
let q = q.reshape([batch, seq, n_heads, head_dim]);
let k = k.reshape([batch, seq, n_heads, head_dim]);
let v = v.reshape([batch, seq, n_heads, head_dim]);
```

**Action:** Create `CHECKPOINT_02_LLAMA2_QKV_PROJECTION.md`

---

### ➕ NEW: CHECKPOINT 1B: RoPE Application
**Status:** CREATE NEW - Critical for Llama-2

**Why new checkpoint:**
- RoPE is unique to Llama-2 (GPT-2 uses learned position embeddings)
- Applied between QKV projection and attention scores
- Critical to get right - position encoding affects all tokens
- Complex rotation math needs validation

**Location:** After QKV projection, before attention scores

**What to validate:**
```rust
fn apply_rope(
    q: &Array3<f32>,  // [batch, seq, n_heads, head_dim]
    k: &Array3<f32>,
    position: usize,
) -> (Array3<f32>, Array3<f32>) {
    // Rotate Q and K using position-dependent rotation
    // V is NOT rotated
}
```

**Test input:**
- Prompt: "Hello" → tokens `[1, 15043]`
- Position 0: BOS token
- Position 1: "Hello" token
- Verify rotation applied correctly

**Validation:**
- Q and K shapes unchanged
- V not modified
- Rotation formula correct: `cos(mθ) * x - sin(mθ) * y`
- Compare with llama.cpp checkpoint

**Action:** Create `CHECKPOINT_01B_ROPE_APPLICATION.md`

---

### 🔄 CHECKPOINT 3: KV Cache
**Status:** UPDATE - Structure changes

**What changes:**
- **Cache per layer:** Llama-2 has 32 layers (not 24)
- **Head dimensions:** 128 (not 64)
- **Num heads:** 32 (not 16)
- **Max context:** 4096 (not 2048)

**Structure stays similar:**
```rust
pub struct KVCache {
    k_cache: Vec<Array3<f32>>,  // 32 layers (was 24)
    v_cache: Vec<Array3<f32>>,  // 32 layers (was 24)
    current_position: usize,
    max_seq_len: usize,         // 4096 (was 2048)
}
```

**Test changes:**
- Layers: 24 → 32
- Heads: 16 → 32
- Head dim: 64 → 128
- Max seq: 2048 → 4096

**Action:** Update `CHECKPOINT_03_KV_CACHE.md` with new dimensions

---

### 🔄 CHECKPOINT 4: Attention Scores
**Status:** UPDATE - Minor changes

**What changes:**
- **Scale factor:** `sqrt(64)` = 8.0 → `sqrt(128)` = 11.31
- **Dimensions:** `[batch, 16, seq, seq]` → `[batch, 32, seq, seq]`
- **Causal mask:** Same concept, different size

**Formula stays same:**
```rust
scores = (Q @ K.T) / sqrt(head_dim)
// OLD: / 8.0
// NEW: / 11.31
```

**Action:** Update `CHECKPOINT_04_ATTENTION_SCORES.md` with new scale factor

---

### 🔄 CHECKPOINT 5: Attention Output
**Status:** UPDATE - Minor changes

**What changes:**
- **Output projection:** `[1024, 1024]` → `[4096, 4096]`
- **No bias:** Remove bias term
- **Heads:** 16 → 32

**OLD:**
```rust
output = output.dot(&self.c_proj_weight) + &self.c_proj_bias;
```

**NEW:**
```rust
output = output.dot(&self.wo);  // No bias
```

**Action:** Update `CHECKPOINT_05_ATTENTION_OUTPUT.md` - remove bias

---

### 🔄 CHECKPOINT 6: FFN Output
**Status:** MAJOR UPDATE - SwiGLU instead of GELU

**Critical changes:**
- **Activation:** GELU → SwiGLU
- **Structure:** 2 projections → 3 projections (gate, up, down)
- **No bias:** All projections have no bias
- **Dimensions:** Different expansion

**OLD (GPT-2):**
```rust
pub struct FFN {
    c_fc: Linear(1024, 4096),      // Up projection + bias
    c_proj: Linear(4096, 1024),    // Down projection + bias
}

fn forward(x) {
    hidden = x @ c_fc_weight + c_fc_bias
    hidden = gelu(hidden)
    output = hidden @ c_proj_weight + c_proj_bias
}
```

**NEW (Llama-2):**
```rust
pub struct SwiGLU {
    gate: Linear(4096, 11008),   // Gate projection, no bias
    up: Linear(4096, 11008),     // Up projection, no bias
    down: Linear(11008, 4096),   // Down projection, no bias
}

fn forward(x) {
    gate = silu(x @ gate_weight)     // SiLU activation
    up = x @ up_weight               // No activation
    hidden = gate * up               // Element-wise multiply
    output = hidden @ down_weight    // No bias
}
```

**SwiGLU formula:**
```rust
fn silu(x: f32) -> f32 {
    x * sigmoid(x)  // Also called Swish
}

fn swiglu(x, gate_w, up_w, down_w) {
    gate = silu(x @ gate_w)
    up = x @ up_w
    hidden = gate * up  // Gating mechanism
    output = hidden @ down_w
}
```

**Action:** Create `CHECKPOINT_06_LLAMA2_SWIGLU_FFN.md`

---

### 🔄 CHECKPOINT 7: First Block
**Status:** UPDATE - Architecture changes

**What changes:**
- **Normalization:** LayerNorm → RMSNorm (2x per block)
- **Attention:** Updated with RoPE
- **FFN:** GELU → SwiGLU
- **Layers:** 24 → 32
- **Dimensions:** 1024 → 4096

**Block structure (stays pre-norm):**
```rust
fn forward(x) {
    // Attention sublayer
    h = x + attention(rms_norm(x))  // RMSNorm, not LayerNorm
    
    // FFN sublayer
    h = h + swiglu_ffn(rms_norm(h))  // SwiGLU, not GELU
    
    h
}
```

**Test changes:**
- Input: `[1, 2, 1024]` → `[1, 2, 4096]`
- All components updated
- 32 layers total

**Action:** Update `CHECKPOINT_07_FIRST_BLOCK.md` with Llama-2 components

---

### 🔄 CHECKPOINT 8: Full Logits
**Status:** UPDATE - Major changes

**What changes:**
- **Layers:** 24 → 32
- **Final norm:** LayerNorm → RMSNorm
- **LM head:** `[1024, 50257]` → `[4096, 32000]`
- **Vocab size:** 50257 → 32000
- **No weight tying:** Llama-2 has separate output weights

**OLD (GPT-2):**
```rust
// After 24 layers
hidden = ln_f(hidden)                    // Final LayerNorm
logits = hidden @ wte.weight.T           // Weight tying with embeddings
// Shape: [batch, seq, 50257]
```

**NEW (Llama-2):**
```rust
// After 32 layers
hidden = rms_norm(hidden, output_norm_weight)  // Final RMSNorm
logits = hidden @ output_weight                // Separate output matrix
// Shape: [batch, seq, 32000]
```

**Test changes:**
- Vocab size: 50257 → 32000
- Layers: 24 → 32
- No weight tying check

**Action:** Update `CHECKPOINT_08_FULL_LOGITS.md` with new dimensions

---

### 🔄 CHECKPOINT 9: Selected Logits
**Status:** MINOR UPDATE

**What changes:**
- **Shape:** `[1, 2, 50257]` → `[1, 2, 32000]`
- **Vocab size:** 50257 → 32000

**Logic stays same:**
```rust
// Select last token logits
logits[:, -1, :]  // Same indexing
```

**Action:** Update dimensions in `CHECKPOINT_09_SELECTED_LOGITS.md`

---

### 🔄 CHECKPOINT 10: Argmax Sampling
**Status:** MINOR UPDATE

**What changes:**
- **Vocab range:** `[0, 50256]` → `[0, 31999]`
- **Token IDs:** Different vocabulary

**Logic stays same:**
```rust
if temperature < 1e-6 {
    token_id = logits.argmax(-1)
}
```

**Test changes:**
- Expected token IDs will differ
- Use llama.cpp reference for validation

**Action:** Update `CHECKPOINT_10_ARGMAX_SAMPLING.md` with new vocab size

---

### 🔄 CHECKPOINT 11: Softmax Probs
**Status:** MINOR UPDATE

**What changes:**
- **Distribution size:** 50257 → 32000
- **Token IDs:** Different vocabulary

**Logic stays same:**
```rust
probs = softmax(logits / temperature)
```

**Action:** Update `CHECKPOINT_11_SOFTMAX_PROBS.md` with new vocab size

---

### 🔄 CHECKPOINT 12: End-to-End
**Status:** MAJOR UPDATE - New test case

**What changes:**
- **Model:** GPT-2 Medium → Llama-2 7B
- **Prompt:** "Hello." → "Hello"
- **Tokens:** `[15496, 13]` → `[1, 15043]` (BOS + "Hello")
- **Expected output:** TBD (run llama.cpp to get reference)
- **Tokenizer:** tiktoken → SentencePiece

**Test case:**
```
Prompt: "Hello"
Tokens: [1, 15043]  # BOS + "Hello"
Model: Llama-2 7B Q8_0
Temperature: 0
Max tokens: 10
Expected: (run llama.cpp to determine)
```

**From our earlier test:**
```bash
./llama-cli -m llama-2-7b.Q8_0.gguf -p "Hello" -n 10 --temp 0.0
Output: ", I am interested in [1000"
```

**Action:** Create `CHECKPOINT_12_LLAMA2_END_TO_END.md` with new test case

---

## New Checkpoint Sequence for Llama-2

### Proposed Order:

```
0.  Foundation Setup (HTTP server)           ✅ Keep as-is
1.  RMSNorm Output                            🔄 REPLACE (done)
1B. RoPE Application                          ➕ NEW
2.  QKV Projection (separate Q, K, V)        🔄 UPDATE
3.  KV Cache (32 layers, 4096 context)       🔄 UPDATE
4.  Attention Scores (scale=11.31)           🔄 UPDATE
5.  Attention Output (no bias)               🔄 UPDATE
6.  SwiGLU FFN (3 projections)               🔄 MAJOR UPDATE
7.  First Block (RMSNorm + RoPE + SwiGLU)    🔄 UPDATE
8.  Full Logits (32 layers, 32K vocab)       🔄 UPDATE
9.  Selected Logits (32K vocab)              🔄 MINOR
10. Argmax Sampling (32K vocab)              🔄 MINOR
11. Softmax Probs (32K vocab)                🔄 MINOR
12. End-to-End (Llama-2 test case)           🔄 MAJOR UPDATE
```

---

## Summary of Changes

### Components to Replace:
1. **LayerNorm → RMSNorm** (simpler, no bias)
2. **GELU → SwiGLU** (3 projections, gating)
3. **Combined QKV → Separate Q, K, V** (no bias)
4. **Add RoPE** (new checkpoint between 1 and 2)

### Dimensions to Update:
- Hidden size: 1024 → 4096
- Layers: 24 → 32
- Heads: 16 → 32
- Head dim: 64 → 128
- FFN intermediate: 4096 → 11008
- Vocab: 50257 → 32000
- Context: 2048 → 4096

### Architecture Changes:
- No bias terms anywhere
- RoPE instead of learned positions
- SwiGLU instead of GELU
- Separate Q, K, V projections
- No weight tying (separate output matrix)

---

## Implementation Priority

### Week 2 (Core Components):
1. ✅ RMSNorm (Checkpoint 1) - Already done
2. ⏳ RoPE (Checkpoint 1B) - NEW
3. ⏳ Separate QKV (Checkpoint 2) - UPDATE
4. ⏳ SwiGLU (Checkpoint 6) - MAJOR UPDATE

### Week 3 (Full Model):
5. ⏳ KV Cache (Checkpoint 3) - UPDATE
6. ⏳ Attention Scores (Checkpoint 4) - UPDATE
7. ⏳ Attention Output (Checkpoint 5) - UPDATE
8. ⏳ First Block (Checkpoint 7) - UPDATE
9. ⏳ Full Logits (Checkpoint 8) - UPDATE

### Week 4 (Validation):
10. ⏳ Sampling (Checkpoints 9-11) - MINOR
11. ⏳ End-to-End (Checkpoint 12) - MAJOR UPDATE

---

## Files to Create/Update

### Create New:
1. `CHECKPOINT_01B_ROPE_APPLICATION.md`
2. `CHECKPOINT_02_LLAMA2_QKV_PROJECTION.md`
3. `CHECKPOINT_06_LLAMA2_SWIGLU_FFN.md`
4. `CHECKPOINT_12_LLAMA2_END_TO_END.md`

### Update Existing:
1. `CHECKPOINT_03_KV_CACHE.md` - dimensions
2. `CHECKPOINT_04_ATTENTION_SCORES.md` - scale factor
3. `CHECKPOINT_05_ATTENTION_OUTPUT.md` - remove bias
4. `CHECKPOINT_07_FIRST_BLOCK.md` - all components
5. `CHECKPOINT_08_FULL_LOGITS.md` - 32 layers, vocab
6. `CHECKPOINT_09_SELECTED_LOGITS.md` - vocab size
7. `CHECKPOINT_10_ARGMAX_SAMPLING.md` - vocab size
8. `CHECKPOINT_11_SOFTMAX_PROBS.md` - vocab size

### Keep As-Is:
1. `CHECKPOINT_00_FOUNDATION.md` - HTTP server (minor doc updates only)

---

## Testing Strategy Changes

### Reference Implementation:
- **Primary:** llama.cpp (not tinygrad)
- **Tool:** Team 006's checkpoint extractor
- **Model:** Llama-2 7B Q8_0

### Test Prompt:
- **OLD:** "Hello." → `[15496, 13]`
- **NEW:** "Hello" → `[1, 15043]` (BOS + "Hello")

### Expected Output:
```bash
# Run llama.cpp to get reference
./llama-cli -m llama-2-7b.Q8_0.gguf -p "Hello" -n 10 --temp 0.0 --seed 42
# Output: ", I am interested in [1000"
```

---

## Next Actions

### Immediate (Today):
1. ✅ Create this analysis document
2. ⏳ Create CHECKPOINT_01B_ROPE_APPLICATION.md
3. ⏳ Create CHECKPOINT_02_LLAMA2_QKV_PROJECTION.md
4. ⏳ Create CHECKPOINT_06_LLAMA2_SWIGLU_FFN.md

### Week 2:
1. Implement RMSNorm (use existing checkpoint)
2. Implement RoPE (new checkpoint)
3. Implement separate QKV (updated checkpoint)
4. Implement SwiGLU (new checkpoint)

### Week 3:
1. Update all attention checkpoints
2. Update block checkpoint
3. Update full model checkpoint

### Week 4:
1. Update sampling checkpoints
2. Create end-to-end test with llama.cpp reference
3. Validate all checkpoints pass

---

## Sign-off

**Created by:** TEAM-008 (Foundation Implementation)  
**Date:** 2025-10-08  
**Status:** Comprehensive checkpoint analysis complete

**Summary:**
- 13 checkpoints analyzed
- 4 new checkpoint docs needed
- 8 checkpoint docs need updates
- 1 checkpoint doc stays as-is
- Clear implementation path defined

---

*"Know what needs to change before you change it."*  
— TEAM-008, Foundation Implementation Division

**END ANALYSIS**
