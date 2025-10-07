# Reference Implementation Analysis
**Date:** 2025-10-07T23:02Z  
**Team:** SHAKESPEARE (Post-Integration Analysis)  
**Purpose:** Compare our implementation with candle, mistral.rs, and llama.cpp to identify potential bugs

---

## Executive Summary

After analyzing candle and mistral.rs Qwen2 implementations, I've identified **CRITICAL DIFFERENCES** that may explain our garbage output:

### 🚨 KEY FINDING: Embedding Table Dimensions

**Candle/Mistral.rs:**
```rust
let embed_tokens = candle_nn::embedding(
    cfg.vocab_size,      // 151936 (rows)
    cfg.hidden_size,     // 896 (columns)
    vb_m.pp("embed_tokens")
)?;
```
- Embedding table: **[vocab_size=151936, hidden_size=896]**
- Layout: **[151936 rows × 896 columns]**
- Token lookup: `embedding_table[token_id][:]` returns 896-dim vector

**Our Implementation:**
```cpp
// From qwen_weight_loader.cpp line 57:
names.push_back("token_embd.weight");

// From embedding.cu line 143:
half value = weight_matrix[token_id * hidden_dim + dim_idx];
// This assumes: [vocab_size, hidden_dim] row-major
```

**⚠️ CRITICAL QUESTION:** What are the ACTUAL dimensions of `token_embd.weight` in our GGUF file?

From VAN GOGH's investigation (TEAM_VAN_GOGH_CHRONICLE.md):
```
🔍 token_embd.weight dimensions: [896, 151936]
```

**🔥 SMOKING GUN FOUND:**
- **Candle expects:** [151936, 896] (vocab × hidden)
- **Our GGUF has:** [896, 151936] (hidden × vocab) **← TRANSPOSED!**
- **Our code assumes:** [151936, 896] (vocab × hidden)

**This means we're accessing the embedding table TRANSPOSED!**

When we do `weight_matrix[token_id * hidden_dim + dim_idx]`:
- We think we're getting embedding for token_id
- But we're actually getting a SLICE ACROSS ALL TOKENS at dimension dim_idx!

**Example:**
- Token ID = 100
- We compute: `offset = 100 * 896 + 0 = 89600`
- We think: "Get first element of token 100's embedding"
- Reality: "Get element from a completely wrong location"

This explains:
- ✅ Why output is garbage (wrong embeddings)
- ✅ Why it's consistent garbage (deterministic wrong lookup)
- ✅ Why llama.cpp works (handles transpose correctly)
- ✅ Why softmax/cuBLAS are correct (they operate on garbage data correctly)

---

## Detailed Comparison

### 1. Embedding Layer

#### Candle (qwen2.rs lines 266-278)
```rust
pub struct Model {
    embed_tokens: candle_nn::Embedding,  // ← Uses candle's embedding layer
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    // ...
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        // ...
    }
}
```

**Key observations:**
- Uses `candle_nn::embedding()` which handles layout automatically
- Tensor name: `"model.embed_tokens"`
- Dimensions: `[vocab_size, hidden_size]` = `[151936, 896]`

#### Mistral.rs (qwen2.rs lines 48-133)
```rust
struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    // ...
}
```

**Key observations:**
- Uses quantization-aware layers
- Handles FP16/FP32 dtype conversion automatically
- Biases are included in projections (line 77: `true` parameter)

#### Our Implementation (embedding.cu lines 72-145)
```cpp
__global__ void embedding_lookup_fp16(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    // ⚠️ CRITICAL LINE:
    half value = weight_matrix[token_id * hidden_dim + dim_idx];
    //                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                         Assumes [vocab_size, hidden_dim] layout
    embeddings[token_idx * hidden_dim + dim_idx] = value;
}
```

**Issues identified:**
1. **Dimension assumption mismatch:**
   - Code assumes: `[vocab_size, hidden_dim]`
   - GGUF has: `[hidden_dim, vocab_size]` (transposed!)
   - This causes completely wrong embedding lookups

2. **No scaling applied:**
   - Candle/mistral.rs may apply scaling (need to verify)
   - Our code does direct lookup with no post-processing

3. **Tensor name mismatch:**
   - Candle uses: `"model.embed_tokens"`
   - We use: `"token_embd.weight"`
   - GGUF might have different name → need to verify actual tensor name

---

### 2. RoPE (Rotary Position Embedding)

#### Candle (qwen2.rs lines 41-79)
```rust
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        // ...
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }
    
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}
```

**Key observations:**
- Precomputes sin/cos tables at initialization
- Uses `rope_theta` from config (default 10000.0 for Qwen2)
- Applies to Q and K, NOT to V
- Uses `seqlen_offset` for KV cache

#### Our Implementation
**Status:** Need to verify our RoPE implementation matches this exactly.

**Questions to answer:**
1. Do we precompute sin/cos or compute on-the-fly?
2. Do we use correct `rope_theta` value?
3. Do we apply only to Q/K (not V)?
4. Do we handle `seqlen_offset` correctly for autoregressive generation?

---

### 3. Attention Scaling

#### Candle (qwen2.rs line 196)
```rust
let scale = 1f64 / f64::sqrt(self.head_dim as f64);
let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
```

#### Mistral.rs (qwen2.rs line 129)
```rust
sdpa_params: SdpaParams {
    n_kv_groups: mistralrs_quant::compute_n_kv_groups(...),
    softcap: None,
    softmax_scale: 1.0 / (head_dim as f32).sqrt(),
    sliding_window: cfg.sliding_window,
},
```

**Key observations:**
- Both use `1.0 / sqrt(head_dim)` scaling
- Applied BEFORE softmax
- For Qwen2.5-0.5B: `head_dim = 896 / 14 = 64` → `scale = 1/8 = 0.125`

**Our implementation:**
Need to verify we apply this scaling correctly.

---

### 4. Attention Mask

#### Candle (qwen2.rs lines 297-324)
```rust
fn prepare_causal_attention_mask(
    &self,
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..tgt_len)
        .flat_map(|i| {
            (0..tgt_len).map(move |j| {
                if i < j || j + self.sliding_window < i {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    // ...
}
```

**Key observations:**
- Causal mask: `i < j` → `-inf` (can't attend to future)
- Sliding window: `j + window < i` → `-inf` (can't attend too far back)
- Applied as additive mask to attention scores
- Only created when `seq_len > 1` (not for single-token generation)

**Our implementation:**
Need to verify our causal mask matches this logic.

---

### 5. Model Forward Pass

#### Candle (qwen2.rs lines 340-362)
```rust
pub fn forward(
    &mut self,
    input_ids: &Tensor,
    seqlen_offset: usize,
    attn_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b_size, seq_len) = input_ids.dims2()?;
    let attention_mask: Option<Tensor> = match attn_mask {
        Some(mask) => Some(self.prepare_attention_mask(mask)?),
        None => {
            if seq_len <= 1 {
                None  // ← No mask for single token!
            } else {
                Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
            }
        }
    };
    let mut xs = self.embed_tokens.forward(input_ids)?;  // ← Embedding lookup
    for layer in self.layers.iter_mut() {
        xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
    }
    xs.apply(&self.norm)  // ← Output norm (RMSNorm)
}
```

**Key observations:**
1. **No attention mask for single token** (`seq_len <= 1`)
2. **Embedding → Layers → Output Norm** (standard flow)
3. **seqlen_offset** passed to every layer (for KV cache)

#### Our Implementation
Need to verify:
1. Do we skip attention mask for single-token generation?
2. Do we pass `seqlen_offset` correctly?
3. Is our forward pass order correct?

---

### 6. LM Head (Final Projection)

#### Candle (qwen2.rs lines 377-397)
```rust
impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            // ⚠️ IMPORTANT: Tied embeddings!
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset, None)?
            .narrow(1, seq_len - 1, 1)?  // ← Only take last token!
            .apply(&self.lm_head)
    }
}
```

**CRITICAL OBSERVATIONS:**

1. **Tied embeddings:** If `lm_head.weight` doesn't exist, reuse `embed_tokens`!
   - This means lm_head and embeddings share the SAME weight matrix
   - Config has `tie_word_embeddings: bool` flag

2. **Only process last token:** `.narrow(1, seq_len - 1, 1)`
   - For efficiency, only compute logits for the last token
   - Our implementation might compute for all tokens (wasteful)

3. **Tensor name:** `"lm_head.weight"` (not `"lm_head"`)

**Our implementation:**
Need to verify:
1. Do we check for tied embeddings?
2. Do we only compute logits for last token?
3. Do we use correct tensor name?

---

## Critical Questions for Round 3

### Priority 1: Embedding Table Transpose 🔥

**HYPOTHESIS:** Our embedding table is transposed in memory.

**Evidence:**
- VAN GOGH found: `token_embd.weight dimensions: [896, 151936]`
- Candle expects: `[151936, 896]`
- Our code assumes: `[151936, 896]` but accesses `[896, 151936]`

**Test:**
```cpp
// Current (WRONG):
half value = weight_matrix[token_id * hidden_dim + dim_idx];

// If transposed, should be:
half value = weight_matrix[dim_idx * vocab_size + token_id];
```

**Action for TEAM FROST:**
1. Dump first 10 values of embedding for token_id=0 from our engine
2. Dump first 10 values of embedding for token_id=0 from llama.cpp
3. Compare - if different, we have transpose bug
4. If transposed, fix indexing in `embedding.cu` line 143

### Priority 2: Tied Embeddings

**HYPOTHESIS:** lm_head and embeddings might share weights.

**Test:**
```bash
cd .test-models/qwen
python3 << EOF
from gguf import GGUFReader
reader = GGUFReader("qwen2.5-0.5b-instruct-fp16.gguf")
tensors = [t.name for t in reader.tensors]
print("Has lm_head.weight:", "lm_head.weight" in tensors)
print("Has output.weight:", "output.weight" in tensors)
print("All tensor names:", tensors)
EOF
```

**Action:**
If lm_head doesn't exist, we should reuse embedding weights (transposed).

### Priority 3: Attention Mask for Single Token

**HYPOTHESIS:** We might be applying attention mask when we shouldn't.

**Test:**
Check if we skip attention mask when `seq_len == 1` (autoregressive generation).

### Priority 4: RoPE Parameters

**HYPOTHESIS:** RoPE theta or frequency calculation might be wrong.

**Test:**
Compare our RoPE implementation with candle's formula:
```rust
inv_freq[i] = 1.0 / rope_theta^(i / dim)
```

---

## Recommended Actions

### Immediate (Round 3 Start)

1. **TEAM FROST: Fix Embedding Transpose**
   - Verify dimensions of `token_embd.weight` in GGUF
   - If `[896, 151936]`, change indexing to:
     ```cpp
     half value = weight_matrix[dim_idx * vocab_size + token_id];
     ```
   - Test with single token, compare with llama.cpp

2. **TEAM DICKINSON: Use Parity Logging**
   - Enable PICASSO's logging system
   - Compare embeddings layer-by-layer with llama.cpp
   - Find exact divergence point

### Secondary

3. **Check Tied Embeddings**
   - Verify if lm_head exists in GGUF
   - If not, reuse embedding weights

4. **Verify Attention Mask Logic**
   - Ensure no mask for single-token generation
   - Verify causal mask formula

5. **Verify RoPE Implementation**
   - Compare frequency calculation with candle
   - Verify theta value (should be 10000.0 for Qwen2)

---

## Reference File Locations

**Candle Qwen2:**
- `/home/vince/Projects/llama-orch/reference/candle/candle-transformers/src/models/qwen2.rs`

**Mistral.rs Qwen2:**
- `/home/vince/Projects/llama-orch/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs`

**Our Implementation:**
- Embedding: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/kernels/embedding.cu`
- Transformer: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
- Weight Loader: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/model/qwen_weight_loader.cpp`

---

## Conclusion

**Most Likely Bug:** Embedding table transpose issue.

**Evidence Strength:** 🔥🔥🔥 HIGH
- VAN GOGH confirmed dimensions are `[896, 151936]`
- Candle expects `[151936, 896]`
- Our code assumes `[151936, 896]` but data is `[896, 151936]`
- This would cause completely wrong embeddings → garbage output

**Recommended First Action:**
TEAM FROST should immediately test the transpose hypothesis by changing the embedding lookup indexing and comparing output with llama.cpp.

**If this fixes it:** We've found the root cause! 🎉

**If this doesn't fix it:** TEAM DICKINSON should use parity logging to find the exact divergence point.

---

**Analysis Complete:** 2025-10-07T23:02Z  
**Analyst:** TEAM SHAKESPEARE  
**Confidence:** HIGH (transpose bug very likely)  
**Next Team:** TEAM FROST (embedding inspection)
