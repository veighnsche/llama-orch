````markdown name=LLAMACPP_VS_OURS_COMPARISON.md
# llama.cpp vs Our Implementation

Date: 2025-10-05

This document summarizes how llama.cpp implements GQA attention (including KV cache and RoPE) and contrasts it with our Qwen2.5 implementation to surface likely divergences.

## Quick glossary

- n_head = total query heads (e.g., 14)
- n_head_kv = key/value heads (e.g., 2)
- group_size = n_head / n_head_kv (e.g., 7)
- d_head = head_dim (e.g., 64)
- seq_len/cache_len = tokens currently in cache for a sequence

---

## Attention flow comparison

### llama.cpp (conceptual, from llm_graph_context::build_attn path)

1. Pre-norm (RMSNorm/LayerNorm) on hidden state.
2. Linear projections:
   - Q = Wq · x (+ bq if present)
   - K = Wk · x (+ bk if present)
   - V = Wv · x (+ bv if present)
3. Reshape for heads:
   - Q: [batch, seq, n_head, d_head] or [d_head, n_head, seq] depending on backend
   - K/V: [batch, seq, n_head_kv, d_head] or [d_head, n_head_kv, seq]
4. RoPE:
   - Apply rotary embeddings to Q and K (not V)
   - RoPE params (theta/freq_base, scaling) are read from GGUF via keys like "%s.rope.*"
   - Applied after projections and reshape, before attention
5. KV cache write (prefill/first token):
   - Write K and V at position pos (per layer) into cache
   - Layout uses per-layer, per-kv-head, per-position, per-d_head strides
6. Decode step (pos > 0):
   - Read all prior K,V for the relevant kv head(s) from cache
7. GQA grouping:
   - Map each q_head to a kv_head:
     - kv_head = q_head // group_size
     - Extend or index K,V so each q_head attends to its kv_head
8. Attention scores:
   - S = (Q · K^T) * (1 / sqrt(d_head))
   - Apply mask (causal, sinks, windows as applicable)
   - Softmax in FP32 for numerical stability
9. Context:
   - O = softmax(S) · V
10. Output projection and residual:
   - O = Wo · O (+ bo if present)
   - x = x + O (residual connection)
   - Proceed to FFN

Notes:
- Data types: matmuls and softmax reductions are accumulated in FP32; storage may be FP16/BF16 on device.
- QKV bias: handled after matmul with corresponding bias tensors if model has them.
- Shapes are consistent across backends via ggml reshape/view; GEMM stride/transpose flags are chosen per layout.

### Our implementation (as summarized)

1. Pre-norm: likely correct (RMSNorm for Qwen2.5).
2. Q/K/V projections: implemented with bias addition.
3. Reshape: Q heads = 14, KV heads = 2, d_head = 64.
4. RoPE: applied (but not fully verified).
5. KV cache: per layer indexing fixed; uses `layer_idx * context_length * num_kv_heads * head_dim` offset.
6. Decode: reads all cached K for attention; writes K,V at pos.
7. GQA: implemented grouping 14→2.
8. Attention: Q·K^T, scaled, softmax, V weighted sum.
9. Output proj: GEMM in-place suspicion.
10. Residual and FFN: present.

---

## Differences and likely pitfalls

- RoPE parameters/order:
  - llama.cpp applies RoPE to Q and K after projection/reshape, before cache read/write and attention.
  - Ensure our RoPE uses the correct theta/freq_base for Qwen2.5 (commonly 1e6) and rotates the correct dimensions (often d_head, sometimes 2D/M-RoPE only in VL variants).
- KV cache layout:
  - llama.cpp caches K and V with strides covering [batch, kv_head, position, d_head].
  - Our formula `layer_idx * context_length * num_kv_heads * head_dim` misses at least:
    - per-batch stride
    - per-layer base plus per-kv-head channel stride
    - element stride for d_head
  - We must confirm full indexing: offset(layer, batch, kv_head, pos, d) = base(layer) + batch_off + kv_head_off + pos_off + d
- GQA mapping:
  - llama.cpp conceptually uses kv_index = q_head // (n_head / n_head_kv).
  - Our mapping must use the same integer division; verify we don’t round incorrectly for non-perfect divisibility (Qwen2.5 usually divides evenly).
- Scaling and dtype:
  - llama.cpp scales attn scores by 1/sqrt(d_head) and accumulates softmax in FP32.
  - If we accumulate in FP16 or miss scaling, output can become garbage.
- Softmax stability:
  - llama.cpp uses max-subtraction and FP32; ensure we do both (subtract per-row max; do exp/sum in FP32).
- Output projection:
  - llama.cpp uses separate output buffer for Wo matmul; avoid in-place overwrite of inputs to GEMM.
- Bias handling:
  - llama.cpp adds biases where present. Ensure our bias tensors match model dims (Q/K/V/WO), and broadcasting axis is correct.
- Shapes and transpose flags:
  - llama.cpp relies on ggml to do correct transposes. Our GEMM transpose flags must match actual memory layout. Any mismatch silently corrupts math.

---

## Code snippets (from llama.cpp repo)

### Where attention is built (high level)
- llm_graph_context::build_attn ultimately calls a multi-head attention builder (MHA) that:
  - takes Q,K,V, applies masks, computes softmax in FP32, then output projection.

```cpp
// src/llama-graph.cpp (signatures; implementation constructs the MHA subgraph)
ggml_tensor * llm_graph_context::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        float kq_scale,
        int il) const;
```

### Q/K/V + RoPE order (vision example, same ordering principle)
```cpp
// tools/mtmd/clip.cpp (Qwen2VL uses M-RoPE; text models use standard RoPE but placement is analogous)
// 1) Q/K/V = matmul + bias
Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

// 2) reshape to [d_head, n_head, n_pos]
Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

// 3) apply RoPE to Q,K (here multi-rope variant)
// (text models use ggml_rope / ggml_rope_ext)
Qcur = ggml_rope_multi(...);
Kcur = ggml_rope_multi(...);

// 4) build attention, then Wo and residual
cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
cur = ggml_add(ctx0, cur, inpL); // residual
```

---

## Identified likely issues in our code

- KV cache indexing omits batch/head strides → wrong keys/values retrieved.
- Softmax run in FP16 or missing max-subtraction → numeric blowup/underflow.
- RoPE frequency base/dims mismatch for Qwen2.5 (theta often 1e6; rotate full d_head).
- GEMM transpose flags or in-place overwrite in Wo projection → corrupt outputs.
- GQA mapping off-by-one or group_size computed incorrectly.

---
````

````markdown name=ATTENTION_BUGS_FOUND.md
# Attention Bugs Found (Hypotheses from llama.cpp comparison)

Date: 2025-10-05

This lists concrete bug hypotheses that plausibly explain the “garbage” generations.

## Critical (likely to cause nonsense)

1) KV cache indexing/strides incorrect
- Symptom: Attention reads keys/values for the wrong head/position.
- Our indexing formula lacks batch stride and may not include per-head stride:
  - Used: layer_idx * ctx_len * n_kv_heads * d_head
  - Missing: + batch_idx * (… per-batch stride …) + kv_head * (ctx_len * d_head) + pos * d_head + d
- Action: Define and use a single canonical offset(layer,batch,kv_head,pos,d).

2) Softmax not numerically stable / wrong dtype
- Symptom: Exploding/vanishing attention; garbage tokens.
- Risks:
  - Not subtracting per-row max before exp
  - Doing accumulation in FP16
  - Missing 1/sqrt(d_head) scaling on QK^T
- Action: Enforce FP32 for logits and softmax sum; always subtract max; multiply by 1/sqrt(d_head).

3) RoPE misapplied
- Symptom: Positional information corrupt → incoherent text.
- Risks:
  - Wrong theta (Qwen2.5 typically uses 1e6)
  - Rotating wrong slice size (should be d_head unless model-specific)
  - Applying after cache write or inconsistently between prefill/decode
- Action: Apply RoPE to Q and K, same dims/params, before attention and consistent across prefill/decode.

4) Wo projection in-place overwrite
- Symptom: Output corruption after attention even when scores look fine.
- Risk: Using the same buffer for input and output in GEMM.
- Action: Ensure GEMM output buffer != input buffer; then do residual add.

## High

5) GQA head mapping arithmetic bug
- Symptom: Some q_heads look at wrong kv_head.
- Risk: Using modulo instead of integer division; group_size miscomputed.
- Action: kv = q // (n_head / n_head_kv); validate for 14→2 => group_size=7.

6) QKV bias broadcast mismatch
- Symptom: Subtle drift → garbage after deep layers.
- Risk: Bias shape/dtype/broadcast axis wrong.
- Action: Validate bias lengths == proj_out_dim; add tests for a small tensor.

7) GEMM transpose/lda/ldb mismatch
- Symptom: Wrong results with no crash.
- Risk: Interpreting row-major as column-major or wrong leading dimensions.
- Action: Print shapes and flags for all GEMMs; compare to a PyTorch reference.

## Medium

8) Cache length off-by-one
- Symptom: Wrong positions attended.
- Risk: Using pos as cache_len incorrectly; first token paths mismatched.
- Action: For decode at position pos, K/V read window should be [0..pos-1]; write at index pos-1 or pos consistently with implementation.

9) RMSNorm epsilon mismatch
- Symptom: Distribution shifts → unstable attention.
- Risk: Using wrong epsilon or wrong norm type (LayerNorm vs RMSNorm).
- Action: Read from GGUF keys and confirm.

10) Masking logic
- Symptom: Past/future leakage or over-masking.
- Action: Re-check causal mask for prefill vs decode; ensure no -inf overflows in FP16.

---
````

````markdown name=ATTENTION_FIX_PLAN.md
# Attention Fix Plan (Prioritized)

Date: 2025-10-05

## P0: Correctness blockers

1) Implement canonical KV cache indexing (per layer, batch, kv_head, pos, dim)
- Define strides:
  - per_layer = n_seqs * n_kv_heads * n_ctx * d_head
  - per_batch = n_kv_heads * n_ctx * d_head
  - per_kvh   = n_ctx * d_head
  - per_pos   = d_head
  - per_dim   = 1
- Address:
  - off = layer*per_layer + batch*per_batch + kvh*per_kvh + pos*per_pos + d
- Update both write (prefill/decode) and read (decode) paths.

2) Enforce stable softmax in FP32
- Compute logits in FP32 (cast Q,K to FP32 for the dot; keep accumulators FP32).
- Subtract row-wise max before exp.
- Sum in FP32; divide (or multiply by reciprocal) in FP32.
- Only then cast results (e.g., FP16) if needed.

3) Validate RoPE application for Qwen2.5
- Use theta=freq_base=1e6 (unless overridden by GGUF keys).
- Rotate Q and K over d_head (common for Qwen2.x text).
- Apply after projection/reshape, before attention and cache use.
- Prefill and decode must share identical RoPE logic.

4) Fix Wo projection destination buffers
- Ensure GEMM output buffer is distinct from input.
- After Wo, perform residual: out = residual + Wo_out.

## P1: Very likely issues

5) Re-check GQA head mapping
- group_size = n_head / n_head_kv (assert divisibility).
- kv_head = q_head / group_size.
- Unit-test: For n_head=14, n_head_kv=2 → mapping 0..6→0, 7..13→1.

6) Confirm QKV bias shapes and broadcasting
- Validate bias lengths equal output_dim of projection.
- Apply addition on correct axis (per channel).

7) GEMM shapes and transpose flags
- Log shapes and flags at runtime for Q/K/V/WO:
  - (m x k) · (k x n) -> (m x n)
- Compare one layer’s Q,K,V,Wo results vs PyTorch for a synthetic input.

## P2: Diagnostics and cross-checks

8) Layer-by-layer snapshots
- Dump first 16 values after: embed, norm1, Q, K, V (post-bias), Q_rope, K_rope, attn_scores(max/mean/std), softmax row sums, context head0 first 8, Wo_out first 8.
- Do this for layer 0 and layer 1.

9) Single-layer A/B test
- Run 1-layer-only model in our code and in llama.cpp/transformers.
- Compare tensors at each step to locate first divergence.

10) Masking sanity
- For decode at pos p, ensure mask allows [0..p] (or [0..p-1] depending on index semantics) and forbids >p.
- Check that mask values use large negative sentinels in FP32 (e.g., -1e9), not -inf in FP16.

## Small code helpers

GQA mapping:
```cpp
inline int kv_head_for_q(int q_head, int n_head, int n_head_kv) {
    const int group = n_head / n_head_kv; // assert n_head % n_head_kv == 0
    return q_head / group;
}
```

Scaling factor:
```cpp
const float kq_scale = 1.0f / sqrtf((float) d_head); // e.g., d_head=64 -> 0.125f
```

Softmax (row-wise), FP32:
```cpp
// logits: float* L, length = seq_len
float m = -FLT_MAX;
for (int i = 0; i < seq_len; ++i) m = fmaxf(m, L[i]);
float s = 0.f;
for (int i = 0; i < seq_len; ++i) { L[i] = expf(L[i] - m); s += L[i]; }
const float invs = 1.0f / s;
for (int i = 0; i < seq_len; ++i) L[i] *= invs;
```

## Validation steps (fast)

- Compare one attention step (layer 0, pos=10, head 0) against PyTorch:
  - Inputs: fixed seed, small prompt
  - Dump Q,K,V (post-bias + RoPE), softmax row, context, Wo_out
  - Expect near-equal (atol 1e-3..1e-4)

- If mismatch:
  - Check RoPE theta/dims
  - Check cache read indices
  - Check GEMM transpose flags
  - Check scaling and softmax

---
````

If you want, I can:
- add a small CUDA unit-test kernel to dump one head’s attention scores and softmax row sums,
- write a Python harness to cross-validate one layer vs HuggingFace,
- or adapt these files into your repo structure under .plan/ or docs/.