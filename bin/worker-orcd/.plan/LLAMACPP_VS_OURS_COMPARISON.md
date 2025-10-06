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