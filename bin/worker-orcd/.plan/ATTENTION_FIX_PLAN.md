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