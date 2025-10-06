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