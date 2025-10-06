# Team Bravo - Investigation Results

**Mission**: Compare our implementation with the `llama.cpp` reference to find the root cause of the repetitive token bug.

---

## Key Findings

The investigation revealed a critical difference in how our code and `llama.cpp` invoke the `cuBLAS` GEMM operation for the final projection to the vocabulary.

1.  **Transpose Flag Mismatch**: Our code uses `CUBLAS_OP_N` (no transpose) for the `lm_head` weight matrix. `llama.cpp` consistently uses `CUBLAS_OP_T` (transpose).
2.  **Leading Dimension Mismatch**: Our code passes the vocabulary size (`151936`) as the leading dimension (`lda`). `llama.cpp` passes the hidden dimension (`896`) as the leading dimension, which is correct for a transposed row-major matrix.

`llama.cpp` does **not** perform an explicit transpose when loading the tensor. It loads the row-major GGUF tensor and uses the `cuBLAS` parameters to correctly interpret it as a transposed matrix during the GEMM operation.

## Parameter Comparison Table

| Parameter | `llama.cpp` (`ggml-cuda.cu`) | Our Code (`qwen_transformer.cpp`) | Match? |
|---|---|---|---|
| `op_A` (lm_head) | `CUBLAS_OP_T` | `CUBLAS_OP_N` | ❌ No |
| `op_B` (hidden_state) | `CUBLAS_OP_N` | `CUBLAS_OP_N` | ✅ Yes |
| `m` | `dst->ne[0]` (vocab_size) | `config_.vocab_size` | ✅ Yes |
| `n` | `src1_ncols` (batch_size) | `batch_size` | ✅ Yes |
| `k` | `ne10` (hidden_dim) | `config_.hidden_dim` | ✅ Yes |
| `lda` | `ne00` (hidden_dim) | `config_.vocab_size` | ❌ No |
| `ldb` | `ne10` (hidden_dim) | `config_.hidden_dim` | ✅ Yes |
| `ldc` | `ldc` (vocab_size) | `config_.vocab_size` | ✅ Yes |

## Root Cause

The bug is caused by providing incorrect parameters to the `cublasGemmEx` function. Our code incorrectly describes the memory layout of the row-major `lm_head` matrix to the column-major `cuBLAS` library. This causes `cuBLAS` to read from incorrect memory locations, resulting in garbage values at specific logit positions.

## Proposed Fix

Modify the `cublasGemmEx` call in `cuda/src/transformer/qwen_transformer.cpp` to match the parameters used by `llama.cpp`.

1.  Change the first transpose operation from `CUBLAS_OP_N` to `CUBLAS_OP_T`.
2.  Change the leading dimension (`lda`) of the `lm_head` matrix from `config_.vocab_size` to `config_.hidden_dim`.

This will instruct `cuBLAS` to correctly interpret the row-major weight matrix, resolving the memory access issue.
