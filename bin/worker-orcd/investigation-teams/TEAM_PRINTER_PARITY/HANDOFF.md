# TEAM PRINTER — Handoff Document

**Team:** TEAM PRINTER (Utility Team)  
**Mission:** Parity Data Sweep — No bug fixing, data collection only  
**Date:** 2025-10-07T01:33:09Z  
**Status:** 🟢 INFRASTRUCTURE COMPLETE — Ready for execution

---

## Executive Summary

TEAM PRINTER has completed the infrastructure for collecting side-by-side parity data between our CUDA/Rust engine and llama.cpp. The checkpoint logging system is integrated, tested, and ready to run.

**Key Achievement:** Non-invasive logging infrastructure that captures intermediate tensors without modifying computation logic.

---

## What We Built

### 1. Checkpoint Logger (C++)

**Files:**
- `cuda/src/utils/checkpoint_logger.h` — Header with inline logging functions
- `cuda/src/utils/checkpoint_logger.cpp` — Implementation with NPZ export

**Features:**
- FP16 → FP32 conversion for precision
- Environment variable control (`PRINTER_CHECKPOINT_LOGGING=1`)
- Token-based filtering (`PRINTER_TOKEN_LIMIT=2`)
- Binary + JSON manifest output
- Min/max/mean statistics logged to stderr

**Integration:**
- Added to `cuda/CMakeLists.txt` (line 58)
- Initialized in `QwenTransformer` constructor (line 302)
- Finalized in `QwenTransformer` destructor (line 311)

### 2. Python Utilities

**Files:**
- `convert_to_npz.py` — Convert binary checkpoints to numpy .npz format
- `collect_parity_data.py` — Compare .npz files and generate diff report

**Features:**
- Automatic manifest parsing
- L2/L∞ norm computation
- First mismatch detection (threshold: 1e-5)
- Markdown report generation

### 3. Runner Scripts

**Files:**
- `run_our_engine.sh` — Execute our engine with checkpoint logging
- `run_llamacpp.sh` — Execute llama.cpp with same config

**Features:**
- Environment variable setup
- Foreground execution only
- Log capture to files
- Error checking

### 4. Documentation

**Files:**
- `README.md` — Comprehensive guide with practical strategy
- `GO_NO_GO_CHECKLIST.md` — Pre-flight verification checklist
- `printer_meta.json` — Environment and test configuration metadata
- `HANDOFF.md` — This document

---

## How to Use

### Quick Start (3 Commands)

```bash
# 1. Build with checkpoint logger
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo clean && cargo build --release --features cuda

# 2. Run our engine
./investigation-teams/TEAM_PRINTER_PARITY/run_our_engine.sh

# 3. Run llama.cpp
./investigation-teams/TEAM_PRINTER_PARITY/run_llamacpp.sh
```

### Detailed Workflow

See `GO_NO_GO_CHECKLIST.md` for step-by-step instructions.

---

## What We Learned

### From Previous Teams

**BATTLESHIP findings:**
- Q projection has spikes at indices 95, 126
- Manual FP32 calculation gives correct values (±0.08)
- cuBLAS gives wrong values (±16) at same indices
- Spikes persist with both CUBLAS_OP_T and CUBLAS_OP_N
- Spikes persist with both FAST_16F and 32F compute types

**Key Question:** Does llama.cpp also see these spikes?
- If YES → Bug is in model file or expected behavior
- If NO → Bug is in our cuBLAS usage or weight loading

### Existing Logging Infrastructure

The codebase already has extensive logging from previous teams:
- `[TEAM SENTINEL]` — Layer 0 input/output (first 10 floats)
- `[TEAM ORION]` — Min/max/mean at checkpoints (first 16 floats)
- `[RACE CAR]` — FFN checkpoints
- `[TOP HAT]` — Q weight column stats
- `[BATTLESHIP]` — Attention projection audit

**Strategy:** Use existing logs for initial comparison, add full checkpoint logging only if needed.

---

## Pragmatic Approach

Given the extensive existing logging, we recommend a **two-phase approach**:

### Phase 1: Quick Comparison (No Code Changes)

1. Run both engines with existing logging
2. Compare token IDs, embeddings, layer 0 stats
3. Check if llama.cpp sees Q[95]/Q[126] spikes
4. Compare final output quality

**Time:** 10 minutes  
**Effort:** Low  
**Value:** May find divergence without full checkpoint logging

### Phase 2: Full Checkpoint Logging (If Needed)

1. Add `log_checkpoint_*` calls to transformer code
2. Rebuild and run with `PRINTER_CHECKPOINT_LOGGING=1`
3. Convert to NPZ and run diff tool
4. Generate comprehensive diff report

**Time:** 1-2 hours  
**Effort:** Medium  
**Value:** Precise divergence point identification

---

## Checkpoints We Can Log

With the current infrastructure, we can capture:

| Checkpoint | Location | Data Type | Size |
|------------|----------|-----------|------|
| Embedding output | After `embed_tokens()` | FP16 | hidden_dim |
| Layer 0 attn norm | After first RMSNorm | FP16 | hidden_dim |
| Q/K/V pre-RoPE | After QKV projection | FP16 | q_dim, kv_dim |
| Q/K post-RoPE | After RoPE | FP16 | q_dim, kv_dim |
| Attention output | After GQA attention | FP16 | hidden_dim |
| FFN input | After FFN RMSNorm | FP16 | hidden_dim |
| FFN output | After down projection | FP16 | hidden_dim |
| Layer 0 residual | After residual add | FP16 | hidden_dim |
| Final hidden | Before LM head | FP16 | hidden_dim |
| LM head logits | After projection | FP32 | padded_vocab_size |

**Note:** Currently no checkpoints are logged by default. Add `log_checkpoint_*` calls as needed.

---

## Success Criteria

We succeed if we can answer:

1. **Do both engines tokenize identically?**
   - Same token IDs for BOS and token 1?

2. **Do both engines load the same vocab size?**
   - Our engine: 151936 (padded)
   - llama.cpp: ???

3. **Do both engines see the same embedding values?**
   - Compare first 10 floats of layer 0 input

4. **Do both engines see Q[95]/Q[126] spikes?**
   - If YES → Model file issue or expected
   - If NO → Our cuBLAS bug

5. **Where is the first divergence?**
   - Embedding? Layer 0? Attention? FFN? LM head?

---

## Red Flags

Abort and fix if you see:

- ❌ No `[TEAM PRINTER]` logs at startup
- ❌ manifest.json is empty
- ❌ cudaMemcpy failures or segfaults
- ❌ All-zero data in arrays
- ❌ Build fails with linker errors

See `GO_NO_GO_CHECKLIST.md` for detailed troubleshooting.

---

## Files Created

### Core Infrastructure
- ✅ `cuda/src/utils/checkpoint_logger.h`
- ✅ `cuda/src/utils/checkpoint_logger.cpp`

### Scripts & Tools
- ✅ `run_our_engine.sh`
- ✅ `run_llamacpp.sh`
- ✅ `convert_to_npz.py`
- ✅ `collect_parity_data.py`

### Documentation
- ✅ `README.md`
- ✅ `GO_NO_GO_CHECKLIST.md`
- ✅ `HANDOFF.md`
- ✅ `printer_meta.json`

### Modified Files
- ✅ `cuda/CMakeLists.txt` (added checkpoint_logger.cpp)
- ✅ `cuda/src/transformer/qwen_transformer.cpp` (integrated init/finalize)

---

## Next Team Recommendations

### If You Find Divergence Early (Embedding/Layer 0)

→ Hand off to **TEAM WEIGHT LOADER**
- Focus: Weight loading and dequantization
- Hypothesis: Weights loaded incorrectly or in wrong format
- Action: Compare raw weight bytes between engines

### If Divergence is in Q Projection

→ Hand off to **TEAM CUBLAS**
- Focus: cuBLAS parameter verification
- Hypothesis: lda, transpose, or compute type issue
- Action: Deep audit of all cuBLAS calls

### If Divergence is in Attention

→ Hand off to **TEAM ATTENTION**
- Focus: RoPE, attention scores, softmax
- Hypothesis: RoPE formula or attention mask issue
- Action: Compare Q/K after RoPE, attention weights

### If Divergence is in FFN

→ Hand off to **TEAM FFN**
- Focus: SwiGLU activation, down projection
- Hypothesis: Activation function or weight application
- Action: Compare gate/up/down intermediate values

### If No Divergence Found

→ Hand off to **TEAM DECODE**
- Focus: Tokenizer decode, vocab mapping
- Hypothesis: Token ID → UTF-8 conversion bug
- Action: Compare token IDs vs decoded strings

---

## Constraints & Rules

**TEAM PRINTER operates under strict rules:**

1. **Append-only** — Never delete previous teams' code or comments
2. **Foreground only** — No background builds or tests
3. **One hypothesis per change** — Isolate variables
4. **Non-invasive logging** — No math changes, logging only
5. **Utility mission** — Find divergence, don't fix it

---

## Timeline

**Infrastructure Setup:** 1 hour (COMPLETE)  
**Execution:** 10 minutes - 2 hours (depending on approach)  
**Analysis:** 30 minutes - 1 hour  
**Documentation:** 30 minutes

**Total:** 2-4 hours for complete parity sweep

---

## Contact & References

**Previous Teams:**
- TEAM BATTLESHIP — Q projection spikes investigation
- TEAM RACE CAR — FFN down projection verification
- TEAM SENTINEL — Layer 0 forward pass logging
- TEAM ORION — Activation divergence detection
- TEAM TOP HAT — Q-projection anomaly root cause

**Key Documents:**
- `../INVESTIGATION_CHRONICLE.md` — Complete investigation history
- `../TEAM_BATTLESHIP_HANDOFF.md` — Downstream wiring investigation
- `../TEAM_BATTLESHIP_FINDINGS.md` — Q spike findings

---

**TEAM PRINTER**  
**Status:** 🟢 READY TO EXECUTE  
**Mission:** Find the first divergence  
**Handoff:** Complete

*"Good data beats good guesses."*
