# 🔍 Bug Hunt Checklist — Garbage Output Root Cause

**Generated:** 2025-10-07T08:20Z  
**Source:** Synthesized from Investigation Chronicle + codebase sweep  
**Bug:** Model generates mojibake/garbage output instead of readable haiku text  
**Model:** qwen2.5-0.5b-instruct-fp16.gguf (FP16, no quantization)

---

## 🎯 TL;DR — Top 5 Most Likely Suspects (Start Here)

### 1. 🔥 **LM Head Output Projection Parity** [UNTESTED - HIGHEST LEVERAGE]
**Why Now:** RACE CAR verified FFN is healthy. BATTLESHIP verified attention is healthy. Final projection is last untested GEMM.  
**File:** `cuda/src/transformer/qwen_transformer.cpp:1757-1793` (lm_head projection)  
**Evidence Gap:** All intermediate activations healthy (±3 range) but output tokens are wrong → bug must be in final hidden→logits projection.

**Quick Probe (5 min):**
```cpp
// Add at line 1790 (after lm_head GEMM)
if (pos == 0) {  // First token only
    __half* h_logits = new __half[10];
    cudaMemcpy(h_logits, logits_output_, 10 * sizeof(__half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[LM_HEAD] Token 0 first 10 logits: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", __half2float(h_logits[i]));
    fprintf(stderr, "\n");
    delete[] h_logits;
}
```

**Pass/Fail:**
- **PASS:** Logits show peaked distribution (one value >>others, e.g., `2.1 -3.4 -1.2 ...`)
- **FAIL:** All logits similar magnitude (flat distribution) OR extreme outliers (±100+)

**If FAIL → Next Action:**
- Compare lm_head weight dimensions with llama.cpp
- Verify lda/ldb/ldc parameters for lm_head GEMM
- Test CUBLAS_COMPUTE_32F vs FAST_16F for lm_head

---

### 2. 🔥 **RoPE Numeric Output Verification** [Formula verified, computation NOT compared]
**Why Now:** Chronicle shows RoPE formula verified (POLARIS) but actual Q/K values before/after RoPE never compared with llama.cpp.  
**File:** `cuda/kernels/rope.cu:67-140` (RoPE kernel)  
**Evidence Gap:** All teams checked formula, nobody dumped actual rotated values.

**Quick Probe (5 min):**
```cpp
// In qwen_transformer.cpp after RoPE call (around line 1022)
if (layer_idx == 0 && pos == 0) {
    __half* h_q_after = new __half[16];
    __half* h_k_after = new __half[16];
    cudaMemcpy(h_q_after, q_proj_, 16 * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_after, k_proj_, 16 * sizeof(__half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[ROPE] Layer 0 Token 0 Q[0:16] after RoPE: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%.4f ", __half2float(h_q_after[i]));
    fprintf(stderr, "\n[ROPE] Layer 0 Token 0 K[0:16] after RoPE: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%.4f ", __half2float(h_k_after[i]));
    fprintf(stderr, "\n");
    delete[] h_q_after; delete[] h_k_after;
}
```

**Pass/Fail:**
- **PASS:** Values match llama.cpp RoPE output for same prompt (within ±0.01)
- **FAIL:** Values diverge from llama.cpp

**If FAIL → Next Action:**
- Check `rope_freq_base` parameter (should be 1000000.0 per Chronicle line 80)
- Verify head_dim calculation (should be 64 per Chronicle)
- Compare sin/cos computation with llama.cpp implementation

---

### 3. 🔥 **Config Parameter Mismatch** [Assumed correct, never verified]
**Why Now:** llama.cpp log shows specific config (n_ff=4864, n_head=14, n_head_kv=2). Never verified our code uses same values.  
**File:** `cuda/src/model/qwen_weight_loader.cpp:96-130` (config parsing)  
**Evidence Gap:** Teams assumed config is correct. Never explicitly dumped and compared.

**Quick Probe (3 min):**
```cpp
// Add at top of qwen_transformer_forward() first call
static bool config_logged = false;
if (!config_logged) {
    fprintf(stderr, "[CONFIG] num_layers=%u num_heads=%u num_kv_heads=%u head_dim=%u\n",
            config.num_layers, config.num_heads, config.num_kv_heads, config.head_dim);
    fprintf(stderr, "[CONFIG] hidden_dim=%u ffn_dim=%u vocab_size=%u padded_vocab=%u\n",
            config.hidden_dim, config.ffn_dim, config.vocab_size, config.padded_vocab_size);
    fprintf(stderr, "[CONFIG] rope_freq_base=%.1f rms_norm_eps=%.9f\n",
            config.rope_freq_base, config.rms_norm_eps);
    config_logged = true;
}
```

**Pass/Fail:**
- **PASS:** All match llama.cpp: `num_heads=14 num_kv_heads=2 head_dim=64 hidden_dim=896 ffn_dim=4864 rope_freq_base=1000000.0 rms_norm_eps=1e-06`
- **FAIL:** Any value differs from llama.cpp

**If FAIL → Next Action:**
- Fix config parsing in `qwen_weight_loader.cpp`
- Rebuild and retest

---

### 4. 🔥 **Embedding Scaling Factor** [Never checked if scaling applied]
**Why Now:** Chronicle line 889-891 notes "Some models scale by sqrt(hidden_dim)". Never verified if llama.cpp scales and we don't.  
**File:** `cuda/kernels/embedding.cu:117-147` (embedding lookup)  
**Evidence Gap:** Direct lookup with no scaling. llama.cpp might scale embeddings.

**Quick Probe (5 min):**
```cpp
// Add at end of cuda_embedding_lookup kernel (embedding.cu line 145)
if (token_idx == 0 && threadIdx.x < 16 && blockIdx.x == 0) {
    printf("[EMBEDDING] Token 0 first 16 values: %.6f ", __half2float(output[threadIdx.x]));
}
```

Then run llama.cpp with verbose embedding logging and compare.

**Pass/Fail:**
- **PASS:** Values match llama.cpp embedding output exactly
- **FAIL:** Our values are ~30x smaller/larger than llama.cpp (indicates missing `sqrt(896)=29.9` scaling)

**If FAIL → Next Action:**
- Add scaling: `output[idx] = __float2half(__half2float(embedding_value) * sqrtf(hidden_dim));`

---

### 5. 🔥 **Tokenizer Decode Path** [Not investigated at all]
**Why Now:** RACE CAR handoff suggests this as highest priority. All internal activations healthy but output text garbage.  
**File:** `src/inference/cuda_backend.rs:560-595` (token decode), Rust tokenizer integration  
**Evidence Gap:** Nobody verified token IDs→string decode uses correct vocabulary and UTF-8 encoding.

**Quick Probe (5 min):**
```rust
// In cuda_backend.rs after sampling (around line 580)
eprintln!("[TOKENIZER] Generated token_id: {} at pos: {}", token_id, position);
eprintln!("[TOKENIZER] Top 5 token IDs from logits: <dump top 5>");
eprintln!("[TOKENIZER] Decoded string for token {}: {:?}", token_id, self.tokenizer.decode(&[token_id])?);
```

**Pass/Fail:**
- **PASS:** Token IDs match llama.cpp for same prompt, decoded strings are readable
- **FAIL:** Token IDs differ from llama.cpp OR decoded strings are mojibake

**If FAIL → Next Action:**
- Compare tokenizer vocabulary with GGUF file
- Verify UTF-8 encoding in decode step
- Check if using correct tokenizer model (should be "gpt2" per llama.cpp log line 28)

---

## 📋 Full Checklist — Organized by Category

---

## Category A: Weight Loading & Dequantization Parity

### A1. ✅ FFN Down Projection Loading [TESTED by CHARLIE_BETA & RACE_CAR - VERIFIED WORKING]
**Status:** CLOSED ✅  
**Evidence:** Chronicle lines 379-389 show ffn_down was added and tested. RACE CAR verified all FFN weights non-null and activations healthy.

### A2. 🟡 Weight Byte Order / Endianness [UNTESTED]
**Hypothesis:** llama.cpp and our code interpret FP16 bytes differently  
**File:** `cuda/src/model/qwen_weight_loader.cpp:309-312` (load_tensor_to_vram)  
**Why Now:** llama.cpp works, we don't. Same file, different byte interpretation?

**Probe:**
```cpp
// Dump first 32 bytes of blk.0.attn_q.weight in hex
void* h_weight = malloc(32);
cudaMemcpy(h_weight, model->weights.layers[0].attn_q, 32, cudaMemcpyDeviceToHost);
fprintf(stderr, "[BYTE_ORDER] attn_q first 32 bytes (hex): ");
for (int i = 0; i < 32; i++) fprintf(stderr, "%02x ", ((uint8_t*)h_weight)[i]);
fprintf(stderr, "\n");
free(h_weight);
```

Compare with llama.cpp memory dump of same weight.

**Pass/Fail:**
- **PASS:** Hex bytes match llama.cpp
- **FAIL:** Bytes are swapped (e.g., little-endian vs big-endian)

**If FAIL:** Add endianness conversion in weight loading.

### A3. 🟡 Weight Memory Alignment [UNTESTED]
**Hypothesis:** Misaligned weight pointers cause incorrect GEMM reads  
**File:** `cuda/src/model/qwen_weight_loader.cpp:239-242` (alignment check)  
**Why Now:** cuBLAS requires 128-byte alignment for optimal performance.

**Probe:**
```cpp
fprintf(stderr, "[ALIGNMENT] attn_q=%p (%%128=%ld)\n", layers[0].attn_q, (uintptr_t)layers[0].attn_q % 128);
fprintf(stderr, "[ALIGNMENT] ffn_down=%p (%%128=%ld)\n", layers[0].ffn_down, (uintptr_t)layers[0].ffn_down % 128);
fprintf(stderr, "[ALIGNMENT] lm_head=%p (%%128=%ld)\n", lm_head, (uintptr_t)lm_head % 128);
```

**Pass/Fail:**
- **PASS:** All %128 == 0 (properly aligned)
- **FAIL:** Any weight misaligned

**If FAIL:** Use aligned CUDA malloc (`cudaMallocPitch` or `cudaMalloc` with manual alignment).

---

## Category B: Normalization & Scaling

### B1. ✅ RMSNorm Epsilon [VERIFIED by HYPERION]
**Status:** CLOSED ✅  
**Evidence:** Chronicle line 544 confirms epsilon=1e-6f matches llama.cpp.

### B2. 🟡 RMSNorm Numeric Output [Formula verified, actual values NOT compared]
**Hypothesis:** RMSNorm computation has FP16 precision issues  
**File:** `cuda/kernels/rmsnorm.cu:28-66` (RMSNorm kernel)  
**Why Now:** Formula correct but nobody compared actual normed values with llama.cpp.

**Probe:**
```cpp
// After first RMSNorm (attn_norm, layer 0, token 0)
if (layer_idx == 0 && pos == 0) {
    __half* h_normed = new __half[16];
    cudaMemcpy(h_normed, attn_normed_, 16 * sizeof(__half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[RMSNORM] Layer 0 Token 0 attn_norm output[0:16]: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%.4f ", __half2float(h_normed[i]));
    fprintf(stderr, "\n");
    delete[] h_normed;
}
```

Compare with llama.cpp RMSNorm output.

**Pass/Fail:**
- **PASS:** Values match llama.cpp (within ±0.01)
- **FAIL:** Values diverge significantly

**If FAIL:** Check RMS calculation accumulator precision, verify division by sqrt(rms^2 + eps).

### B3. 🔥 **Embedding Scaling** [DUPLICATE of Top 5 #4 - See above]

---

## Category C: Attention Mechanism

### C1. ✅ Q-Projection Spikes at [95, 126] [VERIFIED HARMLESS by BATTLESHIP]
**Status:** CLOSED ✅  
**Evidence:** Chronicle lines 998-1021 show BATTLESHIP proved spikes are filtered by softmax and don't propagate.

### C2. ✅ KV Cache Infrastructure [VERIFIED by WATER]
**Status:** CLOSED ✅  
**Evidence:** Chronicle lines 95-122 show cache_len increments correctly (0,1,2,3...).

### C3. ✅ Causal Masking [VERIFIED by BYGONE]
**Status:** CLOSED ✅  
**Evidence:** Chronicle lines 235-265 show decode kernel only attends to 0..cache_len (IS causal masking).

### C4. 🟡 Attention Softmax Temperature [UNTESTED]
**Hypothesis:** Attention softmax might have wrong temperature/scaling  
**File:** `cuda/kernels/gqa_attention.cu:423-466` (softmax in attention kernel)  
**Why Now:** Attention softmax should scale by `1/sqrt(head_dim)`, never verified.

**Probe:**
```cpp
// In gqa_attention.cu after softmax (around line 460)
if (layer == 0 && pos == 0 && head == 0) {
    printf("[ATTN_SOFTMAX] Head 0 Token 0 attn_weights[0:8]: ");
    for (int i = 0; i < 8 && i <= cache_len; i++) {
        printf("%.6f ", attn_weights[i]);
    }
    printf(" (sum should be ~1.0)\\n");
}
```

**Pass/Fail:**
- **PASS:** Sum ~1.0, values decay smoothly
- **FAIL:** Sum != 1.0 or all values equal

**If FAIL:** Check scaling factor before softmax (should be `1.0/sqrt(64.0) = 0.125`).

---

## Category D: Feed-Forward Network

### D1. ✅ SwiGLU Activation [VERIFIED by HYPERION & RACE CAR]
**Status:** CLOSED ✅  
**Evidence:** Chronicle line 545 (formula verified) + RACE CAR showed healthy FFN activations.

### D2. ✅ FFN Gate/Up/Down Weights [VERIFIED by RACE CAR]
**Status:** CLOSED ✅  
**Evidence:** RACE CAR lines 34-42 show all pointers non-null, activations healthy.

---

## Category E: Final Projection & Sampling

### E1. 🔥 **LM Head Output Projection** [DUPLICATE of Top 5 #1 - See above]

### E2. ✅ Sampling Pipeline Order [FIXED by HELIOS]
**Status:** CLOSED ✅  
**Evidence:** Chronicle lines 710-749 show HELIOS fixed softmax→top-p order (was broken).

### E3. 🟡 Temperature Application [Verified working by HELIOS but worth re-checking]
**Hypothesis:** Temperature 0.7 applied incorrectly  
**File:** `cuda/kernels/sampling_wrapper.cu:289-303`  
**Why Now:** HELIOS verified temp=0.7 applied, but output still wrong.

**Probe:**
```cpp
// After temperature scaling (sampling_wrapper.cu line 295)
if (pos == 0) {
    float* h_logits = new float[10];
    cudaMemcpy(h_logits, logits, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[TEMP] Token 0 first 10 logits AFTER temp=%.2f: ", temperature);
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", h_logits[i]);
    fprintf(stderr, "\n");
    delete[] h_logits;
}
```

**Pass/Fail:**
- **PASS:** Logits scaled correctly (divide by 0.7 makes distribution sharper)
- **FAIL:** No visible difference from raw logits

---

## Category F: Position Encoding

### F1. 🔥 **RoPE Numeric Output** [DUPLICATE of Top 5 #2 - See above]

### F2. 🟡 RoPE Frequency Base [Assumed correct, never logged]
**Hypothesis:** rope_freq_base != 1000000.0  
**File:** `cuda/kernels/rope.cu:90-95` (frequency calculation)  
**Why Now:** llama.cpp log shows freq_base=1000000.0 (line 80), never verified ours matches.

**Probe:**
```cpp
// In rope kernel or before call
printf("[ROPE] freq_base=%.1f (should be 1000000.0)\\n", rope_freq_base);
```

**Pass/Fail:**
- **PASS:** 1000000.0
- **FAIL:** Different value

**If FAIL:** Fix config parsing to use correct freq_base from GGUF metadata.

---

## Category G: Configuration & Metadata

### G1. 🔥 **Config Parameter Mismatch** [DUPLICATE of Top 5 #3 - See above]

### G2. 🟡 Vocab Size Confusion [Partially fixed by CHAIR, worth verifying]
**Hypothesis:** Embedding uses padded_vocab (151936) instead of real vocab (151936)  
**File:** `cuda/kernels/embedding.cu:93-104` (bounds check)  
**Evidence:** CHAIR noted confusion between input vocab (151643?) and output vocab (151936).

**Probe:**
```cpp
// In embedding kernel
if (token_idx == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[EMBEDDING] vocab_size param=%u (embedding table has %u rows)\\n", vocab_size, <actual_rows>);
}
```

**Pass/Fail:**
- **PASS:** Both match (151936)
- **FAIL:** Mismatch causes bounds check failure

**If FAIL:** Use correct vocab size from token_embd.weight dimensions.

---

## Category H: Token Decode & Output

### H1. 🔥 **Tokenizer Decode Path** [DUPLICATE of Top 5 #5 - See above]

### H2. 🟡 Token ID→String Mapping [UNTESTED]
**Hypothesis:** Tokenizer vocabulary doesn't match GGUF file  
**File:** Rust tokenizer integration  
**Why Now:** Generated token IDs might be correct but decoded with wrong vocabulary.

**Probe:**
```rust
// Decode first 10 tokens of vocabulary
for token_id in 0..10 {
    eprintln!("[VOCAB] Token {} = {:?}", token_id, self.tokenizer.decode(&[token_id])?);
}
```

Compare with llama.cpp vocabulary (lines 30-31 of llamacpp.run.log).

**Pass/Fail:**
- **PASS:** Token 0 = "!", Token 1 = "\"", etc. (matches llama.cpp)
- **FAIL:** Vocabulary differs

**If FAIL:** Load tokenizer from GGUF metadata instead of separate vocabulary file.

---

## 🚫 DO NOT RE-INVESTIGATE (Already Verified Correct)

Per Chronicle, these have been thoroughly tested and proven correct:

1. ❌ **Tokenization** (BLUE, PURPLE) - Special tokens correctly inserted as single IDs
2. ❌ **Token Embeddings** (PURPLE) - Valid FP16 values, correct lookup
3. ❌ **cuBLAS Matrix Multiplication Q[0]** (CHARLIE, ORION) - Mathematically verified
4. ❌ **cuBLAS Transpose Parameters** (FELICIA, AURORA, THIMBLE) - CUBLAS_OP_N is correct
5. ❌ **Tensor-Core Compute Type** (TOP HAT) - FAST_16F and 32F both work
6. ❌ **Weight Corruption at Columns 95/126** (TOP HAT) - Weights are normal
7. ❌ **Input Spikes** (TOP HAT) - Normed input is normal (±1 range)
8. ❌ **Q/K/V Biases** (GREEN) - All zeros, not the issue
9. ❌ **KV Cache** (WATER) - Position tracking and cache writes correct
10. ❌ **Causal Masking** (BYGONE) - Implemented correctly in kernel
11. ❌ **Prefill Logic** (BYGONE) - One-at-a-time is correct for autoregressive
12. ❌ **Sampling Order** (HELIOS) - Fixed: softmax before top-p
13. ❌ **Attention Filtering** (BATTLESHIP) - Q spikes filtered by softmax
14. ❌ **Buffer Aliasing** (BATTLESHIP) - No buffer corruption detected
15. ❌ **FFN Down Projection** (CHARLIE_BETA, RACE CAR) - Loaded correctly, activations healthy
16. ❌ **SwiGLU Activation** (HYPERION, RACE CAR) - Formula and values correct
17. ❌ **RMSNorm Formula** (HYPERION) - Formula matches llama.cpp
18. ❌ **RoPE Formula** (POLARIS, HYPERION) - Formula conceptually correct

---

## 🎯 Recommended Investigation Sequence

**Phase 1: Quick Wins (30 minutes)**
1. Run probe for **Top 5 #3** (Config Parameter Mismatch) — easiest to verify
2. Run probe for **Top 5 #5** (Tokenizer Decode Path) — might be the whole issue
3. Run probe for **Top 5 #4** (Embedding Scaling) — common oversight

**Phase 2: Numeric Parity (1 hour)**
4. Run probe for **Top 5 #1** (LM Head Projection) — last untested GEMM
5. Run probe for **Top 5 #2** (RoPE Numeric Output) — formula vs computation gap
6. Run probe for **B2** (RMSNorm Numeric Output) — verify precision

**Phase 3: Deep Dive (2 hours)**
7. Run TEAM PRINTER parity comparison (already has infrastructure)
8. Compare all checkpoints with llama.cpp systematically
9. Identify first divergence point

**Phase 4: If Still Stuck**
10. Check weight byte order (A2)
11. Check attention softmax scaling (C4)
12. Full memory alignment audit (A3)

---

## 📊 Verification Commands

**Primary Test:**
```bash
cd bin/worker-orcd
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1 | tee test_output.log
```

**HTTP Probe:**
```bash
# Start worker first, then:
curl -sS -X POST http://localhost:9999/execute \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Write a haiku about GPU computing that includes the word \"forty-two\" (nonce: TEST)",
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq '.generated_text'
```

**Success Criteria:**
- Output is human-readable English haiku
- No mojibake (Éķ, âĪ¬, ĳľ, etc.)
- No code tokens (FileWriter, strcasecmp, etc.)
- Contains requested word ("forty-two")

---

## 📁 Artifacts Location

All investigation notes: `/home/vince/Projects/llama-orch/bin/worker-orcd/investigation-teams/`

Key documents:
- `INVESTIGATION_CHRONICLE.md` — Full investigation history
- `BUG_HUNT_MISSION_TEMPLATE.md` — Rules and verification commands
- `TEAM_*_HANDOFF.md` — Individual team findings
- `FALSE_LEADS_SUMMARY.md` — What NOT to retry

---

**Generated by Cascade AI**  
**Based on 15+ teams, 33+ hours of investigation**  
**Synthesized from 50+ investigation documents**
