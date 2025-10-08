# TEAM DICKINSON — Hidden-State Parity Report (Round 2)

**Date:** 2025-10-08  
**Team:** DICKINSON  
**Mission:** Find the FIRST point of divergence between our CUDA path and llama.cpp

---

## Executive Summary

**Status:** 🚧 **INSTRUMENTATION COMPLETE** — Ready for execution

TEAM DICKINSON has successfully instrumented the CUDA forward pass with 7 strategic checkpoints to compare hidden states with llama.cpp. The instrumentation is minimal, append-only, and outputs JSONL for easy parsing.

**Deliverables:**
- ✅ Checkpoint instrumentation in `qwen_transformer.cpp`
- ⏳ JSONL logs from our implementation (pending test execution)
- ⏳ JSONL logs from llama.cpp (pending instrumentation)
- ⏳ Comparison analysis (pending data collection)

---

## Checkpoint Definitions

All checkpoints capture the **first token (tok=0)** and **first 16 dimensions** of the hidden state vector.

| Checkpoint | Location | File:Line | Description |
|------------|----------|-----------|-------------|
| **C0** | Post-embedding | `qwen_transformer.cpp:2798` | After token embedding lookup |
| **C1** | After layer 0 | `qwen_transformer.cpp:3044` | After layer 0 output_norm |
| **C5** | After layer 5 | `qwen_transformer.cpp:3045` | After layer 5 output_norm |
| **C10** | After layer 10 | `qwen_transformer.cpp:3046` | After layer 10 output_norm |
| **C23** | After layer 23 | `qwen_transformer.cpp:3047` | After final layer output_norm |
| **C24** | After output_norm | `qwen_transformer.cpp:3143` | After final RMSNorm (pre-lm_head) |
| **C25** | Logits | `qwen_transformer.cpp:3366` | After lm_head projection (pre-softmax) |

---

## JSONL Schema

Both our implementation and llama.cpp use the same schema:

```json
{
  "team": "DICKINSON",
  "ref": "ours|llama.cpp",
  "chk": "C0|C1|C5|C10|C23|C24|C25",
  "tok": 0,
  "dims": 16,
  "dtype": "f16|f32",
  "values": [<16 floats>]
}
```

**Notes:**
- C0-C24 use `dtype=f16` (FP16 hidden states)
- C25 uses `dtype=f32` (FP32 logits)
- Values are truncated to 6 decimal places for file size

---

## Instrumentation Details

### Our Implementation

**File:** `bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`

**Strategy:**
- Trigger on first forward pass only (`dickinson_forward_count == 0`)
- Copy first 16 FP16 values from GPU to host
- Convert to FP32 and print JSONL to stderr
- Zero overhead after first forward pass

**Code Pattern:**
```cpp
// [TEAM DICKINSON 2025-10-08] CHECKPOINT CX
if (do_dickinson_log) {
    const half* hptr = reinterpret_cast<const half*>(data);
    float tmp[16];
    for (int i = 0; i < 16; i++) tmp[i] = __half2float(hptr[i]);
    fprintf(stderr, "{\"team\":\"DICKINSON\",\"ref\":\"ours\",\"chk\":\"CX\",...}\n", ...);
}
```

**Run Command:**
```bash
cd bin/worker-orcd
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  --features cuda --release -- --ignored --nocapture --test-threads=1 \
  > /tmp/dickinson_ours.log 2>&1

grep '"team":"DICKINSON"' /tmp/dickinson_ours.log > /tmp/dickinson_ours.jsonl
```

### llama.cpp Reference

**Status:** ⏳ **PENDING INSTRUMENTATION**

**Strategy:**
- Add similar checkpoints in llama.cpp's forward pass
- Use existing ORCH_LOGGING infrastructure if available
- Emit same JSONL schema with `"ref":"llama.cpp"`

**Run Command (proposed):**
```bash
cd reference/llama.cpp/build
timeout 30s ./bin/llama-cli \
  -m ../../../.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "GPU haiku with word fifty-one: " \
  -n 1 --temp 0.7 --top-k 0 --top-p 1.0 \
  </dev/null > /tmp/dickinson_llama.log 2>&1

grep '"team":"DICKINSON"' /tmp/dickinson_llama.log > /tmp/dickinson_llama.jsonl
```

**Files to modify:**
- `reference/llama.cpp/src/llama.cpp` (core forward pass)
- OR `reference/llama.cpp/tools/main/main.cpp` (if using ORCH_LOGGING)

---

## Comparison Methodology

### Step 1: Parse JSONL Files

```python
import json

def load_checkpoints(filename):
    checkpoints = {}
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            checkpoints[data['chk']] = data['values']
    return checkpoints

ours = load_checkpoints('/tmp/dickinson_ours.jsonl')
theirs = load_checkpoints('/tmp/dickinson_llama.jsonl')
```

### Step 2: Compute Max Absolute Difference

```python
import numpy as np

def compare_checkpoint(chk, ours, theirs):
    v1 = np.array(ours[chk])
    v2 = np.array(theirs[chk])
    diff = np.abs(v1 - v2)
    max_diff = np.max(diff)
    return max_diff

for chk in ['C0', 'C1', 'C5', 'C10', 'C23', 'C24', 'C25']:
    max_diff = compare_checkpoint(chk, ours, theirs)
    status = "✅" if max_diff <= 1e-3 else "❌"
    print(f"{chk}: max_diff={max_diff:.6f} {status}")
```

### Step 3: Identify First Divergence

A checkpoint "matches" if `max_abs_diff <= 1e-3` (FP16-friendly tolerance).

The **first checkpoint** with `max_abs_diff > 1e-3` is the divergence point.

---

## Expected Outcomes

### Scenario 1: Perfect Parity (All Checkpoints Match)

**Verdict:** ✅ Forward pass is correct

**Interpretation:**
- Our CUDA implementation matches llama.cpp numerically
- Bug must be in tokenization, vocab mapping, or sampling
- Escalate to TEAM FROST (sampling) or TEAM SHAKESPEARE (tokenization)

### Scenario 2: Divergence at C0 (Embedding)

**Verdict:** ❌ Embedding lookup or vocab mapping issue

**Hypothesis:**
- Embedding table transposed (dimensions swapped)
- Token ID mapping incorrect
- Embedding scaling missing

**Next Steps:**
- Compare embedding table dimensions with llama.cpp
- Verify token ID → embedding index calculation
- Check for post-embedding scaling (e.g., `sqrt(hidden_dim)`)

### Scenario 3: Divergence at C1-C23 (Layer N)

**Verdict:** ❌ Bug in layer N subsystem

**Hypothesis (by layer):**
- **C1 (layer 0):** Attention, RoPE, or FFN in first layer
- **C5-C10 (mid layers):** Accumulating numerical error or weight loading issue
- **C23 (final layer):** Last-layer FFN or attention projection

**Next Steps:**
- Instrument layer N in detail (Q/K/V projections, attention output, FFN gates)
- Compare cuBLAS parameters for that layer's matmuls
- Verify weight loading for that layer

### Scenario 4: Divergence at C24 (Output Norm)

**Verdict:** ❌ Final RMSNorm or output_norm weights issue

**Hypothesis:**
- RMSNorm epsilon mismatch
- output_norm.weight loaded incorrectly
- Gamma scaling wrong

**Next Steps:**
- Compare RMSNorm formula with llama.cpp
- Dump output_norm.weight and compare with GGUF
- Verify epsilon value (should be 1e-6)

### Scenario 5: Divergence at C25 (Logits)

**Verdict:** ❌ LM head projection issue

**Hypothesis:**
- lm_head weight transposed
- cuBLAS parameters wrong (CUBLAS_OP_T vs CUBLAS_OP_N)
- lda/ldb/ldc stride mismatch

**Next Steps:**
- Compare lm_head cuBLAS call with llama.cpp
- Verify lm_head weight dimensions and layout
- Check if lm_head is tied to embedding table

---

## Current Status

### Completed ✅

1. **Instrumentation Design**
   - 7 strategic checkpoints identified
   - JSONL schema defined
   - Comparison methodology documented

2. **Our Implementation**
   - Checkpoints added to `qwen_transformer.cpp`
   - Append-only breadcrumbs (no refactoring)
   - Zero overhead after first forward pass

### Pending ⏳

1. **Test Execution**
   - Run our implementation and capture JSONL
   - **Blocker:** Test currently failing with HTTP error
   - **Workaround:** Need to fix test infrastructure or run standalone

2. **llama.cpp Instrumentation**
   - Add matching checkpoints to llama.cpp
   - Emit same JSONL schema
   - Run with same prompt

3. **Comparison Analysis**
   - Parse both JSONL files
   - Compute max_abs_diff per checkpoint
   - Identify first divergence

4. **Root Cause Investigation**
   - Once divergence found, drill into that subsystem
   - Compare code/params with llama.cpp
   - Propose fix

---

## Blockers & Risks

### Blocker 1: Test Infrastructure

**Issue:** `haiku_generation_anti_cheat` test fails with HTTP error before reaching inference

**Evidence:**
```
❌ Request failed: error sending request for url (http://localhost:40555/execute)
thread 'test_haiku_generation_stub_pipeline_only' panicked
```

**Impact:** Cannot capture JSONL logs from our implementation

**Workaround Options:**
1. Fix the test infrastructure (HTTP server startup timing?)
2. Run worker-orcd standalone and send manual HTTP request
3. Create minimal C++ test harness that calls forward() directly

### Risk 1: llama.cpp Instrumentation Complexity

**Issue:** llama.cpp's forward pass is complex and spread across multiple files

**Mitigation:**
- Start with existing ORCH_LOGGING infrastructure
- Only add checkpoints at layer boundaries (easier to find)
- Use `llama_get_embeddings_ith()` and similar APIs if available

### Risk 2: Numerical Tolerance

**Issue:** FP16 precision means small differences are expected

**Mitigation:**
- Use tolerance of 1e-3 (generous for FP16)
- Look for **sudden spikes** in diff, not gradual accumulation
- Compare patterns, not absolute values

---

## Next Steps (Priority Order)

1. **Fix test infrastructure** to capture our JSONL logs
   - Debug HTTP error in `haiku_generation_anti_cheat`
   - OR create standalone test harness

2. **Instrument llama.cpp** with matching checkpoints
   - Add C0-C25 checkpoints to llama.cpp forward pass
   - Run with same prompt and capture JSONL

3. **Run comparison analysis**
   - Parse both JSONL files
   - Compute max_abs_diff per checkpoint
   - Generate comparison table

4. **Investigate first divergence**
   - Drill into divergent subsystem
   - Compare with llama.cpp implementation
   - Propose fix

5. **Update chronicle** with findings
   - Document divergence point
   - Record root cause hypothesis
   - Hand off to next team

---

## Files Modified

### Our Implementation

```
bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp
  Lines 2777-2800: C0 checkpoint (post-embedding)
  Lines 2967-2982: Helper for layer checkpoints
  Lines 3042-3048: C1, C5, C10, C23 checkpoints (layer outputs)
  Lines 3142-3151: C24 checkpoint (post output_norm)
  Lines 3365-3374: C25 checkpoint (logits)
```

### llama.cpp (Pending)

```
reference/llama.cpp/src/llama.cpp (TBD)
reference/llama.cpp/tools/main/main.cpp (TBD)
```

---

## Conclusion

TEAM DICKINSON has successfully instrumented the CUDA forward pass with 7 strategic checkpoints. The instrumentation is minimal, non-invasive, and ready for execution.

**Current Blocker:** Test infrastructure failing before inference runs

**Recommended Action:** Fix test infrastructure OR create standalone test harness to capture JSONL logs

Once logs are captured from both implementations, the comparison analysis can proceed and the first divergence point will be identified with high confidence.

---

**TEAM DICKINSON**  
*"Tell all the truth but tell it slant—Success in Circuit lies."*

**Report Status:** 🚧 INSTRUMENTATION COMPLETE — AWAITING EXECUTION  
**Last Updated:** 2025-10-08T00:00Z
