# 🎨 Post-Fix Investigation Plan - Specialized Teams

**Date:** 2025-10-07T13:59Z  
**Purpose:** Systematic re-validation of all components after bug fixes  
**Status:** 🚀 READY TO DEPLOY

---

## 🎯 Mission Overview

**Context:** Multiple critical bugs have been fixed (softmax underflow, sampling order, cuBLAS parameters, corrupted weights, config overrides). Now we need to:

1. **Verify** which fixes are actually applied in current code
2. **Resolve** contradictions between team findings
3. **Re-test** components that were dismissed as "not the bug"
4. **Validate** end-to-end functionality

**Team Structure:** Each team is specialized in ONE unit of work. Teams are named after famous painters and poets.

---

## 📋 Team Roster & Specializations

| Team | Specialization | Mission Type |
|------|----------------|--------------|
| 🎨 **TEAM MONET** | Code Audit | Verify current state |
| 🎨 **TEAM PICASSO** | cuBLAS Resolution | Resolve contradictions |
| 🎨 **TEAM VAN GOGH** | Weight Verification | Resolve contradictions |
| 📝 **TEAM SHAKESPEARE** | End-to-End Testing | Integration validation |
| 📝 **TEAM FROST** | Sampling Verification | Component validation |
| 📝 **TEAM DICKINSON** | Hidden State Parity | Component validation |
| 🎨 **TEAM REMBRANDT** | Reverted Fixes | Re-apply & test |
| 📝 **TEAM WHITMAN** | False Leads Cleanup | Documentation |

---

## 🎨 TEAM MONET - The Code Auditor

**Specialization:** Current State Verification  
**Mission:** Audit current codebase to determine which fixes are actually applied

### 📦 Unit of Work: Code Audit

**Objective:** Create a definitive report of current code state

**Tasks:**

1. **Check cuBLAS Parameters (ALL 8 matmuls)**
   ```bash
   # Files to check:
   - cuda/src/transformer/qwen_transformer.cpp (Q/K/V/attn_out/lm_head)
   - cuda/kernels/swiglu_ffn.cu (FFN gate/up/down)
   ```
   
   **For each matmul, document:**
   - [ ] CUBLAS_OP_N or CUBLAS_OP_T?
   - [ ] lda value?
   - [ ] Line number
   - [ ] Last modified by which team?

2. **Check Softmax Implementation**
   ```bash
   # File to check:
   - cuda/kernels/sampling.cu (softmax kernel)
   ```
   
   **Document:**
   - [ ] Using double precision for sum accumulation?
   - [ ] Line number
   - [ ] CASCADE's fix applied? (Y/N)

3. **Check Sampling Order**
   ```bash
   # File to check:
   - cuda/kernels/sampling_wrapper.cu
   ```
   
   **Document:**
   - [ ] Top-P before or after softmax?
   - [ ] Line number
   - [ ] HELIOS's fix applied? (Y/N)

4. **Check Output Norm Weights**
   ```bash
   # File to check:
   - cuda/src/model/qwen_weight_loader.cpp (output_norm.weight loading)
   ```
   
   **Document:**
   - [ ] Are weights normalized (mean=1.0)?
   - [ ] Or raw (mean=7.14, max=16.75)?
   - [ ] Output Norm Team's fix applied? (Y/N)

5. **Check Q/K/V Biases**
   ```bash
   # Files to check:
   - cuda/src/model/qwen_weight_loader.cpp (bias loading)
   - cuda/src/transformer/qwen_transformer.cpp (bias addition)
   ```
   
   **Document:**
   - [ ] Biases loaded (not nullptr)?
   - [ ] Biases added after projections?
   - [ ] GREEN's fix applied? (Y/N)

6. **Check Configuration Overrides**
   ```bash
   # Files to check:
   - src/inference/cuda_backend.rs (temperature, system prompt)
   ```
   
   **Document:**
   - [ ] Hardcoded temperature removed?
   - [ ] Hardcoded system prompt removed?
   - [ ] FINNEY's fix applied? (Y/N)

### 📊 Deliverable

**File:** `investigation-teams/TEAM_MONET_CODE_AUDIT.md`

**Format:**
```markdown
# TEAM MONET - Code Audit Report

## Summary
- Fixes Applied: X/6
- Fixes Missing: Y/6
- Conflicts Found: Z

## Detailed Findings

### 1. cuBLAS Parameters
| Matmul | File | Line | Operation | lda | Status |
|--------|------|------|-----------|-----|--------|
| Q proj | ... | ... | CUBLAS_OP_T | 896 | ✅ SENTINEL's fix |
| ... | ... | ... | ... | ... | ... |

### 2. Softmax Implementation
- Double precision: ✅ YES / ❌ NO
- CASCADE's fix: ✅ APPLIED / ❌ MISSING

### 3. [Continue for all 6 checks]

## Critical Issues
- [List any missing fixes or conflicts]

## Recommendation
- [Next steps based on findings]
```

**Success Criteria:**
- ✅ All 6 fix categories checked
- ✅ Current state documented with line numbers
- ✅ Conflicts/missing fixes identified

---

## 🎨 TEAM PICASSO - The cuBLAS Resolver

**Specialization:** Contradiction Resolution  
**Mission:** Resolve the CUBLAS_OP_T vs CUBLAS_OP_N contradiction

### 📦 Unit of Work: cuBLAS Contradiction Resolution

**Objective:** Determine definitively whether CUBLAS_OP_T or CUBLAS_OP_N is correct

**Background:**
- FELICIA: "CUBLAS_OP_T is WRONG" (reverted)
- AURORA: "CUBLAS_OP_T is WRONG" (reverted)
- SENTINEL: "CUBLAS_OP_T is CORRECT" (applied)
- ALPHA: "CUBLAS_OP_N is CORRECT" (verified)

**Tasks:**

1. **Read Current Code State (from TEAM MONET)**
   - [ ] Which operation is currently used?
   - [ ] Which lda values are currently used?

2. **Reproduce ALPHA's Verification**
   ```bash
   # Run ALPHA's manual verification test
   cd bin/worker-orcd
   cargo test --test verify_manual_q0 --features cuda --release -- --nocapture
   ```
   
   **Document:**
   - [ ] Does manual calculation match cuBLAS?
   - [ ] What are the actual values?
   - [ ] Test passes with current code?

3. **Reproduce SENTINEL's Verification**
   ```bash
   # Check if SENTINEL's verification code still exists
   # If not, recreate it based on TEAM_SENTINEL_VICTORY.md
   ```
   
   **Document:**
   - [ ] Does manual calculation match cuBLAS?
   - [ ] What are the actual values?
   - [ ] Test passes with current code?

4. **Compare Against llama.cpp Ground Truth**
   ```bash
   # Run llama.cpp with same model and prompt
   cd reference/llama.cpp
   ./llama-cli -m ../../models/qwen2.5-0.5b-instruct.gguf \
     -p "Write a haiku about GPU computing" \
     --log-disable 0 > llama_cpp_output.log 2>&1
   
   # Extract Q[0] value from layer 0, token 0
   grep "Q\[0\]" llama_cpp_output.log
   ```
   
   **Document:**
   - [ ] What is llama.cpp's Q[0] value?
   - [ ] Does it match ALPHA's value?
   - [ ] Does it match SENTINEL's value?

5. **Test Both Approaches**
   
   **Test A: With CUBLAS_OP_N (ALPHA's approach)**
   - [ ] Modify code to use CUBLAS_OP_N
   - [ ] Run haiku test
   - [ ] Document output quality
   
   **Test B: With CUBLAS_OP_T (SENTINEL's approach)**
   - [ ] Modify code to use CUBLAS_OP_T
   - [ ] Run haiku test
   - [ ] Document output quality

6. **Analyze Why FELICIA/AURORA Failed**
   - [ ] Compare their lda values with SENTINEL's
   - [ ] Check if they fixed ALL 8 matmuls consistently
   - [ ] Check if other bugs were present when they tested

### 📊 Deliverable

**File:** `investigation-teams/TEAM_PICASSO_CUBLAS_RESOLUTION.md`

**Format:**
```markdown
# TEAM PICASSO - cuBLAS Contradiction Resolution

## Current State
- Operation: CUBLAS_OP_T / CUBLAS_OP_N
- Applied by: TEAM [name]

## Verification Results

### ALPHA's Test (CUBLAS_OP_N)
- Manual Q[0]: [value]
- cuBLAS Q[0]: [value]
- Diff: [value]
- Status: ✅ PASS / ❌ FAIL

### SENTINEL's Test (CUBLAS_OP_T)
- Manual Q[0]: [value]
- cuBLAS Q[0]: [value]
- Diff: [value]
- Status: ✅ PASS / ❌ FAIL

### llama.cpp Ground Truth
- Q[0]: [value]
- Matches ALPHA: ✅ / ❌
- Matches SENTINEL: ✅ / ❌

## End-to-End Test Results

### With CUBLAS_OP_N
- Output: [first 50 chars]
- Quality: ✅ Coherent / ❌ Garbage
- Test: ✅ PASS / ❌ FAIL

### With CUBLAS_OP_T
- Output: [first 50 chars]
- Quality: ✅ Coherent / ❌ Garbage
- Test: ✅ PASS / ❌ FAIL

## Root Cause Analysis

### Why FELICIA/AURORA Failed
- [Analysis of their approach]
- [What was different from SENTINEL]

## Final Verdict
- ✅ CUBLAS_OP_T is correct
- OR ✅ CUBLAS_OP_N is correct
- Reasoning: [explanation]

## Recommendation
- [Which approach to use]
- [Any code changes needed]
```

**Success Criteria:**
- ✅ Both approaches tested with current code
- ✅ llama.cpp ground truth obtained
- ✅ Clear verdict with evidence
- ✅ Explanation of why previous teams conflicted

---

## 🎨 TEAM VAN GOGH - The Weight Inspector

**Specialization:** Weight Verification  
**Mission:** Resolve the output norm weight contradiction (16.75x amplification)

### 📦 Unit of Work: Output Norm Weight Resolution

**Objective:** Determine if 16.75x amplification is intentional or a bug

**Background:**
- LAMINATOR: "16.75x amplification is INTENTIONAL"
- Output Norm Team: "16.75x amplification is a BUG" (fixed by normalizing)

**Tasks:**

1. **Check Current Code State (from TEAM MONET)**
   - [ ] Are weights normalized (mean=1.0)?
   - [ ] Or raw (mean=7.14, max=16.75)?

2. **Extract Weights from GGUF Model**
   ```bash
   # Use gguf-dump or custom tool
   cd bin/worker-orcd
   # Extract output_norm.weight tensor
   # Document first 20 values
   ```
   
   **Document:**
   - [ ] Raw weight values from GGUF
   - [ ] Mean, min, max
   - [ ] Are they normalized in the file?

3. **Check llama.cpp Behavior**
   ```bash
   # Check llama.cpp source code
   cd reference/llama.cpp
   grep -A 10 "output_norm" src/llama-model.cpp
   ```
   
   **Document:**
   - [ ] Does llama.cpp normalize these weights?
   - [ ] Or use them as-is?
   - [ ] What does the code say?

4. **Test Both Approaches**
   
   **Test A: With Normalized Weights (Output Norm Team's approach)**
   - [ ] Ensure weights are normalized (mean=1.0)
   - [ ] Run haiku test
   - [ ] Document output quality
   - [ ] Check for repetitive tokens
   
   **Test B: With Raw Weights (LAMINATOR's approach)**
   - [ ] Use raw weights (mean=7.14, max=16.75)
   - [ ] Run haiku test
   - [ ] Document output quality
   - [ ] Check for repetitive tokens

5. **Compare Hidden State Ranges**
   
   **With normalized weights:**
   - [ ] Dump hidden states after output_norm
   - [ ] Document min, max, mean
   
   **With raw weights:**
   - [ ] Dump hidden states after output_norm
   - [ ] Document min, max, mean
   
   **Compare with llama.cpp:**
   - [ ] What range does llama.cpp produce?

6. **Analyze Impact on Logits**
   
   **With normalized weights:**
   - [ ] Dump logits (top 20 values)
   - [ ] Document range
   
   **With raw weights:**
   - [ ] Dump logits (top 20 values)
   - [ ] Document range

### 📊 Deliverable

**File:** `investigation-teams/TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md`

**Format:**
```markdown
# TEAM VAN GOGH - Output Norm Weight Resolution

## Current State
- Weights: Normalized / Raw
- Applied by: TEAM [name]

## GGUF Model Analysis
- Raw weights (first 20): [values]
- Mean: [value]
- Min: [value]
- Max: [value]
- Normalized in file: ✅ / ❌

## llama.cpp Behavior
- Code location: [file:line]
- Normalizes weights: ✅ / ❌
- Uses raw: ✅ / ❌

## End-to-End Test Results

### With Normalized Weights (mean=1.0)
- Output: [first 50 chars]
- Quality: ✅ Coherent / ❌ Garbage / ⚠️ Repetitive
- Hidden state range: [min, max]
- Logit range: [min, max]
- Test: ✅ PASS / ❌ FAIL

### With Raw Weights (mean=7.14, max=16.75)
- Output: [first 50 chars]
- Quality: ✅ Coherent / ❌ Garbage / ⚠️ Repetitive
- Hidden state range: [min, max]
- Logit range: [min, max]
- Test: ✅ PASS / ❌ FAIL

## llama.cpp Ground Truth
- Hidden state range: [min, max]
- Logit range: [min, max]

## Final Verdict
- ✅ Normalized weights are correct
- OR ✅ Raw weights are correct
- Reasoning: [explanation]

## Recommendation
- [Which approach to use]
- [Any code changes needed]
```

**Success Criteria:**
- ✅ Both approaches tested
- ✅ llama.cpp behavior verified
- ✅ Clear verdict with evidence
- ✅ Impact on output quality documented

---

## 📝 TEAM SHAKESPEARE - The Integration Validator

**Specialization:** End-to-End Testing  
**Mission:** Validate complete pipeline with all fixes applied

### 📦 Unit of Work: End-to-End Validation

**Objective:** Determine if the model NOW generates correct output with all fixes

**Tasks:**

1. **Wait for TEAM MONET's Report**
   - [ ] Confirm all 6 fixes are applied
   - [ ] If any missing, wait for them to be applied

2. **Run Haiku Test**
   ```bash
   cd bin/worker-orcd
   REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
     test_haiku_generation_stub_pipeline_only \
     --features cuda --release \
     -- --ignored --nocapture --test-threads=1
   ```
   
   **Document:**
   - [ ] Test result (PASS/FAIL)
   - [ ] Generated output (full text)
   - [ ] Minute word found? (Y/N)
   - [ ] Output quality (coherent/garbage/repetitive)

3. **Run Multiple Times (Repeatability)**
   ```bash
   # Run 5 times at different minutes
   for i in {1..5}; do
     REQUIRE_REAL_LLAMA=1 cargo test ... >> test_run_$i.log 2>&1
     sleep 60  # Wait for minute to change
   done
   ```
   
   **Document:**
   - [ ] Pass rate (X/5)
   - [ ] Output consistency
   - [ ] Any patterns in failures?

4. **Compare with llama.cpp**
   ```bash
   # Run llama.cpp with same prompt
   cd reference/llama.cpp
   ./llama-cli -m ../../models/qwen2.5-0.5b-instruct.gguf \
     -p "Write a haiku about GPU computing" \
     -n 100 --temp 0.7 --top-k 0 --top-p 1.0
   ```
   
   **Document:**
   - [ ] llama.cpp output
   - [ ] Quality comparison
   - [ ] Token-by-token similarity?

5. **Test Different Prompts**
   ```bash
   # Test with 3 different prompts
   prompts=(
     "Write a haiku about GPU computing"
     "Explain quantum physics in simple terms"
     "Write a short story about a robot"
   )
   ```
   
   **Document:**
   - [ ] All prompts produce coherent output?
   - [ ] Any prompt-specific issues?

6. **Performance Metrics**
   - [ ] Tokens per second
   - [ ] Memory usage
   - [ ] GPU utilization

### 📊 Deliverable

**File:** `investigation-teams/TEAM_SHAKESPEARE_INTEGRATION_REPORT.md`

**Format:**
```markdown
# TEAM SHAKESPEARE - End-to-End Integration Report

## Prerequisites
- All fixes applied: ✅ / ❌
- Missing fixes: [list if any]

## Haiku Test Results

### Single Run
- Result: ✅ PASS / ❌ FAIL
- Output: [full text]
- Minute word: [word] - Found: ✅ / ❌
- Quality: ✅ Coherent / ❌ Garbage / ⚠️ Repetitive

### Repeatability (5 runs)
- Pass rate: X/5
- Outputs: [summary of all 5]
- Consistency: ✅ High / ⚠️ Medium / ❌ Low

## Comparison with llama.cpp
- Our output: [text]
- llama.cpp output: [text]
- Similarity: ✅ High / ⚠️ Medium / ❌ Low
- Quality match: ✅ / ❌

## Different Prompts Test
| Prompt | Output Quality | Coherent? |
|--------|---------------|-----------|
| Haiku | [summary] | ✅ / ❌ |
| Quantum | [summary] | ✅ / ❌ |
| Robot story | [summary] | ✅ / ❌ |

## Performance Metrics
- Tokens/sec: [value]
- Memory: [value] GB
- GPU util: [value]%

## Final Verdict
- ✅ ALL BUGS FIXED - Model works correctly!
- OR ❌ BUGS REMAIN - See issues below

## Issues Found (if any)
- [List any remaining issues]

## Recommendation
- [Next steps]
```

**Success Criteria:**
- ✅ Test run at least 5 times
- ✅ Comparison with llama.cpp done
- ✅ Multiple prompts tested
- ✅ Clear verdict on whether bugs are fixed

---

## 📝 TEAM FROST - The Sampling Validator

**Specialization:** Sampling Verification  
**Mission:** Verify softmax and sampling fixes are working correctly

### 📦 Unit of Work: Sampling Validation

**Objective:** Confirm CASCADE's softmax fix and HELIOS's sampling order fix are working

**Tasks:**

1. **Verify Softmax Output**
   ```cpp
   // Add instrumentation to cuda/kernels/sampling.cu
   // After softmax computation, dump:
   // - Sum of all probabilities (should be 1.0)
   // - First 20 probabilities (should all be > 0)
   // - Number of zero probabilities (should be 0)
   ```
   
   **Document:**
   - [ ] Sum of probabilities: [value] (expected: 1.0)
   - [ ] First 20 probs: [values]
   - [ ] Zero count: [value] (expected: 0)
   - [ ] All 151,936 probs > 0? ✅ / ❌

2. **Verify Sampling Order**
   ```cpp
   // Check cuda/kernels/sampling_wrapper.cu
   // Verify order is:
   // 1. Temperature scale
   // 2. Top-K
   // 3. Softmax
   // 4. Top-P (if enabled)
   // 5. Sample
   ```
   
   **Document:**
   - [ ] Current order: [list steps]
   - [ ] Matches HELIOS's fix? ✅ / ❌
   - [ ] Top-P position: Before / After softmax

3. **Test Temperature Scaling**
   ```bash
   # Run with different temperatures
   temps=(0.1 0.5 0.7 1.0 1.5)
   for temp in "${temps[@]}"; do
     # Modify test to use this temperature
     # Run and document output diversity
   done
   ```
   
   **Document:**
   - [ ] temp=0.1: [output diversity]
   - [ ] temp=0.5: [output diversity]
   - [ ] temp=0.7: [output diversity]
   - [ ] temp=1.0: [output diversity]
   - [ ] temp=1.5: [output diversity]
   - [ ] Behavior matches expectations? ✅ / ❌

4. **Test Top-K Filtering**
   ```bash
   # Run with different top-k values
   topks=(1 10 50 100 0)  # 0 = disabled
   for topk in "${topks[@]}"; do
     # Run and document token selection
   done
   ```
   
   **Document:**
   - [ ] top-k=1: Always selects highest prob? ✅ / ❌
   - [ ] top-k=10: Selects from top 10? ✅ / ❌
   - [ ] top-k=0: No filtering? ✅ / ❌

5. **Compare Token Selection with llama.cpp**
   ```bash
   # Run both with SAME seed, prompt, temp, top-k
   # Compare first 20 generated tokens
   ```
   
   **Document:**
   - [ ] Our tokens: [list]
   - [ ] llama.cpp tokens: [list]
   - [ ] Exact match? ✅ / ❌
   - [ ] Similar distribution? ✅ / ❌

6. **Verify No Underflow**
   ```cpp
   // Check for any probability being exactly 0.0
   // (except for tokens filtered by top-k)
   ```
   
   **Document:**
   - [ ] Any underflow detected? ✅ / ❌
   - [ ] Minimum non-zero prob: [value]
   - [ ] Expected minimum: ~1/151936 = 6.6e-6

### 📊 Deliverable

**File:** `investigation-teams/TEAM_FROST_SAMPLING_REPORT.md`

**Format:**
```markdown
# TEAM FROST - Sampling Validation Report

## Softmax Verification
- Sum of probs: [value] (expected: 1.0)
- Diff from 1.0: [value]
- First 20 probs: [values]
- Zero count: [value] (expected: 0)
- All probs > 0: ✅ / ❌
- CASCADE's fix working: ✅ / ❌

## Sampling Order
- Current order:
  1. [step]
  2. [step]
  3. [step]
  4. [step]
  5. [step]
- Matches HELIOS's fix: ✅ / ❌
- Top-P position: After softmax ✅ / Before softmax ❌

## Temperature Scaling Test
| Temp | Output Diversity | Expected Behavior | Match? |
|------|-----------------|-------------------|--------|
| 0.1 | [low/med/high] | Low (peaked) | ✅ / ❌ |
| 0.7 | [low/med/high] | Medium | ✅ / ❌ |
| 1.5 | [low/med/high] | High (flat) | ✅ / ❌ |

## Top-K Filtering Test
| Top-K | Behavior | Expected | Match? |
|-------|----------|----------|--------|
| 1 | [description] | Always max | ✅ / ❌ |
| 10 | [description] | Top 10 only | ✅ / ❌ |
| 0 | [description] | No filter | ✅ / ❌ |

## Comparison with llama.cpp
- Our tokens: [list]
- llama.cpp tokens: [list]
- Exact match: ✅ / ❌
- Distribution similar: ✅ / ❌
- Explanation: [why different/same]

## Underflow Check
- Underflow detected: ✅ / ❌
- Min non-zero prob: [value]
- Expected min: 6.6e-6
- Within range: ✅ / ❌

## Final Verdict
- ✅ Softmax fix working correctly
- ✅ Sampling order correct
- ✅ No underflow issues
- OR ❌ Issues found: [list]

## Recommendation
- [Next steps if issues found]
```

**Success Criteria:**
- ✅ Softmax sum verified = 1.0
- ✅ No zero probabilities (underflow)
- ✅ Sampling order verified
- ✅ Temperature/top-k behavior correct

---

## 📝 TEAM DICKINSON - The Parity Checker

**Specialization:** Hidden State Verification  
**Mission:** Compare hidden states with llama.cpp layer-by-layer

### 📦 Unit of Work: Hidden State Parity Validation

**Objective:** Find where (if anywhere) our hidden states diverge from llama.cpp

**Tasks:**

1. **Instrument Layer Outputs**
   ```cpp
   // Add to cuda/src/transformer/qwen_transformer.cpp
   // After each layer, dump first 10 hidden state values
   // For token 0 only (to keep output manageable)
   ```
   
   **Layers to instrument:**
   - [ ] After embedding
   - [ ] After layer 0
   - [ ] After layer 5
   - [ ] After layer 10
   - [ ] After layer 15
   - [ ] After layer 20
   - [ ] After layer 23 (final)
   - [ ] After output_norm
   - [ ] After lm_head (logits)

2. **Run Our Implementation**
   ```bash
   cd bin/worker-orcd
   REQUIRE_REAL_LLAMA=1 cargo test ... > our_hidden_states.log 2>&1
   ```
   
   **Extract:**
   - [ ] Hidden states at each checkpoint
   - [ ] Save to structured format (JSON/CSV)

3. **Run llama.cpp with Logging**
   ```bash
   # Modify llama.cpp to dump same checkpoints
   # Or use existing debug output
   cd reference/llama.cpp
   ./llama-cli -m ../../models/qwen2.5-0.5b-instruct.gguf \
     -p "Write a haiku about GPU computing" \
     --log-disable 0 > llama_hidden_states.log 2>&1
   ```
   
   **Extract:**
   - [ ] Hidden states at each checkpoint
   - [ ] Save to structured format (JSON/CSV)

4. **Compare Layer-by-Layer**
   ```python
   # Create comparison script
   import json
   
   our_states = load_json("our_hidden_states.json")
   llama_states = load_json("llama_hidden_states.json")
   
   for layer in layers:
       diff = compute_diff(our_states[layer], llama_states[layer])
       print(f"Layer {layer}: max_diff={diff}")
   ```
   
   **Document:**
   - [ ] Embedding: diff = [value]
   - [ ] Layer 0: diff = [value]
   - [ ] Layer 5: diff = [value]
   - [ ] Layer 10: diff = [value]
   - [ ] Layer 15: diff = [value]
   - [ ] Layer 20: diff = [value]
   - [ ] Layer 23: diff = [value]
   - [ ] Output norm: diff = [value]
   - [ ] Logits: diff = [value]

5. **Identify Divergence Point**
   - [ ] First layer with significant diff (>0.01)?
   - [ ] Does diff accumulate over layers?
   - [ ] Or sudden spike at specific layer?

6. **Analyze Root Cause (if divergence found)**
   - [ ] Check cuBLAS parameters at divergence layer
   - [ ] Check RMSNorm at divergence layer
   - [ ] Check RoPE at divergence layer
   - [ ] Check attention at divergence layer

### 📊 Deliverable

**File:** `investigation-teams/TEAM_DICKINSON_PARITY_REPORT.md`

**Format:**
```markdown
# TEAM DICKINSON - Hidden State Parity Report

## Methodology
- Prompt: "Write a haiku about GPU computing"
- Token analyzed: Token 0
- Checkpoints: 9 (embedding → logits)
- Comparison metric: Max absolute difference

## Layer-by-Layer Comparison

| Checkpoint | Our Values (first 10) | llama.cpp Values (first 10) | Max Diff | Status |
|------------|----------------------|----------------------------|----------|--------|
| Embedding | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Layer 0 | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Layer 5 | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Layer 10 | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Layer 15 | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Layer 20 | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Layer 23 | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Output norm | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |
| Logits | [values] | [values] | [value] | ✅ / ⚠️ / ❌ |

**Legend:**
- ✅ Match (diff < 0.001)
- ⚠️ Small diff (0.001 < diff < 0.01)
- ❌ Large diff (diff > 0.01)

## Divergence Analysis

### First Divergence Point
- Layer: [number]
- Max diff: [value]
- Pattern: Sudden spike / Gradual accumulation

### Root Cause Investigation
- cuBLAS params: ✅ Correct / ❌ Wrong
- RMSNorm: ✅ Correct / ❌ Wrong
- RoPE: ✅ Correct / ❌ Wrong
- Attention: ✅ Correct / ❌ Wrong

## Final Verdict
- ✅ Perfect parity with llama.cpp
- ⚠️ Small differences (within FP16 tolerance)
- ❌ Significant divergence at layer [X]

## Recommendation
- [Next steps if divergence found]
- [Which component to investigate]
```

**Success Criteria:**
- ✅ All 9 checkpoints compared
- ✅ Divergence point identified (if any)
- ✅ Root cause analyzed (if divergence found)
- ✅ Clear verdict on parity

---

## 🎨 TEAM REMBRANDT - The Fix Restorer

**Specialization:** Reverted Fix Re-Application  
**Mission:** Re-apply fixes that were incorrectly reverted

### 📦 Unit of Work: Reverted Fix Restoration

**Objective:** Identify and re-apply any fixes that were reverted but are actually correct

**Tasks:**

1. **Wait for TEAM PICASSO's Report**
   - [ ] If CUBLAS_OP_T is correct but currently CUBLAS_OP_N:
     - Re-apply CUBLAS_OP_T to all 8 matmuls
     - Use SENTINEL's lda values
     - Test and verify

2. **Wait for TEAM VAN GOGH's Report**
   - [ ] If weight normalization is correct but currently raw:
     - Apply normalization
     - Test and verify
   - [ ] If raw weights are correct but currently normalized:
     - Remove normalization
     - Test and verify

3. **Check for Other Reverted Fixes**
   ```bash
   # Search git history for reverted commits
   cd bin/worker-orcd
   git log --all --grep="revert" --oneline
   git log --all --grep="undo" --oneline
   git log --all --grep="rollback" --oneline
   ```
   
   **Document:**
   - [ ] List of reverted commits
   - [ ] Reason for revert
   - [ ] Should they be re-applied?

4. **Re-Apply Fixes (if needed)**
   - [ ] For each fix to restore:
     - Create new branch
     - Apply fix
     - Run tests
     - Document results
     - Merge if successful

5. **Verify No Regressions**
   ```bash
   # After re-applying fixes, run full test suite
   cargo test --features cuda --release
   ```
   
   **Document:**
   - [ ] All tests pass? ✅ / ❌
   - [ ] Any regressions? [list]

### 📊 Deliverable

**File:** `investigation-teams/TEAM_REMBRANDT_RESTORATION_REPORT.md`

**Format:**
```markdown
# TEAM REMBRANDT - Fix Restoration Report

## Fixes to Restore

### 1. CUBLAS_OP_T (if needed)
- Current state: CUBLAS_OP_N
- Should be: CUBLAS_OP_T
- Reason: TEAM PICASSO proved it's correct
- Action taken:
  - [ ] Re-applied to all 8 matmuls
  - [ ] Used lda values: [list]
  - [ ] Tested: ✅ PASS / ❌ FAIL

### 2. Weight Normalization (if needed)
- Current state: [normalized/raw]
- Should be: [normalized/raw]
- Reason: TEAM VAN GOGH proved it's correct
- Action taken:
  - [ ] Applied normalization / Removed normalization
  - [ ] Tested: ✅ PASS / ❌ FAIL

### 3. Other Reverted Fixes
| Commit | Date | Reason for Revert | Should Restore? | Action |
|--------|------|-------------------|-----------------|--------|
| [hash] | [date] | [reason] | ✅ / ❌ | [action taken] |

## Test Results After Restoration
- Haiku test: ✅ PASS / ❌ FAIL
- Full test suite: ✅ PASS / ❌ FAIL
- Regressions: [list if any]

## Final Status
- Fixes restored: X
- Tests passing: ✅ / ❌
- Ready for production: ✅ / ❌

## Recommendation
- [Next steps]
```

**Success Criteria:**
- ✅ All necessary fixes re-applied
- ✅ Tests pass after restoration
- ✅ No regressions introduced

---

## 📝 TEAM WHITMAN - The Documentarian

**Specialization:** Documentation Cleanup  
**Mission:** Update FALSE_LEADS_SUMMARY.md and related docs

### 📦 Unit of Work: Documentation Cleanup

**Objective:** Correct misleading documentation based on new findings

**Tasks:**

1. **Wait for All Other Teams**
   - [ ] TEAM MONET: Code audit complete
   - [ ] TEAM PICASSO: cuBLAS resolved
   - [ ] TEAM VAN GOGH: Weights resolved
   - [ ] TEAM SHAKESPEARE: Integration tested
   - [ ] TEAM FROST: Sampling verified
   - [ ] TEAM DICKINSON: Parity checked
   - [ ] TEAM REMBRANDT: Fixes restored

2. **Update FALSE_LEADS_SUMMARY.md**
   
   **Mark as REAL BUGS (not false leads):**
   - [ ] False Lead #8: CUBLAS_OP_T (if PICASSO confirms)
   - [ ] False Lead #9: Output RMSNorm (if VAN GOGH confirms corruption)
   - [ ] False Lead #12: Softmax (CASCADE already proved this)
   
   **Add new section:**
   ```markdown
   ## ⚠️ POST-FIX UPDATE (2025-10-07)
   
   After fixing multiple bugs (softmax underflow, sampling order, cuBLAS, 
   weights, config), several "false leads" were re-validated and found to 
   be REAL BUGS that were masked by other bugs.
   
   ### False Leads That Were Actually Real Bugs:
   - #8: CUBLAS_OP_T [CONFIRMED BY TEAM PICASSO]
   - #9: Output norm weights [CONFIRMED BY TEAM VAN GOGH]
   - #12: Softmax underflow [CONFIRMED BY TEAM CASCADE]
   
   ### Lesson Learned:
   "Still broken after fix" ≠ "Not a bug"
   Multiple bugs can exist simultaneously. Fixing one doesn't guarantee output is correct.
   ```

3. **Update Team Reports with Warnings**
   
   **Add to TEAM_FELICIA_FINAL.md:**
   ```markdown
   ## ⚠️ POST-FIX UPDATE (2025-10-07)
   
   TEAM PICASSO re-validated this approach and found that CUBLAS_OP_T 
   IS correct. The "stuck repetition" we observed was caused by OTHER bugs 
   (softmax underflow, sampling order) that have since been fixed.
   
   Our fix was CORRECT but INSUFFICIENT alone.
   ```
   
   **Add to TEAM_AURORA_HANDOFF.md:**
   ```markdown
   ## ⚠️ POST-FIX UPDATE (2025-10-07)
   
   TEAM PICASSO confirmed that CUBLAS_OP_T with our lda values IS correct.
   The test failures we observed were due to downstream bugs that have 
   since been fixed.
   ```
   
   **Add to TEAM_SENTINEL_VICTORY.md:**
   ```markdown
   ## ✅ POST-FIX VALIDATION (2025-10-07)
   
   TEAM PICASSO confirmed our fix is correct. The "still garbage" output 
   was caused by OTHER bugs (softmax underflow, sampling order) that have 
   since been fixed by TEAM CASCADE and TEAM HELIOS.
   
   Our fix WAS necessary. It just wasn't sufficient alone.
   ```

4. **Create Summary Document**
   
   **File:** `investigation-teams/POST_FIX_VALIDATION_SUMMARY.md`
   
   **Contents:**
   - Summary of all team findings
   - Which fixes are applied
   - Which contradictions were resolved
   - Current status of the codebase
   - Lessons learned

5. **Update winners.md**
   
   **Add note about cascade effect:**
   ```markdown
   ## 🔄 Important Note About Bug Fixes
   
   These bugs formed a CONSTELLATION - all needed fixing for the model to work.
   
   Several teams (FELICIA, AURORA) found correct fixes but reverted them 
   because output was still broken due to OTHER bugs. This led to confusion 
   about which fixes were correct.
   
   After ALL bugs were fixed, we re-validated previous investigations and 
   found that many "false leads" were actually REAL bugs.
   ```

### 📊 Deliverable

**Files Updated:**
1. `investigation-teams/FALSE_LEADS_SUMMARY.md`
2. `investigation-teams/TEAM_FELICIA_FINAL.md`
3. `investigation-teams/TEAM_AURORA_HANDOFF.md`
4. `investigation-teams/TEAM_SENTINEL_VICTORY.md`
5. `investigation-teams/POST_FIX_VALIDATION_SUMMARY.md` (new)
6. `investigation-teams/winners.md`

**Format for POST_FIX_VALIDATION_SUMMARY.md:**
```markdown
# Post-Fix Validation Summary

**Date:** 2025-10-07  
**Status:** ✅ COMPLETE

## Team Results

### TEAM MONET - Code Audit
- Fixes applied: X/6
- Conflicts found: Y
- Status: [summary]

### TEAM PICASSO - cuBLAS Resolution
- Verdict: CUBLAS_OP_T / CUBLAS_OP_N
- Reasoning: [summary]
- Status: [summary]

### TEAM VAN GOGH - Weight Resolution
- Verdict: Normalized / Raw
- Reasoning: [summary]
- Status: [summary]

### TEAM SHAKESPEARE - Integration
- Test result: PASS / FAIL
- Output quality: [summary]
- Status: [summary]

### TEAM FROST - Sampling
- Softmax working: ✅ / ❌
- Sampling order: ✅ / ❌
- Status: [summary]

### TEAM DICKINSON - Parity
- Divergence found: ✅ / ❌
- Layer: [number if found]
- Status: [summary]

### TEAM REMBRANDT - Restoration
- Fixes restored: X
- Tests passing: ✅ / ❌
- Status: [summary]

## Final Status
- ✅ ALL BUGS FIXED
- OR ❌ BUGS REMAIN: [list]

## Lessons Learned
1. [lesson]
2. [lesson]
3. [lesson]

## Recommendations
- [next steps]
```

**Success Criteria:**
- ✅ All misleading docs updated
- ✅ Warnings added to reverted fix reports
- ✅ Summary document created
- ✅ Clear record of validation process

---

## 📅 Execution Timeline

### Phase 1: Audit & Resolution (Parallel)
**Duration:** 2-4 hours

- 🎨 **TEAM MONET** starts immediately (no dependencies)
- 🎨 **TEAM PICASSO** starts after MONET (needs current state)
- 🎨 **TEAM VAN GOGH** starts after MONET (needs current state)

### Phase 2: Validation (Parallel, after Phase 1)
**Duration:** 2-3 hours

- 📝 **TEAM SHAKESPEARE** starts after MONET (needs fixes confirmed)
- 📝 **TEAM FROST** starts after MONET (needs fixes confirmed)
- 📝 **TEAM DICKINSON** starts after MONET (needs fixes confirmed)

### Phase 3: Restoration (After Phase 1 & 2)
**Duration:** 1-2 hours

- 🎨 **TEAM REMBRANDT** starts after PICASSO & VAN GOGH (needs verdicts)

### Phase 4: Documentation (After all others)
**Duration:** 1 hour

- 📝 **TEAM WHITMAN** starts after all teams complete

**Total Duration:** ~6-10 hours

---

## 🎯 Success Criteria

### Overall Mission Success
- ✅ All 6 fixes verified as applied or not
- ✅ All 3 contradictions resolved
- ✅ End-to-end test passes (or clear diagnosis if fails)
- ✅ Documentation updated and accurate
- ✅ Clear verdict on whether all bugs are fixed

### Individual Team Success
Each team must:
- ✅ Complete all tasks in their unit of work
- ✅ Produce deliverable document
- ✅ Provide clear verdict/recommendation
- ✅ Document evidence for conclusions

---

## 📝 Team Coordination

### Communication
- Each team creates their deliverable document in `investigation-teams/`
- Teams can reference other teams' documents
- No team should block on another team (except documented dependencies)

### Dependencies
```
MONET (no deps)
  ├─> PICASSO (needs MONET)
  ├─> VAN GOGH (needs MONET)
  ├─> SHAKESPEARE (needs MONET)
  ├─> FROST (needs MONET)
  └─> DICKINSON (needs MONET)

PICASSO + VAN GOGH
  └─> REMBRANDT (needs both)

ALL TEAMS
  └─> WHITMAN (needs all)
```

### Conflict Resolution
If teams reach conflicting conclusions:
1. Document the conflict clearly
2. Escalate to next investigation round
3. Don't block on resolution - document and move on

---

## 🏆 Mission Complete Criteria

The investigation is COMPLETE when:

1. ✅ All 8 teams have produced their deliverables
2. ✅ TEAM SHAKESPEARE has clear verdict (PASS/FAIL)
3. ✅ TEAM WHITMAN has updated all documentation
4. ✅ Either:
   - Model works correctly (test passes)
   - OR clear diagnosis of remaining bugs with new investigation plan

---

**Mission Status:** 🚀 READY TO DEPLOY  
**Teams:** 8 specialized units  
**Estimated Duration:** 6-10 hours  
**Expected Outcome:** Clear verdict on bug fix status

---

*"In art and science, specialization reveals truth."*
