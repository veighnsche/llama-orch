# 🎨 TEAM MONET - Code Audit Chronicle

**Round:** 2  
**Specialization:** Current State Verification  
**Mission:** Audit current codebase to determine which fixes are actually applied  
**Status:** 🚀 READY TO START

---

## 👥 Team Introduction

**Team Name:** MONET (after Claude Monet, master of capturing light and detail)

**Why This Name:**
Just as Monet captured fleeting moments of light with precision, TEAM MONET captures the current state of the codebase with meticulous detail. We observe what IS, not what should be.

**Team Philosophy:**
*"We paint the truth of the code as it exists today."*

**Specialization:**
We are the auditors. Before any team can resolve contradictions or validate fixes, they need to know the CURRENT STATE. That's our job. We read the code, document every parameter, every value, every line number. We are the foundation upon which all other teams build.

---

## 📋 Mission Briefing

**Objective:** Create a definitive report of current code state

**Why This Matters:**
Round 1 ended with multiple contradictions:
- FELICIA/AURORA said CUBLAS_OP_T is wrong (reverted)
- SENTINEL said CUBLAS_OP_T is correct (applied)
- Who won? What's in the code NOW?

We need to know the TRUTH before we can proceed.

**Dependencies:**
- None - We start first

**Teams Depending On Us:**
- TEAM PICASSO (needs cuBLAS state)
- TEAM VAN GOGH (needs weight state)
- TEAM SHAKESPEARE (needs all fixes confirmed)
- TEAM FROST (needs sampling state)
- TEAM DICKINSON (needs all fixes confirmed)

---

## 📝 Investigation Log

### Session 1: [Date/Time]

**Investigator:** [Your name/handle]

**What I'm investigating:**
- [ ] cuBLAS parameters (8 matmuls)
- [ ] Softmax implementation
- [ ] Sampling order
- [ ] Output norm weights
- [ ] Q/K/V biases
- [ ] Configuration overrides

**Findings:**

```
[Document your findings here as you work]

Example:
- Checked cuda/src/transformer/qwen_transformer.cpp line 327
- Q projection uses: CUBLAS_OP_T
- lda value: 896
- Last modified: [commit hash or team name if visible in comments]
```

**Questions/Blockers:**

```
[Any questions or blockers you encounter]
```

**Next Steps:**

```
[What you'll investigate next]
```

---

### Session 2: [Date/Time]

**Investigator:** [Your name/handle]

**What I'm investigating:**

**Findings:**

**Questions/Blockers:**

**Next Steps:**

---

## 🔍 Detailed Findings

### 1. cuBLAS Parameters Audit

| Matmul | File | Line | Operation | lda | Last Modified By |
|--------|------|------|-----------|-----|------------------|
| Q proj | cuda/src/transformer/qwen_transformer.cpp | ??? | CUBLAS_OP_? | ??? | ??? |
| K proj | cuda/src/transformer/qwen_transformer.cpp | ??? | CUBLAS_OP_? | ??? | ??? |
| V proj | cuda/src/transformer/qwen_transformer.cpp | ??? | CUBLAS_OP_? | ??? | ??? |
| Attn out | cuda/src/transformer/qwen_transformer.cpp | ??? | CUBLAS_OP_? | ??? | ??? |
| FFN gate | cuda/kernels/swiglu_ffn.cu | ??? | CUBLAS_OP_? | ??? | ??? |
| FFN up | cuda/kernels/swiglu_ffn.cu | ??? | CUBLAS_OP_? | ??? | ??? |
| FFN down | cuda/kernels/swiglu_ffn.cu | ??? | CUBLAS_OP_? | ??? | ??? |
| lm_head | cuda/src/transformer/qwen_transformer.cpp | ??? | CUBLAS_OP_? | ??? | ??? |

**Summary:**
- All using CUBLAS_OP_T? ✅ / ❌
- All using CUBLAS_OP_N? ✅ / ❌
- Mixed? ⚠️ (list which)

### 2. Softmax Implementation

**File:** `cuda/kernels/sampling.cu`

**Findings:**
- [ ] Line number: ???
- [ ] Uses double precision for sum? ✅ / ❌
- [ ] CASCADE's fix applied? ✅ / ❌

**Code snippet:**
```cpp
[Paste relevant code here]
```

### 3. Sampling Order

**File:** `cuda/kernels/sampling_wrapper.cu`

**Current order:**
1. ???
2. ???
3. ???
4. ???
5. ???

**Analysis:**
- Top-P before softmax? ✅ / ❌
- Top-P after softmax? ✅ / ❌
- HELIOS's fix applied? ✅ / ❌

### 4. Output Norm Weights

**File:** `cuda/src/model/qwen_weight_loader.cpp`

**Findings:**
- [ ] Line number: ???
- [ ] Weights normalized? ✅ / ❌
- [ ] Raw weights used? ✅ / ❌
- [ ] Output Norm Team's fix applied? ✅ / ❌

**Code snippet:**
```cpp
[Paste relevant code here]
```

### 5. Q/K/V Biases

**Files:**
- `cuda/src/model/qwen_weight_loader.cpp` (loading)
- `cuda/src/transformer/qwen_transformer.cpp` (addition)

**Findings:**
- [ ] Biases loaded (not nullptr)? ✅ / ❌
- [ ] Biases added after projections? ✅ / ❌
- [ ] GREEN's fix applied? ✅ / ❌

**Code snippets:**
```cpp
[Paste relevant code here]
```

### 6. Configuration Overrides

**File:** `src/inference/cuda_backend.rs`

**Findings:**
- [ ] Hardcoded temperature removed? ✅ / ❌
- [ ] Hardcoded system prompt removed? ✅ / ❌
- [ ] FINNEY's fix applied? ✅ / ❌

**Code snippet:**
```rust
[Paste relevant code here]
```

---

## 📊 Summary Report

### Fixes Applied: X/6

- [ ] cuBLAS parameters (CUBLAS_OP_T)
- [ ] Softmax (double precision)
- [ ] Sampling order (Top-P after softmax)
- [ ] Output norm weights (normalized)
- [ ] Q/K/V biases (loaded and added)
- [ ] Config overrides (removed)

### Critical Issues Found

```
[List any missing fixes or conflicts]
```

### Conflicts Detected

```
[List any contradictions in the code]
Example: Q/K/V use CUBLAS_OP_T but FFN uses CUBLAS_OP_N
```

---

## 🎯 Final Verdict

**Current Code State:**
- [Summary of what's actually in the code]

**Recommendation:**
- [What other teams should know]
- [Any immediate concerns]

---

## 📦 Deliverable

**Status:** 🚧 IN PROGRESS / ✅ COMPLETE

**File:** `investigation-teams/TEAM_MONET_CODE_AUDIT.md`

**Handoff To:**
- TEAM PICASSO (cuBLAS state documented)
- TEAM VAN GOGH (weight state documented)
- TEAM SHAKESPEARE (all fixes status documented)
- TEAM FROST (sampling state documented)
- TEAM DICKINSON (all fixes status documented)

---

## 💭 Reflections

**What Went Well:**

**What Was Challenging:**

**Lessons Learned:**

**Advice for Future Teams:**

---

**TEAM MONET**  
*"We paint the truth of the code as it exists today."*

**Chronicle Status:** 🚧 ACTIVE  
**Last Updated:** [Date/Time]
