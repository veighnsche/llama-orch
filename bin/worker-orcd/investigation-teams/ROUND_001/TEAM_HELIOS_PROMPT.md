# 🛰️ **TEAM HELIOS — Next Investigation Prompt**

**Date:** 2025-10-08
**Focus:** FP16 model — Sampling & generation logic (post-logits phase)
**Predecessor:** Team AEGIS
**Status:** All 8 matmuls fixed & verified, weight loading matches llama.cpp, output still mojibake.

---

## 📝 **Summary of Current State**

### ✅ Confirmed Fixed

* ✅ Q4_K dequantization (scale/min decoding + formula)
* ✅ Weight loading parity checks for FP16 tensors (Team VANGUARD + SENTINEL)
* ✅ All 8 matrix multiplications corrected (Team SENTINEL)
* ✅ lm_head cuBLAS parameters reverted correctly (Team AEGIS final)
* ✅ UTF-8 decoding is correct (Team AEGIS byte-level instrumentation)
* ✅ Prefill temperature = 0.0 is expected behavior

### ❌ Still Broken

* ❌ Model generates **mojibake / repetitive tokens**
* ❌ “eight” sometimes appears but not deterministically → indicates **partial parity** but **sampling or logits bug** remains
* ❌ Top-10 logits are sane, but **temperature during generation** was not properly logged — false lead

---

## 🚨 **Mission Focus for Team HELIOS**

The remaining bug is **almost certainly downstream of logits**:

* Sampling temperature handling
* RNG seeding / deterministic sampling paths
* Top-p / top-k cutoff logic
* Argmax vs sampling mismatch between llama.cpp and our implementation

---

## 🧭 **Investigation Plan**

### 1. **Temperature & Sampling Parity**

* Instrument **first 20 generated tokens** (after prefill!)
* Log temperature, top_k, top_p, RNG seed, and sampled token ID at each step.
* Run **llama.cpp with the same model + same prompt + temperature 0.7** and capture its first 20 tokens.
* Compare **token IDs** and **logits distributions** step by step.

👉 **Goal:** Confirm whether our logits or sampling diverge first.

---

### 2. **Top-p / Top-k Implementation Check**

* Cross-read `cuda_sample_token` against llama.cpp’s sampling implementation.
* Check:

  * Is top-p cumulative distribution correct?
  * Are logits sorted or truncated the same way?
  * Is RNG seeded identically?
  * Are tokens masked the same way (e.g., special tokens, BOS/EOS)?

---

### 3. **Argmax vs Probabilistic**

* Test with:

  * `temperature = 0.0` (pure argmax)
  * `temperature = 0.7`
  * `temperature = 1.0`

Run haiku test for each and compare:

* If `temperature=0.0` matches llama.cpp, logits are likely correct and **sampling is the culprit**.
* If it still diverges, there’s still a **logits / softmax bug** lurking.

---

## 📚 **Rules & Expectations**

* 🧱 **APPEND-ONLY COMMENTS** — Never delete or overwrite previous team notes.
* 📝 Use `SUSPECT`, `PLAN`, `OBSERVED`, `FALSE_FIX`, `LESSON` consistently.
* 🚫 Do **not** retest cuBLAS or weight loading — they’re verified.
* 📊 Always **compare against llama.cpp ground truth**, not just internal math.
* 🧪 Focus on **generation phase**, not prefill.

---

## 📎 Useful Tools

* ✅ `investigation-teams/llama_cpp_weight_dumper.cpp` — for weight parity
* ✅ Logit/top-k logging added by Team AEGIS in `ffi_inference.cpp`
* ✅ Byte-level token logging in `cuda_backend.rs`

---

## 🎯 **Success Criteria**

Team HELIOS succeeds if it can demonstrate **first 20 generated token IDs** match llama.cpp at `temperature=0.7`.
Even partial matching with clear divergence point is a success if logged cleanly.
