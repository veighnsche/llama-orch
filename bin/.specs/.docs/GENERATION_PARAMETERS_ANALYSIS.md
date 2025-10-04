# Generation Parameters Analysis: M0 vs Industry Standards

**Date**: 2025-10-04  
**Status**: Analysis Complete  
**Context**: User question about missing generation parameters compared to OpenAI API

---

## Executive Summary

**Current M0 Scope**: Only `temperature` (0.0-2.0) is implemented as a generation parameter.

**Industry Standard (OpenAI, Anthropic, etc.)**: 10+ generation parameters including:
- `temperature`
- `top_p` (nucleus sampling)
- `top_k`
- `frequency_penalty`
- `presence_penalty`
- `repetition_penalty`
- `min_p`
- `stop` sequences
- `logit_bias`
- `max_tokens`

**Gap**: M0 is missing 9+ standard generation parameters that users expect from LLM APIs.

---

## 1. Current M0 State

### 1.1 Implemented Parameters (M0-W-1300)

```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku about GPU computing",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

**Parameters**:
1. ✅ `temperature` (0.0-2.0) — Temperature scaling for sampling
2. ✅ `max_tokens` (1-2048) — Maximum output tokens
3. ✅ `seed` (uint64) — RNG seed for reproducibility

**Total**: 3 parameters (2 generation parameters + 1 seed)

### 1.2 Planned But Not Exposed (GAP_ANALYSIS.md)

The old CUDA specs and GAP_ANALYSIS.md reference `top_k` and `top_p` in the `InferenceConfig` struct:

```cpp
struct InferenceConfig {
    int max_tokens;
    float temperature;
    uint64_t seed;
    int top_k = 50;      // For future use
    float top_p = 0.95f; // For future use
};
```

**Status**: Defined in internal structs but NOT exposed in HTTP API, NOT implemented in sampling kernels.

---

## 2. Industry Standard: OpenAI Chat Completions API

### 2.1 OpenAI Generation Parameters

Based on OpenAI's Chat Completions API (as of 2024):

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `temperature` | float | 0.0-2.0 | Sampling temperature |
| `top_p` | float | 0.0-1.0 | Nucleus sampling (cumulative probability cutoff) |
| `frequency_penalty` | float | -2.0 to 2.0 | Penalize tokens based on frequency in output |
| `presence_penalty` | float | -2.0 to 2.0 | Penalize tokens based on presence in output |
| `max_tokens` | int | 1-∞ | Maximum output tokens |
| `stop` | string[] | - | Stop sequences (up to 4) |
| `logit_bias` | map | -100 to 100 | Bias specific tokens |
| `seed` | int | - | Deterministic sampling (beta) |
| `n` | int | 1-∞ | Number of completions |
| `stream` | bool | - | SSE streaming |

**Total**: 10 parameters

### 2.2 Additional Parameters in Other APIs

**Anthropic Claude**:
- `top_k` — Top-k sampling (select from top k tokens)

**llama.cpp / Ollama / LM Studio**:
- `top_k` — Top-k sampling
- `top_p` — Nucleus sampling
- `repeat_penalty` / `repetition_penalty` — Penalize repeated tokens
- `min_p` — Minimum probability threshold
- `tfs_z` — Tail-free sampling
- `typical_p` — Locally typical sampling
- `mirostat` — Mirostat sampling (adaptive temperature)

**HuggingFace Transformers**:
- `do_sample` — Enable sampling vs greedy
- `top_k`
- `top_p`
- `temperature`
- `repetition_penalty`
- `length_penalty`
- `no_repeat_ngram_size`

---

## 3. Gap Analysis: M0 vs Industry

### 3.1 Missing Core Parameters

| Parameter | Priority | Used By | M0 Status |
|-----------|----------|---------|-----------|
| `top_p` | **HIGH** | OpenAI, Anthropic, llama.cpp, HF | ❌ Missing |
| `top_k` | **HIGH** | Anthropic, llama.cpp, HF | ❌ Missing |
| `frequency_penalty` | **MEDIUM** | OpenAI | ❌ Missing |
| `presence_penalty` | **MEDIUM** | OpenAI | ❌ Missing |
| `repetition_penalty` | **MEDIUM** | llama.cpp, HF | ❌ Missing |
| `stop` sequences | **HIGH** | OpenAI, Anthropic, llama.cpp | ❌ Missing |
| `logit_bias` | **LOW** | OpenAI | ❌ Missing |
| `min_p` | **LOW** | llama.cpp | ❌ Missing |
| `mirostat` | **LOW** | llama.cpp | ❌ Missing |

### 3.2 Impact Assessment

**User Expectations**:
- Users coming from OpenAI/Anthropic/llama.cpp expect `top_p` and `top_k` as standard
- `stop` sequences are critical for structured output (JSON, code, etc.)
- Penalty parameters are important for controlling repetition and diversity

**Current M0 Limitations**:
- ❌ Cannot do nucleus sampling (top_p)
- ❌ Cannot do top-k sampling
- ❌ Cannot control repetition
- ❌ Cannot use stop sequences
- ❌ Only temperature-based sampling

**Competitive Gap**:
- LM Studio: 8+ parameters
- Ollama: 10+ parameters
- llama.cpp: 12+ parameters
- **M0**: 2 parameters (temperature, max_tokens)

---

## 4. Why M0 Only Has Temperature

### 4.1 Scope Decision (Hybrid Approach)

From `01_M0_worker_orcd.md` §0.0:

> **Decision Date**: 2025-10-03  
> **Approach**: Performance Bundle Deferral (Hybrid)  
> **Rationale**: Balance faster delivery (4-5 weeks) with critical safety features

**M0 Focus**:
- ✅ Core inference pipeline (model load, tokenization, forward pass, sampling)
- ✅ VRAM-only enforcement
- ✅ Architecture adapters (Llama, GPT)
- ✅ Test reproducibility (seeded RNG, temp=0)
- ✅ Basic sampling (temperature scaling)

**Deferred to M1+**:
- ❌ Advanced sampling strategies (top-p, top-k, penalties)
- ❌ Performance optimization
- ❌ Metrics/observability
- ❌ Graceful shutdown

### 4.2 Current Sprint Focus

From Sprint 3 (Shared Kernels):
- FT-017: Temperature scaling kernel ✅
- FT-018: Greedy sampling (temp=0) ✅
- FT-019: Stochastic sampling (temp>0) ✅
- FT-020: Seeded RNG ✅

**Scope**: Basic temperature-based sampling only.

---

## 5. Recommendations

### 5.1 Short-Term (M0)

**Keep M0 scope minimal**:
- ✅ Temperature (0.0-2.0)
- ✅ Max tokens
- ✅ Seed

**Rationale**:
- M0 is already 6-7 weeks (foundation + architecture adapters)
- Adding 9+ parameters would delay M0 by 2-3 weeks
- Temperature alone is sufficient for basic inference validation

### 5.2 Medium-Term (M1)

**Add high-priority parameters**:
1. **`top_p`** (nucleus sampling) — Industry standard, high user demand
2. **`top_k`** — Common in llama.cpp, Anthropic
3. **`stop`** sequences — Critical for structured output
4. **`repetition_penalty`** — Control repetition

**Implementation**:
- Add to `InferenceConfig` struct (already stubbed)
- Implement sampling kernels (top-k/top-p filtering)
- Add HTTP API parameters
- Add validation

**Estimated effort**: 1-2 weeks

### 5.3 Long-Term (M2+)

**Add advanced parameters**:
- `frequency_penalty` / `presence_penalty` (OpenAI compatibility)
- `logit_bias` (token-level control)
- `min_p`, `mirostat`, `tfs_z`, `typical_p` (advanced sampling)

**Estimated effort**: 2-3 weeks

---

## 6. Technical Implementation Notes

### 6.1 Top-P (Nucleus Sampling)

**Algorithm**:
1. Sort logits descending
2. Compute cumulative probabilities
3. Find cutoff where cumsum >= top_p
4. Zero out logits below cutoff
5. Renormalize and sample

**Complexity**: O(V log V) where V = vocab size (~50k-100k)

**Implementation**:
```cpp
int sample_top_p(const std::vector<float>& logits, float top_p, RNG& rng) {
    // Sort indices by logit value (descending)
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&](int a, int b) { return logits[a] > logits[b]; });
    
    // Compute softmax and cumulative sum
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (int idx : indices) {
        probs[idx] = std::exp(logits[idx]);
        sum += probs[idx];
    }
    for (float& p : probs) p /= sum;
    
    // Find cutoff
    float cumsum = 0.0f;
    std::vector<int> nucleus;
    for (int idx : indices) {
        cumsum += probs[idx];
        nucleus.push_back(idx);
        if (cumsum >= top_p) break;
    }
    
    // Renormalize nucleus
    float nucleus_sum = 0.0f;
    for (int idx : nucleus) nucleus_sum += probs[idx];
    for (int idx : nucleus) probs[idx] /= nucleus_sum;
    
    // Sample from nucleus
    float r = rng.uniform();
    cumsum = 0.0f;
    for (int idx : nucleus) {
        cumsum += probs[idx];
        if (r < cumsum) return idx;
    }
    return nucleus.back();
}
```

### 6.2 Top-K Sampling

**Algorithm**:
1. Sort logits descending
2. Keep top k tokens
3. Zero out rest
4. Renormalize and sample

**Complexity**: O(V log V) or O(V) with partial sort

**Implementation**:
```cpp
int sample_top_k(const std::vector<float>& logits, int top_k, RNG& rng) {
    // Partial sort to find top k
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    
    // Compute softmax over top k
    std::vector<float> probs(top_k);
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp(logits[indices[i]]);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;
    
    // Sample
    float r = rng.uniform();
    float cumsum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        cumsum += probs[i];
        if (r < cumsum) return indices[i];
    }
    return indices[top_k - 1];
}
```

### 6.3 Repetition Penalty

**Algorithm**:
1. Track generated token IDs
2. Apply penalty to logits of already-generated tokens
3. `logit[token] /= penalty` if token in history

**Implementation**:
```cpp
void apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int>& history,
    float penalty
) {
    for (int token : history) {
        if (token >= 0 && token < logits.size()) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}
```

### 6.4 Stop Sequences

**Algorithm**:
1. Maintain sliding window of generated tokens
2. After each token, check if any stop sequence matches
3. If match, terminate generation

**Implementation**:
```cpp
bool check_stop_sequences(
    const std::vector<int>& generated_tokens,
    const std::vector<std::vector<int>>& stop_sequences
) {
    for (const auto& stop_seq : stop_sequences) {
        if (generated_tokens.size() >= stop_seq.size()) {
            auto start = generated_tokens.end() - stop_seq.size();
            if (std::equal(start, generated_tokens.end(), stop_seq.begin())) {
                return true;
            }
        }
    }
    return false;
}
```

---

## 7. API Design Proposal (M1)

### 7.1 Extended Request Schema

```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku about GPU computing",
  
  // Output control
  "max_tokens": 100,
  "stop": ["\n\n", "END"],
  
  // Sampling parameters
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  
  // Reproducibility
  "seed": 42
}
```

### 7.2 Validation Rules

| Parameter | Type | Range | Default | Required |
|-----------|------|-------|---------|----------|
| `temperature` | float | 0.0-2.0 | 1.0 | No |
| `top_p` | float | 0.0-1.0 | 1.0 | No |
| `top_k` | int | 1-vocab_size | 0 (disabled) | No |
| `repetition_penalty` | float | 0.0-2.0 | 1.0 | No |
| `stop` | string[] | max 4 sequences | [] | No |
| `max_tokens` | int | 1-context_length | 2048 | No |
| `seed` | uint64 | - | auto-generated | No |

### 7.3 Backward Compatibility

**M0 → M1 Migration**:
- All new parameters are optional with sensible defaults
- M0 requests (temperature only) continue to work
- New parameters are additive, not breaking

---

## 8. Comparison Table: M0 vs Industry

| Feature | M0 | OpenAI | Anthropic | llama.cpp | LM Studio |
|---------|----|----|-----------|-----------|-----------|
| `temperature` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `max_tokens` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `seed` | ✅ | ✅ (beta) | ❌ | ✅ | ✅ |
| `top_p` | ❌ | ✅ | ✅ | ✅ | ✅ |
| `top_k` | ❌ | ❌ | ✅ | ✅ | ✅ |
| `frequency_penalty` | ❌ | ✅ | ❌ | ❌ | ✅ |
| `presence_penalty` | ❌ | ✅ | ❌ | ❌ | ✅ |
| `repetition_penalty` | ❌ | ❌ | ❌ | ✅ | ✅ |
| `stop` sequences | ❌ | ✅ | ✅ | ✅ | ✅ |
| `logit_bias` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `min_p` | ❌ | ❌ | ❌ | ✅ | ✅ |
| `mirostat` | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Total** | **3** | **10** | **6** | **12** | **13** |

---

## 9. Conclusion

### 9.1 Answer to User's Question

> "What about all the other config that other engines expose to the API. Like I know that the OpenAI API when requesting a chat response you have so many config choices. Are we doing that? Why and why not?"

**Answer**:

**Why M0 only has temperature**:
1. **Scope decision**: M0 is focused on proving the core inference pipeline works (model load, tokenization, forward pass, basic sampling). Adding 9+ generation parameters would delay M0 by 2-3 weeks.
2. **Incremental delivery**: M0 is the foundation. M1 will add high-priority parameters (top-p, top-k, stop sequences).
3. **Test reproducibility**: M0 prioritizes deterministic testing (seeded RNG, temp=0) over production sampling diversity.

**Why we WILL add them (M1+)**:
1. **User expectations**: Users coming from OpenAI/llama.cpp/LM Studio expect top-p, top-k, and stop sequences as standard.
2. **Competitive parity**: All major LLM APIs have 6-13 parameters. M0's 3 parameters is a temporary limitation.
3. **Already planned**: The internal `InferenceConfig` struct already has `top_k` and `top_p` stubbed for future use.

**Timeline**:
- **M0 (current)**: Temperature only (3 params total)
- **M1 (next)**: Add top-p, top-k, stop, repetition_penalty (7 params total)
- **M2+**: Add frequency/presence penalties, logit_bias, advanced sampling (10-12 params total)

### 9.2 Recommendation

**For M0**: Keep current scope (temperature only). Ship M0 in 6-7 weeks.

**For M1**: Add high-priority parameters (top-p, top-k, stop, repetition_penalty) in 1-2 weeks after M0.

**For M2+**: Achieve parity with OpenAI/llama.cpp (10-12 parameters).

---

**End of Analysis**
