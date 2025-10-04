# FT-019-EXT-2: Repetition Penalty

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Advanced Sampling  
**Size**: S (1 day)  
**Days**: 4 - 4  
**Spec Ref**: M0-W-1421, GENERATION_PARAMETERS_ANALYSIS.md

---

## Story Description

Implement repetition penalty to reduce repetitive text generation. Penalizes tokens that have already been generated, encouraging more diverse outputs.

**Formula**: 
- If `logits[token] > 0`: `logits[token] /= penalty`
- If `logits[token] <= 0`: `logits[token] *= penalty`

---

## Acceptance Criteria

- [ ] Kernel applies penalty to tokens in history
- [ ] History buffer tracks generated tokens
- [ ] Handles empty history (no penalty applied)
- [ ] Handles full history (all tokens penalized)
- [ ] Unit tests validate penalty application (4+ tests)
- [ ] Integration tests validate with generation loop
- [ ] Performance acceptable (<0.5ms per token)

---

## Technical Details

### Kernel Implementation

```cpp
/**
 * Apply repetition penalty to logits.
 * 
 * Penalizes tokens that appear in generation history.
 * Formula:
 * - If logits[token] > 0: logits[token] /= penalty
 * - If logits[token] <= 0: logits[token] *= penalty
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param history Device pointer to generated token IDs
 * @param history_length Number of tokens in history
 * @param penalty Repetition penalty factor (>1.0 = penalize, <1.0 = encourage)
 */
__global__ void apply_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* history,
    int history_length,
    float penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= vocab_size) {
        return;
    }
    
    // If penalty disabled or no history, skip
    if (penalty == 1.0f || history == nullptr || history_length == 0) {
        return;
    }
    
    // Check if token is in history
    bool in_history = false;
    for (int i = 0; i < history_length; ++i) {
        if (history[i] == idx) {
            in_history = true;
            break;
        }
    }
    
    // Apply penalty if in history
    if (in_history) {
        if (logits[idx] > 0.0f) {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

/**
 * Launch repetition penalty kernel.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param history Device pointer to generated token IDs
 * @param history_length Number of tokens in history
 * @param penalty Repetition penalty factor
 * @param stream CUDA stream
 */
void launch_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* history,
    int history_length,
    float penalty,
    cudaStream_t stream = 0
) {
    // Validate inputs
    if (penalty == 1.0f || history == nullptr || history_length == 0) {
        return;  // No penalty to apply
    }
    
    if (vocab_size <= 0 || logits == nullptr) {
        fprintf(stderr, "Invalid inputs to repetition penalty\n");
        return;
    }
    
    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    apply_repetition_penalty<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, vocab_size, history, history_length, penalty
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Repetition penalty kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}
```

### History Buffer Management

```cpp
class InferenceState {
public:
    // Allocate history buffer
    InferenceState(int max_tokens) : max_tokens_(max_tokens) {
        cudaMalloc(&d_history_, max_tokens * sizeof(int));
        history_length_ = 0;
    }
    
    ~InferenceState() {
        cudaFree(d_history_);
    }
    
    // Add token to history
    void add_token(int token_id) {
        if (history_length_ < max_tokens_) {
            cudaMemcpy(
                d_history_ + history_length_,
                &token_id,
                sizeof(int),
                cudaMemcpyHostToDevice
            );
            history_length_++;
        }
    }
    
    // Get history for penalty application
    const int* history() const { return d_history_; }
    int history_length() const { return history_length_; }
    
private:
    int* d_history_;
    int history_length_;
    int max_tokens_;
};
```

---

## Testing Strategy

### Unit Tests (4 tests)

1. **BasicPenalty**
   - Given: logits=[1.0, 2.0, 3.0], history=[1], penalty=1.5
   - When: apply_repetition_penalty
   - Then: logits[1] reduced by factor of 1.5

2. **NoHistory**
   - Given: logits, history=[], penalty=1.5
   - When: apply_repetition_penalty
   - Then: logits unchanged

3. **FullHistory**
   - Given: logits, history=[all tokens], penalty=1.5
   - When: apply_repetition_penalty
   - Then: all logits penalized

4. **PenaltyDisabled**
   - Given: logits, history=[1, 2, 3], penalty=1.0
   - When: apply_repetition_penalty
   - Then: logits unchanged

### Integration Tests (2 tests)

1. **GenerationWithPenalty**
   - Generate 50 tokens with penalty=1.2
   - Compare repetition rate vs no penalty
   - Verify reduced repetition

2. **PenaltyWithTemperature**
   - Apply temperature + repetition penalty
   - Verify both effects applied correctly

---

## Performance Targets

- **Latency**: <0.5ms per token
- **Memory**: ~4 KB for history (1000 tokens × 4 bytes)
- **Overhead**: <10% of total sampling time

---

## Definition of Done

- [ ] Kernel implemented and tested (4 tests)
- [ ] History buffer management implemented
- [ ] Integration tests passing (2 tests)
- [ ] Performance within budget (<0.5ms)
- [ ] Documentation updated
- [ ] Code reviewed (self-review)

---

## References

- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`
- **llama.cpp implementation**: Reference for penalty formula

---
Built by Foundation-Alpha 🏗️
