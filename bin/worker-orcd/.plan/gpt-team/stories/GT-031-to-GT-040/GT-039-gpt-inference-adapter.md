# GT-039: GPTInferenceAdapter

**Team**: GPT-Gamma  
**Sprint**: Sprint 7 (Adapter + E2E)  
**Size**: L (3 days)  
**Days**: 90-92  
**Spec Ref**: M0-W-1213, M0-W-1214

---

## Story Description

Implement GPTInferenceAdapter following the InferenceAdapter pattern. This adapter orchestrates GPT-specific inference pipeline (LayerNorm, MHA, GELU FFN) and integrates with the architecture detection system.

---

## Acceptance Criteria

- [ ] GPTInferenceAdapter implements InferenceAdapter interface
- [ ] Adapter orchestrates GPT-specific kernels
- [ ] Adapter handles both Q4_K_M and MXFP4 weights
- [ ] Adapter integrates with architecture detection
- [ ] Unit tests validate adapter correctness
- [ ] Integration test validates full inference pipeline
- [ ] Documentation updated with GPT adapter details
- [ ] Ready for Gate 3 validation

---

## Dependencies

### Upstream (Blocks This Story)
- GT-038: MXFP4 Numerical Validation (needs validated MXFP4 pipeline)
- FT-033: InferenceAdapter Interface (needs adapter pattern)

### Downstream (This Story Blocks)
- GT-040: GPT-OSS-20B MXFP4 E2E (needs GPT adapter)
- GT-041: Gate 3 Participation (needs complete adapter)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/adapters/gpt_adapter.cpp` - GPT adapter implementation
- `bin/worker-orcd/cuda/src/adapters/gpt_adapter.h` - GPT adapter interface
- `bin/worker-orcd/src/adapters/gpt_adapter.rs` - Rust GPT adapter wrapper

### Key Interfaces
```cpp
class GPTInferenceAdapter : public InferenceAdapter {
public:
    GPTInferenceAdapter(const GPTConfig& config);
    
    // InferenceAdapter interface
    void load_weights(const std::string& model_path) override;
    void prefill(const std::vector<int>& tokens, InferenceState& state) override;
    int decode_next_token(InferenceState& state, float temperature, uint64_t seed) override;
    void free_state(InferenceState& state) override;
    
private:
    GPTConfig config_;
    GPTWeights weights_;
    // GPT-specific state
};
```

---

## Testing Strategy

### Unit Tests
- Test adapter initialization
- Test weight loading
- Test prefill phase
- Test decode phase

### Integration Tests
- Test full inference pipeline
- Test with GPT-OSS-20B
- Compare with reference

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Ready for Gate 3

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.4
- Gap Analysis: `M0_ARCHITECTURAL_GAP_ANALYSIS.md` (Gap 2)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
