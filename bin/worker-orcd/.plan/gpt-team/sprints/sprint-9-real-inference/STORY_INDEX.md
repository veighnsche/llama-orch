# Sprint 9: Story Index

**Sprint**: Sprint 9 - Real Inference  
**Team**: GPT-Gamma 🤖  
**Total Stories**: 7  
**Total Estimate**: 22-31 hours  
**Deadline**: 2025-10-15

---

## Stories

| ID | Title | Size | Estimate | Status | Dependencies |
|----|-------|------|----------|--------|--------------|
| GT-051 | GGUF Config Parsing | S | 2-3h | TODO | None |
| GT-052 | GGUF Weight Loading | L | 6-8h | TODO | GT-051 |
| GT-053 | BPE Tokenizer | M | 5-7h | TODO | GT-051 |
| GT-054 | Transformer Layer Execution | M | 4-6h | TODO | GT-052 |
| GT-055 | LM Head Implementation | S | 2-3h | TODO | GT-052 |
| GT-056 | Wire Real Inference | S | 2-3h | TODO | GT-052, GT-053, GT-054, GT-055 |
| GT-057 | Test Cleanup & Verification | XS | 1-2h | TODO | GT-056 |

**Total**: 22-31 hours

---

## Execution Order

### Phase 1: Foundation (Day 1-2)
1. **GT-051** - Parse real GGUF config (2-3h)
2. **GT-052** - Load weights to GPU (6-8h)

**Deliverable**: Weights loaded to VRAM, verified with `nvidia-smi`

### Phase 2: Tokenizer (Day 3-4)
3. **GT-053** - BPE tokenizer (5-7h)

**Deliverable**: Encode/decode working

### Phase 3: Inference (Day 5-6)
4. **GT-054** - Transformer execution (4-6h)
5. **GT-055** - LM head (2-3h)

**Deliverable**: Forward pass working

### Phase 4: Integration (Day 7)
6. **GT-056** - Wire it together (2-3h)

**Deliverable**: Real haiku generation

### Phase 5: Cleanup (Day 8)
7. **GT-057** - Test cleanup (1-2h)

**Deliverable**: Fine remediated, Testing Team sign-off

---

## Critical Path

```
GT-051 (config)
  ├─> GT-052 (weights) ──┬─> GT-054 (transformer) ──┐
  │                      └─> GT-055 (LM head) ──────┤
  └─> GT-053 (tokenizer) ────────────────────────────┼─> GT-056 (wire) ─> GT-057 (cleanup)
```

**Parallel work possible**:
- GT-053 can run parallel to GT-052
- GT-054 and GT-055 can run parallel after GT-052

---

## Story Files

- [GT-051: GGUF Config Parsing](stories/GT-051-gguf-config-parsing.md)
- [GT-052: GGUF Weight Loading](stories/GT-052-gguf-weight-loading.md)
- [GT-053: BPE Tokenizer](stories/GT-053-bpe-tokenizer.md)
- [GT-054: Transformer Layer Execution](stories/GT-054-transformer-layer-execution.md)
- [GT-055: LM Head Implementation](stories/GT-055-lm-head-implementation.md)
- [GT-056: Wire Real Inference](stories/GT-056-wire-real-inference.md)
- [GT-057: Test Cleanup & Verification](stories/GT-057-test-cleanup-verification.md)

---

## Success Criteria

### Sprint Complete When:
- [ ] All 7 stories marked DONE
- [ ] Haiku test passes with real inference
- [ ] Different haiku each run
- [ ] Minute word present in haiku
- [ ] No stub warnings
- [ ] Testing Team sign-off received
- [ ] Fine FINE-001-20251005 marked RESOLVED

### Verification:
```bash
# Run test 5 times
for i in {1..5}; do
  REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
    test_haiku_generation_anti_cheat --features cuda --release \
    -- --ignored --nocapture --test-threads=1
done

# Verify:
# - All 5 runs pass
# - Different haiku each time
# - Minute word in each haiku
# - No stub warnings
# - nvidia-smi shows VRAM usage
```

---

## Related Documents

- [Sprint 9 README](README.md) - Sprint overview
- [Fine #001](../../../../test-harness/FINES.md) - Original fine
- [Remediation Summary](../../FINE_REMEDIATION_SUMMARY.md) - Immediate actions
- [Issue Tracking](../../ISSUE_REAL_GPU_INFERENCE.md) - Implementation plan
- [PM Investigation](../../PM_INVESTIGATION_FOUNDATION_TEAM_GAP.md) - Root cause
- [Spec Gap Analysis](../../SPEC_GAP_ANALYSIS.md) - Spec vs implementation

---

**Created by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Status**: Ready for execution
