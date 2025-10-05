# FT-030: Bug Fixes and Integration Cleanup

**Team**: Foundation-Alpha  
**Sprint**: Sprint 5 - Support + Prep  
**Size**: L (3 days)  
**Days**: 57-59  
**Status**: 📋 Ready for execution

---

## Story Overview

Address bugs discovered during Llama/GPT integration, refine interfaces based on usage patterns, and perform general cleanup before Gate 2. This story consolidates all integration feedback and prepares the Foundation layer for the next sprint.

---

## Bug Categories

### 1. FFI Boundary Issues
**Priority**: Critical  
**Files**: `src/cuda_ffi/mod.rs`

**Known Issues**:
- [ ] Memory leaks in CUDA allocation/deallocation
- [ ] Error handling gaps in FFI calls
- [ ] Lifetime issues with `SafeCudaPtr`
- [ ] Missing null pointer checks

**Fix Strategy**:
1. Audit all FFI functions for memory safety
2. Add comprehensive error handling
3. Implement Drop correctly for all CUDA resources
4. Add FFI boundary tests

**Verification**:
- Run with Valgrind/AddressSanitizer
- Test all error paths
- Verify no memory leaks
- Check CUDA error codes

### 2. CUDA Context Management
**Priority**: High  
**Files**: `src/cuda_ffi/mod.rs`

**Potential Issues**:
- [ ] Context initialization race conditions
- [ ] Device selection logic
- [ ] cuBLAS handle management
- [ ] Context cleanup on shutdown

**Fix Strategy**:
1. Review context lifecycle
2. Add thread-safety if needed
3. Implement proper cleanup
4. Document context usage patterns

### 3. VRAM Enforcement Edge Cases
**Priority**: High  
**Files**: `src/cuda_ffi/mod.rs`, weight loaders

**Potential Issues**:
- [ ] VRAM calculation inaccuracies
- [ ] Allocation failure handling
- [ ] KV cache size edge cases
- [ ] Fragmentation issues

**Fix Strategy**:
1. Audit VRAM calculation logic
2. Test with various model sizes
3. Handle allocation failures gracefully
4. Add VRAM pressure tests

### 4. KV Cache Management
**Priority**: Medium  
**Files**: Model implementations

**Potential Issues**:
- [ ] Cache size calculation errors
- [ ] Cache invalidation bugs
- [ ] Memory alignment issues
- [ ] Cache reuse logic

**Fix Strategy**:
1. Review cache sizing formulas
2. Test cache behavior under load
3. Validate alignment requirements
4. Add cache management tests

### 5. Sampling Kernel Edge Cases
**Priority**: Medium  
**Files**: Sampling kernel implementations

**Potential Issues**:
- [ ] Temperature edge cases (0.0, very high)
- [ ] Top-k/top-p boundary conditions
- [ ] Seed reproducibility issues
- [ ] Numerical stability

**Fix Strategy**:
1. Test extreme parameter values
2. Validate reproducibility
3. Check numerical precision
4. Add edge case tests

### 6. Error Propagation
**Priority**: Medium  
**Files**: All modules

**Potential Issues**:
- [ ] Errors swallowed silently
- [ ] Unhelpful error messages
- [ ] Missing context in errors
- [ ] Panic instead of Result

**Fix Strategy**:
1. Audit all error paths
2. Add context to errors
3. Convert panics to Results
4. Improve error messages

---

## Code Cleanup Tasks

### Task 1: Remove TODO Markers
**Priority**: High  
**Effort**: 2 hours

Address or document all TODO markers:
```bash
# Find all TODOs
grep -r "TODO" src/ --include="*.rs"

# Categories:
# - ARCH-CHANGE: Document as future work
# - FIXME: Fix immediately
# - TODO: Implement or remove
```

**Action Plan**:
- [ ] Review all 7 TODOs in `src/cuda_ffi/mod.rs`
- [ ] Document ARCH-CHANGE items in `docs/FUTURE_WORK.md`
- [ ] Fix or remove other TODOs
- [ ] Add tracking issues for deferred work

### Task 2: Code Style Consistency
**Priority**: Medium  
**Effort**: 1 hour

Ensure consistent code style:
- [ ] Run `cargo fmt` on all files
- [ ] Fix clippy warnings
- [ ] Consistent error handling patterns
- [ ] Consistent naming conventions

### Task 3: Documentation Cleanup
**Priority**: Medium  
**Effort**: 2 hours

Update documentation:
- [ ] Fix outdated comments
- [ ] Add missing doc comments
- [ ] Update README files
- [ ] Document integration patterns

### Task 4: Test Organization
**Priority**: Low  
**Effort**: 1 hour

Organize test files:
- [ ] Group related tests
- [ ] Remove duplicate tests
- [ ] Add test documentation
- [ ] Improve test names

---

## Integration Pain Points (From Feedback)

### Pain Point 1: Adapter Pattern Confusion
**Symptom**: Teams unsure how to extend adapter  
**Fix**:
- [ ] Create `docs/ADAPTER_PATTERN_GUIDE.md`
- [ ] Add code examples
- [ ] Document design decisions
- [ ] Add FAQ section

### Pain Point 2: FFI Interface Unclear
**Symptom**: Memory safety questions  
**Fix**:
- [ ] Document ownership rules
- [ ] Add FFI usage examples
- [ ] Create FFI best practices guide
- [ ] Add more FFI tests

### Pain Point 3: VRAM Calculation Opaque
**Symptom**: Unexpected VRAM usage  
**Fix**:
- [ ] Document calculation formulas
- [ ] Add VRAM breakdown logging
- [ ] Create VRAM debugging guide
- [ ] Add VRAM validation tests

### Pain Point 4: Error Messages Unhelpful
**Symptom**: Hard to debug failures  
**Fix**:
- [ ] Add context to all errors
- [ ] Include relevant values in messages
- [ ] Add troubleshooting hints
- [ ] Improve error documentation

---

## Performance Issues (If Reported)

### Issue 1: Slow Model Loading
**Investigation**:
- [ ] Profile GGUF parsing
- [ ] Profile weight transfer
- [ ] Check for unnecessary copies
- [ ] Optimize hot paths

### Issue 2: High Memory Usage
**Investigation**:
- [ ] Audit allocations
- [ ] Check for leaks
- [ ] Optimize data structures
- [ ] Add memory profiling

### Issue 3: Slow Inference
**Investigation**:
- [ ] Profile kernel execution
- [ ] Check for synchronization overhead
- [ ] Validate kernel parameters
- [ ] Optimize critical paths

---

## Regression Prevention

### Add Regression Tests
**Priority**: High  
**Effort**: 3 hours

For each bug fixed:
1. Create minimal reproduction test
2. Verify test fails before fix
3. Verify test passes after fix
4. Add test to CI suite

**Test Categories**:
- [ ] FFI boundary tests
- [ ] VRAM enforcement tests
- [ ] Error handling tests
- [ ] Edge case tests

### Update CI Pipeline
**Priority**: Medium  
**Effort**: 1 hour

Strengthen CI:
- [ ] Add memory leak detection
- [ ] Add sanitizer runs
- [ ] Increase test coverage
- [ ] Add performance benchmarks

---

## Documentation Updates

### Update Based on Integration Feedback
**Priority**: High  
**Effort**: 2 hours

- [ ] `README.md`: Add integration examples
- [ ] `INTEGRATION_TEST_FRAMEWORK.md`: Update patterns
- [ ] `docs/FFI_GUIDE.md`: Add best practices
- [ ] `docs/TROUBLESHOOTING.md`: Add common issues

### Create New Documentation
**Priority**: Medium  
**Effort**: 2 hours

- [ ] `docs/ADAPTER_PATTERN_GUIDE.md`
- [ ] `docs/VRAM_DEBUGGING_GUIDE.md`
- [ ] `docs/INTEGRATION_CHECKLIST.md`
- [ ] `docs/FUTURE_WORK.md`

---

## Acceptance Criteria

- [ ] All reported bugs fixed
- [ ] All TODO markers addressed
- [ ] Code style consistent
- [ ] Documentation updated
- [ ] Regression tests added
- [ ] All tests passing
- [ ] No clippy warnings
- [ ] No memory leaks detected
- [ ] Integration pain points addressed
- [ ] Performance issues resolved (if any)

---

## Definition of Done

- [ ] All bugs resolved
- [ ] Code cleanup complete
- [ ] Documentation updated
- [ ] Tests passing
- [ ] CI green
- [ ] Story marked complete in day-tracker.md

---

## Verification Checklist

```bash
# Run all checks
cargo fmt --check
cargo clippy -- -D warnings
cargo test --all-features
cargo test --test '*integration*'

# Check for memory leaks (if available)
valgrind --leak-check=full target/debug/worker-orcd

# Check for undefined behavior (if available)
RUSTFLAGS="-Z sanitizer=address" cargo test
```

---

**Created**: 2025-10-05  
**Owner**: Foundation-Alpha  
**Dependencies**: FT-028, FT-029  
**Blocks**: Sprint 6 (Adapter + Gate 3)

---
Built by Foundation-Alpha 🏗️
