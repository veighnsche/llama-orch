# FT-028: Support Llama Integration

**Team**: Foundation-Alpha  
**Sprint**: Sprint 5 - Support + Prep  
**Size**: M (2 days)  
**Days**: 53-54  
**Status**: 📋 Ready for execution

---

## Story Overview

Provide integration support for Llama team as they integrate Llama-specific kernels and adapters with the Foundation layer. This is a reactive support story - tasks will be executed as integration issues are discovered.

---

## Current State Assessment

### ✅ Already Complete
- **Adapter Pattern**: `LlamaInferenceAdapter` implemented in `src/models/adapter.rs`
- **Llama Config**: `LlamaConfig` struct with GQA/MHA detection in `src/model/llama_config.rs`
- **Integration Tests**: Comprehensive test suite in `tests/llama_integration_suite.rs`
- **Qwen Support**: Full Qwen 2.5 implementation with tests
- **Phi-3 Support**: Full Phi-3 implementation with tests

### 🔍 Areas Ready for Support

#### 1. FFI Interface Support
**File**: `src/cuda_ffi/mod.rs`  
**Status**: Stub implementation with TODO markers  
**Support Activities**:
- [ ] Answer questions about FFI boundary design
- [ ] Debug memory ownership issues if reported
- [ ] Clarify error handling patterns
- [ ] Review FFI usage in Llama kernels

**Known TODOs**:
```rust
// Line 102: TODO(ARCH-CHANGE): Implement actual CUDA memcpy
// Line 133: TODO(ARCH-CHANGE): Implement actual CUDA memcpy from device
// Line 152: TODO(ARCH-CHANGE): Implement CUDA deallocation
// Line 195: TODO(ARCH-CHANGE): Implement actual CUDA initialization
// Line 211: TODO(ARCH-CHANGE): Implement actual CUDA allocation
// Line 228: TODO(ARCH-CHANGE): Implement VRAM query
// Line 236: TODO(ARCH-CHANGE): Implement VRAM query
```

#### 2. GGUF Parsing Support
**Files**: `src/model/llama_config.rs`, GGUF loader modules  
**Support Activities**:
- [ ] Debug GGUF metadata extraction issues
- [ ] Validate Llama-specific tensor layouts
- [ ] Support GQA configuration edge cases
- [ ] Clarify RoPE parameter handling

#### 3. VRAM Allocation Support
**Files**: `src/cuda_ffi/mod.rs`, weight loaders  
**Support Activities**:
- [ ] Debug VRAM calculation discrepancies
- [ ] Support KV cache sizing for Llama models
- [ ] Validate total VRAM usage against limits
- [ ] Optimize allocation patterns if needed

#### 4. Integration Test Support
**Files**: `tests/llama_integration_suite.rs`, `tests/adapter_integration.rs`  
**Support Activities**:
- [ ] Add Llama-specific test cases as requested
- [ ] Debug test failures in Llama integration
- [ ] Validate test coverage for Llama kernels
- [ ] Support end-to-end pipeline testing

#### 5. Documentation Support
**Files**: Various README.md, inline docs  
**Support Activities**:
- [ ] Clarify adapter pattern usage
- [ ] Document GQA vs MHA differences
- [ ] Add examples for Llama model loading
- [ ] Update API documentation based on feedback

---

## Proactive Tasks (Can Start Immediately)

### Task 1: Document Adapter Pattern Usage
**File**: Create `docs/ADAPTER_PATTERN_GUIDE.md`  
**Priority**: High  
**Effort**: 1 hour

Document how to use `LlamaInferenceAdapter` for new model types:
- Creating adapters for new models
- Implementing model-specific forward passes
- Handling configuration differences
- Testing adapter implementations

### Task 2: Add Llama-Specific Integration Tests
**File**: `tests/llama_integration_suite.rs`  
**Priority**: Medium  
**Effort**: 2 hours

Add tests for:
- [ ] GQA-specific attention patterns
- [ ] RoPE frequency variations
- [ ] Long context handling (32K tokens)
- [ ] Llama 2 vs Llama 3 differences

### Task 3: Validate FFI Error Handling
**File**: `src/cuda_ffi/mod.rs`  
**Priority**: Medium  
**Effort**: 1 hour

Review and document:
- [ ] Error propagation patterns
- [ ] Memory leak prevention
- [ ] CUDA error handling best practices
- [ ] Safe cleanup in Drop implementations

### Task 4: Create Integration Checklist
**File**: Create `docs/LLAMA_INTEGRATION_CHECKLIST.md`  
**Priority**: Low  
**Effort**: 30 minutes

Checklist for Llama team:
- [ ] Model configuration validated
- [ ] GGUF parsing working
- [ ] VRAM allocation correct
- [ ] Forward pass implemented
- [ ] Tests passing
- [ ] Documentation updated

---

## Reactive Tasks (Wait for Issues)

### Issue Template: FFI Boundary Problem
**Trigger**: Llama team reports FFI issue  
**Response**:
1. Reproduce issue in isolated test
2. Identify root cause (ownership, lifetime, error handling)
3. Implement fix with regression test
4. Document pattern for future reference

### Issue Template: GGUF Parsing Failure
**Trigger**: Llama model fails to load  
**Response**:
1. Examine GGUF file structure
2. Validate metadata extraction
3. Fix parser or add special case
4. Add test with problematic GGUF

### Issue Template: VRAM Calculation Wrong
**Trigger**: VRAM usage doesn't match expectations  
**Response**:
1. Audit VRAM calculation logic
2. Compare with manual calculation
3. Fix calculation or document discrepancy
4. Add validation test

### Issue Template: Integration Test Failure
**Trigger**: Llama integration test fails  
**Response**:
1. Isolate failing component
2. Debug with minimal reproduction
3. Fix issue or update test expectations
4. Verify all tests pass

---

## Communication Protocol

### Daily Standup Questions
1. Any blocking issues from Llama team?
2. Any FFI questions or clarifications needed?
3. Any integration test failures?
4. Any documentation gaps identified?

### Coordination Files
- Update `execution/day-tracker.md` with support activities
- Log issues in `execution/dependencies.md`
- Track fixes in sprint completion summary

---

## Acceptance Criteria

- [ ] All proactive tasks completed
- [ ] Llama team reports no blocking issues
- [ ] All integration tests passing
- [ ] Documentation updated based on feedback
- [ ] FFI interface questions answered
- [ ] GGUF parsing working for Llama models
- [ ] VRAM allocation validated
- [ ] Integration checklist created and validated

---

## Definition of Done

- [ ] Llama team unblocked
- [ ] Integration successful
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Story marked complete in day-tracker.md

---

**Created**: 2025-10-05  
**Owner**: Foundation-Alpha  
**Dependencies**: FT-027 (Gate 1 checkpoint)  
**Blocks**: LT-027 (Llama Gate 2 checkpoint)

---
Built by Foundation-Alpha 🏗️
