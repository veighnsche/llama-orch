# FT-029: Support GPT Integration

**Team**: Foundation-Alpha  
**Sprint**: Sprint 5 - Support + Prep  
**Size**: M (2 days)  
**Days**: 55-56  
**Status**: 📋 Ready for execution

---

## Story Overview

Provide integration support for GPT team as they integrate GPT-specific kernels (LayerNorm, GELU, MHA) with the Foundation layer. This is a reactive support story - tasks will be executed as integration issues are discovered.

---

## Current State Assessment

### ✅ Already Complete
- **Adapter Pattern**: `LlamaInferenceAdapter` supports multiple model types
- **FFI Layer**: CUDA FFI interface ready for GPT kernels
- **Integration Test Framework**: Comprehensive testing infrastructure
- **HTTP/SSE Endpoints**: Ready for GPT model integration

### 🔍 Areas Ready for Support

#### 1. GPT-Specific Kernel Integration
**Support Activities**:
- [ ] Support LayerNorm kernel integration
- [ ] Support GELU activation kernel integration
- [ ] Support MHA (Multi-Head Attention) kernel integration
- [ ] Debug kernel launch parameters
- [ ] Validate kernel performance

**Integration Points**:
- FFI boundary for kernel calls
- Memory layout for GPT tensors
- Error handling for kernel failures
- Performance profiling hooks

#### 2. GGUF v3 / MXFP4 Support
**Support Activities**:
- [ ] Debug GGUF v3 parsing for GPT models
- [ ] Support MXFP4 tensor format if needed
- [ ] Validate quantization handling
- [ ] Test weight loading for GPT architectures

**Files to Review**:
- GGUF parser modules
- Weight loader implementations
- Quantization handling code

#### 3. GPT Adapter Implementation
**Support Activities**:
- [ ] Extend `LlamaInferenceAdapter` for GPT models
- [ ] Add `ModelType::GPT2` and `ModelType::GPT3` variants
- [ ] Implement GPT-specific forward pass
- [ ] Add GPT configuration struct

**Example Implementation**:
```rust
// In src/models/adapter.rs
pub enum ModelType {
    Qwen2_5,
    Phi3,
    Llama2,
    Llama3,
    GPT2,      // Add this
    GPT3,      // Add this
}

// In src/models/gpt.rs (to be created by GPT team)
pub struct GPTConfig { /* ... */ }
pub struct GPTModel { /* ... */ }
pub struct GPTForward { /* ... */ }
```

#### 4. FFI Interface for GPT Kernels
**Support Activities**:
- [ ] Review FFI signatures for GPT kernels
- [ ] Validate memory safety
- [ ] Support async kernel execution
- [ ] Debug FFI boundary issues

**Known Patterns**:
- Use `SafeCudaPtr` for GPU memory
- Propagate errors via `Result<T, CudaError>`
- Document ownership and lifetime rules
- Add FFI integration tests

#### 5. Integration Test Support
**Support Activities**:
- [ ] Create `tests/gpt_integration.rs`
- [ ] Add GPT-specific test cases
- [ ] Validate end-to-end pipeline
- [ ] Support performance benchmarking

---

## Proactive Tasks (Can Start Immediately)

### Task 1: Prepare GPT Adapter Skeleton
**File**: `src/models/gpt.rs` (create)  
**Priority**: High  
**Effort**: 2 hours

Create skeleton for GPT team:
```rust
//! GPT Model Implementation
//!
//! Supports GPT-2 and GPT-3 architectures.
//! Integrates with Foundation layer via LlamaInferenceAdapter.

use crate::cuda_ffi::CudaContext;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub context_length: usize,
}

#[derive(Debug)]
pub struct GPTModel {
    pub config: GPTConfig,
    pub total_vram_bytes: usize,
    // Add weight tensors here
}

#[derive(Debug, Error)]
pub enum GPTError {
    #[error("Forward pass failed: {0}")]
    ForwardPassFailed(String),
}

pub struct GPTWeightLoader;
pub struct GPTForward;

// TODO: Implement by GPT-Gamma team
```

### Task 2: Add GPT Integration Test Template
**File**: `tests/gpt_integration.rs` (create)  
**Priority**: High  
**Effort**: 1 hour

Create test template:
```rust
// GPT Integration Tests - GT-XXX
//
// Integration tests for GPT-2/GPT-3 models.
// Tests LayerNorm, GELU, and MHA kernels.

use worker_orcd::models::gpt::{GPTConfig, GPTWeightLoader};

#[test]
fn test_gpt_model_loading() {
    // TODO: Implement by GPT-Gamma team
}

#[test]
fn test_gpt_layernorm_kernel() {
    // TODO: Implement by GPT-Gamma team
}

#[test]
fn test_gpt_gelu_kernel() {
    // TODO: Implement by GPT-Gamma team
}

#[test]
fn test_gpt_mha_kernel() {
    // TODO: Implement by GPT-Gamma team
}
```

### Task 3: Document GPT Integration Guide
**File**: `docs/GPT_INTEGRATION_GUIDE.md` (create)  
**Priority**: Medium  
**Effort**: 1 hour

Guide for GPT team:
- How to extend adapter pattern
- FFI interface patterns
- Testing requirements
- Performance expectations

### Task 4: Validate FFI for GPT Kernels
**File**: `src/cuda_ffi/mod.rs`  
**Priority**: Medium  
**Effort**: 1 hour

Review FFI interface for GPT needs:
- [ ] Kernel launch interface adequate?
- [ ] Memory management patterns clear?
- [ ] Error handling sufficient?
- [ ] Documentation complete?

---

## Reactive Tasks (Wait for Issues)

### Issue Template: Kernel Integration Problem
**Trigger**: GPT team reports kernel issue  
**Response**:
1. Review kernel FFI signature
2. Validate memory layout
3. Debug with minimal test case
4. Fix FFI or document usage pattern

### Issue Template: GGUF v3 Parsing Failure
**Trigger**: GPT model fails to load  
**Response**:
1. Examine GGUF v3 format differences
2. Update parser for new format
3. Add MXFP4 support if needed
4. Test with GPT model files

### Issue Template: Adapter Extension Issue
**Trigger**: GPT team has questions about adapter  
**Response**:
1. Review adapter pattern design
2. Provide code examples
3. Pair on implementation if needed
4. Update documentation

### Issue Template: Performance Issue
**Trigger**: GPT kernels slower than expected  
**Response**:
1. Profile kernel execution
2. Identify bottleneck
3. Suggest optimization
4. Validate improvement

---

## Communication Protocol

### Daily Standup Questions
1. Any blocking issues from GPT team?
2. Any kernel integration problems?
3. Any GGUF v3 parsing issues?
4. Any adapter extension questions?

### Coordination Files
- Update `execution/day-tracker.md` with support activities
- Log issues in `execution/dependencies.md`
- Track fixes in sprint completion summary

---

## Acceptance Criteria

- [ ] All proactive tasks completed
- [ ] GPT team reports no blocking issues
- [ ] GPT adapter skeleton created
- [ ] Integration test template created
- [ ] Documentation complete
- [ ] FFI interface validated for GPT needs
- [ ] GGUF v3 parsing working (if needed)
- [ ] All tests passing

---

## Definition of Done

- [ ] GPT team unblocked
- [ ] Integration successful
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Story marked complete in day-tracker.md

---

**Created**: 2025-10-05  
**Owner**: Foundation-Alpha  
**Dependencies**: FT-028 (Llama integration support)  
**Blocks**: GT-028 (GPT Gate 2 checkpoint)

---
Built by Foundation-Alpha 🏗️
