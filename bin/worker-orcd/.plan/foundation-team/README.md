# Foundation Team (Foundation-Alpha)

**Agent**: Foundation-Alpha  
**Timeline**: Days 1-89 (12 weeks)  
**Total Stories**: 49 story cards  
**Critical Milestones**: FFI Lock (Day 11), Gate 1 (Day 52), Gate 3 (Day 71), Gate 4 (Day 89)

---

## Team Mission

Build the foundational infrastructure for worker-orcd: HTTP server, FFI boundary, CUDA context management, shared kernels, and integration framework. This work enables Llama and GPT teams to implement model-specific logic.

---

## Story Organization

### Sprint 1: HTTP Foundation (Days 1-9)
**Stories**: FT-001 to FT-005 (5 stories)
- HTTP server with Axum
- POST /execute endpoint
- SSE streaming
- Correlation ID middleware
- Request validation

### Sprint 2: FFI Layer (Days 10-22)
**Stories**: FT-006 to FT-010 (5 stories)
- 🔒 **FFI Interface Definition** (Day 11 - LOCK)
- Rust FFI bindings
- Error code system (C++)
- Error conversion (Rust)
- CUDA context initialization

### Sprint 3: Shared Kernels (Days 23-38)
**Stories**: FT-011 to FT-020 (10 stories)
- VRAM-only enforcement
- FFI integration tests
- Device memory RAII
- VRAM residency verification
- Embedding lookup kernel
- cuBLAS GEMM wrapper
- Temperature scaling
- Greedy sampling
- Stochastic sampling
- Seeded RNG

### Sprint 4: Integration + Gate 1 (Days 39-52)
**Stories**: FT-021 to FT-027 (7 stories)
- KV cache allocation
- KV cache management
- Integration test framework
- HTTP-FFI-CUDA integration test
- Gate 1 validation tests
- Error handling integration
- 🎯 **Gate 1 Checkpoint** (Day 52)

### Sprint 5: Support + Prep (Days 53-60)
**Stories**: FT-028 to FT-030 (3 stories)
- Support Llama integration
- Support GPT integration
- Bug fixes and cleanup

### Sprint 6: Adapter + Gate 3 (Days 61-71)
**Stories**: FT-031 to FT-038 (8 stories)
- Performance baseline prep
- Gate 2 checkpoint
- InferenceAdapter interface
- Adapter factory pattern
- Architecture detection integration
- Update integration tests for adapters
- API documentation
- 🎯 **Gate 3 Checkpoint** (Day 71)

### Sprint 7: Final Integration (Days 72-89)
**Stories**: FT-039 to FT-050 (11 stories)
- CI/CD pipeline
- Performance baseline measurements
- All models integration test
- OOM recovery test
- UTF-8 streaming edge cases
- Cancellation integration test
- Documentation complete
- Final validation
- 🎯 **Gate 4 Checkpoint** (Day 89)
- Model load progress events
- Narration-core logging

---

## Critical Milestones

### FFI Interface Lock (Day 11)
**Story**: FT-006  
**Deliverable**: `coordination/FFI_INTERFACE_LOCKED.md`  
**Blocks**: Llama-Beta prep (LT-000), GPT-Gamma prep (GT-000)

### Gate 1: Foundation Complete (Day 52)
**Story**: FT-027  
**Validates**: HTTP, FFI, CUDA foundation working end-to-end  
**Blocks**: Llama Gate 1 (LT-020), GPT Gate 1 (GT-022)

### Gate 3: Adapter Complete (Day 71)
**Story**: FT-038  
**Validates**: InferenceAdapter pattern working for both architectures  
**Blocks**: Llama Gate 3 (LT-034), GPT Gate 3 (GT-041)

### Gate 4: M0 Complete (Day 89)
**Story**: FT-047  
**Validates**: All Foundation work complete, ready for production

---

## Dependencies

### Upstream (Blocks Foundation Team)
- None (Foundation team starts first)

### Downstream (Foundation Team Blocks)
- **Llama-Beta**: Waits for FFI lock (Day 11), Gate 1 (Day 52)
- **GPT-Gamma**: Waits for FFI lock (Day 11), Gate 1 (Day 52)

---

## Story Index

### FT-001 to FT-010
- [FT-001: HTTP Server Setup](stories/FT-001-to-FT-010/FT-001-http-server-setup.md)
- [FT-002: POST /execute Endpoint Skeleton](stories/FT-001-to-FT-010/FT-002-execute-endpoint-skeleton.md)
- [FT-003: SSE Streaming Implementation](stories/FT-001-to-FT-010/FT-003-sse-streaming.md)
- [FT-004: Correlation ID Middleware](stories/FT-001-to-FT-010/FT-004-correlation-id-middleware.md)
- [FT-005: Request Validation Framework](stories/FT-001-to-FT-010/FT-005-request-validation.md)
- [FT-006: FFI Interface Definition](stories/FT-001-to-FT-010/FT-006-ffi-interface-definition.md) 🔒
- [FT-007: Rust FFI Bindings](stories/FT-001-to-FT-010/FT-007-rust-ffi-bindings.md)
- [FT-008: Error Code System (C++)](stories/FT-001-to-FT-010/FT-008-error-code-system-cpp.md)
- [FT-009: Error Code to Result Conversion (Rust)](stories/FT-001-to-FT-010/FT-009-error-code-to-result-rust.md)
- [FT-010: CUDA Context Initialization](stories/FT-001-to-FT-010/FT-010-cuda-context-init.md)

### FT-011 to FT-020
- [FT-011: VRAM-Only Enforcement](stories/FT-011-to-FT-020/FT-011-vram-only-enforcement.md)
- [FT-012: FFI Integration Tests](stories/FT-011-to-FT-020/FT-012-ffi-integration-tests.md)
- [FT-013: Device Memory RAII Wrapper](stories/FT-011-to-FT-020/FT-013-device-memory-raii.md)
- [FT-014: VRAM Residency Verification](stories/FT-011-to-FT-020/FT-014-vram-residency-verification.md)
- [FT-015: Embedding Lookup Kernel](stories/FT-011-to-FT-020/FT-015-embedding-lookup-kernel.md)
- [FT-016: cuBLAS GEMM Wrapper](stories/FT-011-to-FT-020/FT-016-cublas-gemm-wrapper.md)
- [FT-017: Temperature Scaling Kernel](stories/FT-011-to-FT-020/FT-017-temperature-scaling-kernel.md)
- [FT-018: Greedy Sampling](stories/FT-011-to-FT-020/FT-018-greedy-sampling.md)
- [FT-019: Stochastic Sampling](stories/FT-011-to-FT-020/FT-019-stochastic-sampling.md)
- [FT-020: Seeded RNG](stories/FT-011-to-FT-020/FT-020-seeded-rng.md)

### FT-021 to FT-030
- [FT-021: KV Cache Allocation](stories/FT-021-to-FT-030/FT-021-kv-cache-allocation.md)
- [FT-022: KV Cache Management](stories/FT-021-to-FT-030/FT-022-kv-cache-management.md)
- [FT-023: Integration Test Framework](stories/FT-021-to-FT-030/FT-023-integration-test-framework.md)
- [FT-024: HTTP-FFI-CUDA Integration Test](stories/FT-021-to-FT-030/FT-024-http-ffi-cuda-integration-test.md)
- [FT-025: Gate 1 Validation Tests](stories/FT-021-to-FT-030/FT-025-gate1-validation-tests.md)
- [FT-026: Error Handling Integration](stories/FT-021-to-FT-030/FT-026-error-handling-integration.md)
- [FT-027: Gate 1 Checkpoint](stories/FT-021-to-FT-030/FT-027-gate1-checkpoint.md) 🎯
- [FT-028: Support Llama Integration](stories/FT-021-to-FT-030/FT-028-support-llama-integration.md)
- [FT-029: Support GPT Integration](stories/FT-021-to-FT-030/FT-029-support-gpt-integration.md)
- [FT-030: Bug Fixes and Integration Cleanup](stories/FT-021-to-FT-030/FT-030-bug-fixes-integration.md)

### FT-031 to FT-050
*(Stories 31-50 to be created in Phase 1 Day 2)*

---

## Execution Tracking

- **Day Tracker**: [execution/day-tracker.md](execution/day-tracker.md)
- **Dependencies**: [execution/dependencies.md](execution/dependencies.md)
- **Milestones**: [execution/milestones.md](execution/milestones.md)

---

## Integration Gates

- **Gate 1**: [integration-gates/gate-1-foundation-complete.md](integration-gates/gate-1-foundation-complete.md)
- **Gate 3**: [integration-gates/gate-3-adapter-complete.md](integration-gates/gate-3-adapter-complete.md)
- **Gate 4**: [integration-gates/gate-4-m0-complete.md](integration-gates/gate-4-m0-complete.md)

---

**Status**: 📋 30/49 stories complete (Day 1 deliverables)  
**Next**: Complete remaining 19 stories + sprint/gate documentation  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
