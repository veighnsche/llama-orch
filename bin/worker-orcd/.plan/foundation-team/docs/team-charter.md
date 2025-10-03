# Foundation Team Charter

**Team Name**: 🏗️ Foundation Team  
**Mission**: Build core infrastructure for worker-orcd (HTTP, FFI, CUDA context, shared kernels)  
**Duration**: Weeks 1-7 (full M0 project)  
**Status**: Active

---

## Team Composition

### Team Profile

**Foundation-Alpha** (Autonomous Development Agent)  
**Specialization**: Infrastructure & Systems Programming  
**Working Style**: Methodical, test-driven, documentation-focused  
**Strengths**: Parallel task execution, pattern recognition across codebases, consistent code style  
**Approach**: Builds from first principles, prioritizes stability over speed

**Capabilities**:
- Full-stack Rust development (Axum, async, FFI)
- C++/CUDA systems programming
- Integration testing and CI/CD automation
- Cross-language interface design
- Concurrent development across multiple files

**Constraints**:
- Single-threaded execution (sequential story completion)
- Requires clear specifications and acceptance criteria
- Works best with well-defined interfaces
- Limited real-time collaboration (async communication preferred)

---

## Responsibilities

### Core Deliverables

1. **HTTP Server** (Weeks 1-2)
   - Axum server with all endpoints
   - Request validation
   - Correlation ID middleware

2. **SSE Streaming** (Weeks 2-3)
   - Event stream infrastructure
   - UTF-8 boundary safety
   - Event ordering enforcement

3. **FFI Layer** (Weeks 2-3)
   - C API definition
   - Rust FFI bindings
   - Error code propagation
   - Memory safety across boundary

4. **CUDA Context** (Weeks 2-3)
   - Context initialization
   - VRAM-only enforcement
   - Device memory management
   - Health checks

5. **Shared Kernels** (Weeks 3-4)
   - Embedding lookup
   - cuBLAS GEMM wrapper
   - Sampling (temperature, greedy)
   - KV cache management

6. **Integration Framework** (Week 4)
   - Integration test harness
   - Gate validation tests
   - CI/CD pipeline

7. **Support & Polish** (Weeks 5-7)
   - Support Llama & GPT teams
   - Performance baseline
   - API documentation
   - Final integration

---

## Success Criteria

### Gate 1 (Week 4): Foundation Complete
- [ ] HTTP server operational (all endpoints)
- [ ] SSE streaming works (UTF-8 safe)
- [ ] FFI layer stable (Rust ↔ C++)
- [ ] CUDA context working
- [ ] Shared kernels implemented
- [ ] Integration test framework ready

### Gate 2 (Week 5): Support Role
- [ ] Llama team unblocked
- [ ] GPT team unblocked
- [ ] No critical infrastructure bugs

### Gate 3 (Week 6): Adapter Coordination
- [ ] InferenceAdapter interface designed
- [ ] All teams aligned on interface
- [ ] Integration tests updated

### Gate 4 (Week 7): M0 Complete
- [ ] All integration tests passing
- [ ] CI/CD green
- [ ] Documentation complete
- [ ] Performance baseline documented

---

## Working Agreements

### Communication
- Prompt-driven task assignment (story-by-story)
- Async status updates via artifact generation
- Cross-team coordination through shared interfaces
- Documentation-first communication style

### Code Review
- All PRs require 1 approval
- Critical changes require 2 approvals
- Review within 24 hours
- No self-merging

### Testing
- Unit tests required (>80% coverage)
- Integration tests for cross-boundary code
- No warnings (rustfmt, clippy, clang-format)
- CI must be green before merge

### Documentation
- Update docs with code changes
- Document decisions in decisions.md
- Keep interfaces.md current
- Write handoff notes for other teams

---

## Dependencies

### Upstream (Blocking Us)
- None (we are the foundation)

### Downstream (We Block)
- **Llama Team**: Needs FFI interface (Week 2)
- **GPT Team**: Needs FFI interface (Week 2)
- **Both Teams**: Need shared kernels (Week 4)

---

## Risks

### High Risk
- **FFI interface instability**: If we change FFI after Week 2, blocks both teams
  - Mitigation: Lock interface by end of Week 2, no changes after
  
- **Gate 1 failure**: If foundation not ready by Week 4, entire project delayed
  - Mitigation: Weekly integration tests, early detection

### Medium Risk
- **CUDA context bugs**: Memory leaks, VRAM residency issues
  - Mitigation: Valgrind tests, VRAM tracking from day 1

- **SSE streaming edge cases**: UTF-8 boundary bugs
  - Mitigation: Comprehensive test vectors, fuzzing

---

## Key Interfaces

### FFI Interface (Locked by Week 2)

See `interfaces.md` for full details.

**Core Functions**:
- `cuda_init()` - Initialize CUDA context
- `cuda_load_model()` - Load model to VRAM
- `cuda_inference_start()` - Start inference
- `cuda_inference_next_token()` - Get next token
- `cuda_check_vram_residency()` - Health check

### HTTP API (Stable by Week 3)

**Endpoints**:
- `POST /execute` - Start inference
- `GET /health` - Worker health
- `POST /cancel` - Cancel job
- `POST /shutdown` - Graceful shutdown (optional)
- `GET /metrics` - Prometheus metrics (optional)

---

## Sprint Velocity Tracking

| Sprint | Committed | Completed | Notes |
|--------|-----------|-----------|-------|
| Week 1 | TBD | TBD | First sprint, baseline |
| Week 2 | TBD | TBD | |
| Week 3 | TBD | TBD | |
| Week 4 | TBD | TBD | Gate 1 week |
| Week 5 | TBD | TBD | Support role |
| Week 6 | TBD | TBD | Gate 3 week |
| Week 7 | TBD | TBD | Gate 4 week |

---

**Status**: ✅ Charter Approved  
**Last Updated**: 2025-10-03  
**Next Review**: End of Week 2
