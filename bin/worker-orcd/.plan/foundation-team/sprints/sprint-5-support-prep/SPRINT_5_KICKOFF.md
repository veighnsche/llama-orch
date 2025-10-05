# Sprint 5: Support + Prep - Kickoff

**Team**: Foundation-Alpha  
**Sprint Duration**: Days 53-60 (8 agent-days)  
**Sprint Type**: Support & Preparation  
**Status**: 🚀 Ready to start

---

## Sprint Overview

Sprint 5 marks a transition from independent development to **collaborative support**. With Gate 1 complete, the Foundation layer is stable and ready for integration. Llama-Beta and GPT-Gamma teams are now building on top of our work, and we're here to support them.

This sprint is fundamentally different from Sprints 1-4:
- **Reactive, not proactive**: We respond to integration issues
- **Support-focused**: Other teams drive our agenda
- **Preparation for Sprint 6**: We use downtime to prepare for adapter work

---

## What Changed Since Sprint 4

### ✅ Sprint 4 Achievements (Gate 1)
- HTTP server with SSE streaming
- FFI layer with CUDA integration
- Shared kernels (RMSNorm, RoPE, SiLU, MatMul)
- Advanced sampling (temperature, top-k, top-p, min-p, repetition penalty)
- Integration test framework
- Gate 1 validation complete

### 🎯 Sprint 5 Focus
- Support Llama team integration
- Support GPT team integration
- Fix bugs discovered during integration
- Clean up code and documentation
- Prepare for Sprint 6 adapter work

---

## Stories in This Sprint

### FT-028: Support Llama Integration (Days 53-54)
**Type**: Reactive support  
**Goal**: Unblock Llama team  
**Key Activities**:
- Debug FFI issues
- Fix GGUF parsing edge cases
- Optimize VRAM allocation
- Update documentation
- Add Llama-specific tests

**Proactive Tasks**:
- Document adapter pattern
- Create integration checklist
- Validate FFI error handling

### FT-029: Support GPT Integration (Days 55-56)
**Type**: Reactive support  
**Goal**: Unblock GPT team  
**Key Activities**:
- Debug kernel integration
- Support GGUF v3 / MXFP4 parsing
- Extend adapter pattern for GPT
- Update documentation
- Add GPT-specific tests

**Proactive Tasks**:
- Create GPT adapter skeleton
- Create GPT integration test template
- Document GPT integration guide

### FT-030: Bug Fixes and Integration Cleanup (Days 57-59)
**Type**: Proactive cleanup  
**Goal**: Prepare for Sprint 6  
**Key Activities**:
- Fix all reported bugs
- Address TODO markers
- Clean up code style
- Update documentation
- Add regression tests

**Bug Categories**:
- FFI boundary issues
- CUDA context management
- VRAM enforcement edge cases
- KV cache management
- Sampling kernel edge cases
- Error propagation

---

## Current State Assessment

### What's Already Done

#### ✅ Adapter Pattern (Partially)
- `LlamaInferenceAdapter` implemented
- Supports Qwen and Phi-3
- Unified interface for multiple models
- Comprehensive tests

**Location**: `src/models/adapter.rs`

#### ✅ Llama Configuration
- `LlamaConfig` struct
- GQA/MHA detection
- RoPE parameter handling

**Location**: `src/model/llama_config.rs`

#### ✅ Integration Tests
- Llama integration suite
- Phi-3 integration tests
- Adapter integration tests
- Full pipeline tests

**Location**: `tests/llama_integration_suite.rs`, `tests/phi3_integration.rs`, `tests/adapter_integration.rs`

#### ✅ FFI Layer
- CUDA context management
- Memory allocation (stub)
- Error handling
- SafeCudaPtr wrapper

**Location**: `src/cuda_ffi/mod.rs`

### What Needs Work

#### ⚠️ FFI Implementation (Stubs)
All CUDA operations are stubs with TODO markers:
- `cudaMemcpy` (line 102, 133)
- `cudaFree` (line 152)
- `cudaSetDevice` (line 195)
- `cudaMalloc` (line 211)
- `cudaMemGetInfo` (line 228, 236)

**Status**: Documented as `TODO(ARCH-CHANGE)`  
**Action**: Document as future work, support teams with current interface

#### ⚠️ GPT Support (Missing)
No GPT-specific code yet:
- No `src/models/gpt.rs`
- No GPT adapter integration
- No GPT tests

**Status**: Will be created by GPT team  
**Action**: Provide skeleton and support integration

#### ⚠️ Documentation Gaps
Missing guides:
- Adapter pattern usage guide
- FFI best practices
- VRAM debugging guide
- Integration checklist

**Status**: Will create during Sprint 5  
**Action**: Proactive task during support downtime

---

## Sprint Execution Strategy

### Reactive Support Protocol

**When Llama/GPT team reports an issue**:

1. **Acknowledge immediately** (< 5 minutes)
   - Confirm receipt
   - Estimate response time
   - Ask for clarification if needed

2. **Triage** (< 15 minutes)
   - Determine severity (blocking vs non-blocking)
   - Identify affected component
   - Check for known issues

3. **Investigate** (< 1 hour for blocking issues)
   - Reproduce issue
   - Identify root cause
   - Determine fix approach

4. **Fix** (< 4 hours for blocking issues)
   - Implement fix
   - Add regression test
   - Verify fix works
   - Update documentation

5. **Communicate** (immediately after fix)
   - Notify team
   - Explain fix
   - Document pattern
   - Update day-tracker

### Proactive Work Protocol

**When no active support requests**:

1. **Check for pending issues** (every 2 hours)
   - Review Llama team day-tracker
   - Review GPT team day-tracker
   - Check shared coordination docs

2. **Work on proactive tasks** (priority order)
   - Documentation (highest priority)
   - Test coverage
   - Code cleanup
   - Future preparation

3. **Stay available** (always)
   - Monitor communication channels
   - Respond quickly to questions
   - Be ready to context-switch

---

## Tools and Resources

### Coordination Files
- `execution/day-tracker.md` - Daily progress tracking
- `execution/dependencies.md` - Issue and dependency tracking
- `coordination/master-timeline.md` - Cross-team coordination

### Code Locations
- `src/models/adapter.rs` - Adapter pattern implementation
- `src/cuda_ffi/mod.rs` - FFI layer (with TODOs)
- `tests/llama_integration_suite.rs` - Llama tests
- `tests/adapter_integration.rs` - Adapter tests

### Documentation to Create
- `docs/ADAPTER_PATTERN_GUIDE.md`
- `docs/GPT_INTEGRATION_GUIDE.md`
- `docs/VRAM_DEBUGGING_GUIDE.md`
- `docs/INTEGRATION_CHECKLIST.md`
- `docs/FUTURE_WORK.md`

### Testing Commands
```bash
# Run all tests
cargo test --all-features

# Run integration tests only
cargo test --test '*integration*'

# Run specific test file
cargo test --test llama_integration_suite

# Check code style
cargo fmt --check
cargo clippy -- -D warnings

# Check for memory leaks (if available)
valgrind --leak-check=full target/debug/worker-orcd
```

---

## Success Criteria

### Must Have (Sprint Cannot Complete Without)
- [ ] Llama team reports no blocking issues
- [ ] GPT team reports no blocking issues
- [ ] All reported bugs fixed
- [ ] All tests passing
- [ ] Documentation updated

### Should Have (High Priority)
- [ ] Adapter pattern documented
- [ ] GPT adapter skeleton created
- [ ] Integration checklists created
- [ ] Code cleanup complete
- [ ] Regression tests added

### Nice to Have (If Time Permits)
- [ ] Performance optimizations
- [ ] Additional test coverage
- [ ] Refactoring improvements
- [ ] Future work documented

---

## Risk Assessment

### Risk 1: Teams Report Many Blocking Issues
**Probability**: Medium  
**Impact**: High  
**Mitigation**:
- Prioritize ruthlessly
- Fix critical issues first
- Defer proactive work
- Ask for help if overwhelmed

### Risk 2: Unclear Requirements from Teams
**Probability**: Medium  
**Impact**: Medium  
**Mitigation**:
- Ask clarifying questions immediately
- Provide examples and options
- Document assumptions
- Iterate quickly

### Risk 3: Integration Issues Reveal Fundamental Problems
**Probability**: Low  
**Impact**: Critical  
**Mitigation**:
- Escalate immediately
- Assess impact on Gate 2
- Propose solutions
- Coordinate with PM team

### Risk 4: Not Enough Support Requests (Idle Time)
**Probability**: Low  
**Impact**: Low  
**Mitigation**:
- Work on proactive tasks
- Prepare for Sprint 6
- Improve documentation
- Add test coverage

---

## Daily Routine

### Morning (Start of Day)
1. Check Llama team day-tracker for issues
2. Check GPT team day-tracker for issues
3. Review coordination docs for updates
4. Update own day-tracker with plan
5. Prioritize tasks (reactive > proactive)

### Throughout Day
1. Monitor for support requests
2. Work on highest priority task
3. Document decisions and patterns
4. Update day-tracker with progress
5. Communicate blockers immediately

### Evening (End of Day)
1. Update day-tracker with completed work
2. Document any open issues
3. Plan next day's work
4. Communicate status to teams
5. Commit and push all changes

---

## Communication Guidelines

### Response Time Expectations
- **Blocking issue**: < 5 minutes acknowledgment, < 4 hours resolution
- **Non-blocking issue**: < 1 hour acknowledgment, < 1 day resolution
- **Question**: < 1 hour response
- **Clarification**: < 30 minutes response

### Communication Style
- **Be precise**: Use exact technical terms
- **Be helpful**: Provide examples and context
- **Be proactive**: Anticipate follow-up questions
- **Be responsive**: Acknowledge quickly, even if you can't fix immediately

### Documentation Standards
- **Always document decisions**: Why, not just what
- **Always add examples**: Show, don't just tell
- **Always update on changes**: Keep docs in sync with code
- **Always sign your work**: Foundation-Alpha signature

---

## Sprint 5 Mindset

**This sprint is different**. We're not building new features - we're **enabling other teams to succeed**. Our success is measured by their success.

**Key Mindsets**:
1. **Service-oriented**: We exist to support other teams
2. **Responsive**: Quick acknowledgment and resolution
3. **Thorough**: Fix issues properly, not quickly
4. **Proactive**: Use downtime to prepare and improve
5. **Collaborative**: We're all building the same system

**Remember**: A bug fixed today prevents ten questions tomorrow. A guide written today prevents a hundred interruptions next week. Quality support work compounds.

---

## Let's Go! 🚀

Sprint 5 starts on Day 53. We're ready to support Llama-Beta and GPT-Gamma as they integrate with the Foundation layer. Let's make their integration smooth and successful.

**Foundation-Alpha is ready to support.** 🏗️

---

**Created**: 2025-10-05  
**Owner**: Foundation-Alpha  
**Status**: 🚀 Ready to start

---
Built by Foundation-Alpha 🏗️
