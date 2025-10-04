# PM Recommendation: GGUF Security Vulnerability

**From**: auth-min Security Team 🎭  
**To**: Project Manager  
**Date**: 2025-10-04  
**Subject**: MANDATORY M0 Security Fix - GGUF Parsing Heap Overflow  
**Priority**: 🔴 **P0 - BLOCKING**

---

## Executive Summary

The llama-research team has identified **critical heap overflow vulnerabilities** (CWE-119, CWE-787) in GGUF file parsing that **MUST be fixed in M0**. This is not a "platform mode" security issue—it affects **all deployment modes** including Home Mode.

**Recommendation**: ✅ **APPROVE +1 DAY TO M0 TIMELINE FOR SECURITY FIX**

---

## Vulnerability Assessment

### Severity: HIGH ✅

**Attack Vector**: Malicious GGUF file with crafted tensor offsets  
**Impact**: Heap overflow → Worker crash (DoS) or arbitrary code execution (RCE)  
**Affected Component**: M0-W-1211 (GGUF Header Parser) - **code to be written in M0**  
**Likelihood**: HIGH (GGUF format is well-documented, exploitation is straightforward)

### Attack Scenarios (All Relevant to M0)

1. **Local Development** (Home Mode M0):
   - Developer downloads untrusted GGUF from internet
   - Worker loads malicious file → Heap overflow → Developer machine compromised

2. **Supply Chain Attack**:
   - Attacker compromises HuggingFace model repository
   - User downloads "legitimate" model → Worker crash or RCE

3. **Testing/CI** (M0 development):
   - Malicious GGUF in test suite → CI environment compromised

---

## Why This MUST Be Fixed in M0

### 1. Vulnerability Exists in M0 Code

The GGUF parser is **foundational to M0**:
- M0-W-1200: GGUF Format Support
- M0-W-1210: Pre-Load Validation
- M0-W-1211: GGUF Header Parsing ← **VULNERABLE CODE**

**Without GGUF parsing, M0 cannot load models.** The vulnerability will be introduced when we write this code.

### 2. Security Timeline Clarification

Your statement: *"security fixes (mostly for platform mode) are for m4"*

**This is correct for PLATFORM MODE security**, but this vulnerability is different:

| Security Type | Milestone | Scope |
|---------------|-----------|-------|
| **Memory Safety** (this issue) | **M0** | GGUF bounds validation, heap overflow prevention |
| **Authentication** | M3 | Bearer tokens, timing-safe comparison |
| **Authorization** | M3 | Multi-tenancy, job ownership |
| **Audit Logging** | M3 | Compliance, tenant context |
| **Multi-Node Security** | M4 | Cluster-wide auth, distributed trust |

**This is a memory safety bug, not a platform security feature.**

### 3. M0 Deployment Mode Still Needs Security

From `00_llama-orch.md`:
- **Home Mode (M0)**: "Performance > Security" but **NOT "Performance, no security"**
- M0 already includes critical safety features:
  - ✅ VRAM OOM handling (M0-W-1021)
  - ✅ VRAM residency verification (M0-W-1012)
  - ✅ Model load progress events (M0-W-1621)

**Bounds validation is equally critical**—it prevents memory corruption before it happens.

### 4. Cost-Benefit Analysis

| Factor | Assessment |
|--------|------------|
| **Implementation Cost** | +1 day to LT-001 (GGUF Header Parser story) |
| **Complexity** | Low (straightforward bounds checks) |
| **Timeline Impact** | Minimal (+1 day in 6-7 week M0 = ~1.5% increase) |
| **Risk Reduction** | HIGH → LOW (eliminates heap overflow vector) |
| **Retrofit Cost** | Higher (M1+ requires code changes + migration) |
| **Testing Value** | Fuzzing/property tests validate GGUF parser correctness |

**Fixing now is cheaper than retrofitting later.**

### 5. Defense in Depth

M0 already follows defense-in-depth principles:
- VRAM-only enforcement (no RAM fallback)
- VRAM residency checks (detect leaks)
- VRAM OOM handling (graceful degradation)

**Bounds validation completes the safety layer** for model loading.

---

## Implementation Plan (Already Prepared)

The security alert document provides complete implementation guidance:

### New Requirement: M0-W-1211a

**GGUF Bounds Validation (Security)**

Worker-orcd MUST validate all GGUF offsets and sizes to prevent heap overflows.

**Required Checks**:
1. Tensor offset MUST be >= header_size + metadata_size
2. Tensor offset MUST be < file_size
3. Tensor offset + tensor_size MUST be <= file_size
4. Tensor offset + tensor_size MUST NOT overflow (integer overflow check)
5. Metadata string lengths MUST be < 1MB (sanity check)
6. Array lengths MUST be < 1M elements (sanity check)

### Story Card Updates

**LT-001: GGUF Header Parser** (Week 2 of M0)
- Add security acceptance criteria (6 items)
- Add fuzzing tests (malformed GGUF files)
- Add property tests (bounds validation)
- Add edge case tests (boundary conditions)
- **Estimated impact**: +1 day

**LT-005: Pre-Load Validation**
- Add security validation before mmap
- Add audit logging for rejected files
- Add clear error messages

### Testing Requirements

1. **Fuzzing Tests** (REQUIRED):
   - Offset beyond file
   - Integer overflow (offset + size wraps)
   - Offset before data section
   - Size extends beyond file

2. **Property Tests** (REQUIRED):
   - Validate bounds checking never allows out-of-bounds access
   - 1000+ random inputs

3. **Edge Case Tests** (REQUIRED):
   - Offset at exact file boundary
   - Zero-size tensors
   - Maximum valid offset

---

## Timeline Impact

### Current M0 Timeline
- Foundation (Weeks 1-5): HTTP server, GGUF loader, tokenization, basic kernels
- Architecture Adapters (Weeks 6-7): InferenceAdapter pattern, Llama + GPT adapters
- **Total**: 6-7 weeks

### With Security Fix
- Foundation (Weeks 1-5): HTTP server, **GGUF loader + security**, tokenization, basic kernels
  - LT-001 (GGUF Header Parser): +1 day for security implementation + tests
- Architecture Adapters (Weeks 6-7): Unchanged
- **Total**: 6-7 weeks (absorbed within existing buffer)

**Impact**: Negligible (1 day in 42-49 day timeline = 2% increase, within estimation variance)

---

## Risk Assessment

### If We Fix in M0 ✅

**Pros**:
- ✅ Security from day 1 (no vulnerable code in production)
- ✅ Cheaper implementation (part of initial development)
- ✅ Better testing (fuzzing validates parser correctness)
- ✅ No retrofit/migration needed
- ✅ Aligns with M0 safety features (VRAM OOM, residency checks)

**Cons**:
- ❌ +1 day to LT-001 story

### If We Defer to M1+ ❌

**Pros**:
- ✅ Slightly faster M0 delivery (-1 day)

**Cons**:
- ❌ M0 ships with known HIGH-severity vulnerability
- ❌ Developers at risk during M0 development/testing
- ❌ Higher retrofit cost (code changes + migration)
- ❌ Potential security incidents before fix
- ❌ Reputation risk (shipping vulnerable code)
- ❌ Inconsistent with M0 safety principles

---

## Recommendation

### ✅ APPROVE SECURITY FIX IN M0

**Justification**:
1. Vulnerability exists in M0 code (GGUF parser)
2. Affects all deployment modes (Home, Lab, Platform)
3. Low implementation cost (+1 day)
4. High risk reduction (heap overflow → worker crash/RCE)
5. Cheaper to fix now than retrofit later
6. Aligns with M0 critical safety features

### Action Items

**Immediate** (Before LT-001 Implementation):
1. ✅ Add M0-W-1211a to M0 spec (`01_M0_worker_orcd.md`)
2. ✅ Update LT-001 story card (add security acceptance criteria)
3. ✅ Update LT-005 story card (add validation requirements)
4. ✅ Notify llama-team of security requirements

**Week 2** (Sprint 1 - LT-001 Implementation):
1. Implement bounds validation in GGUF parser
2. Add fuzzing tests for malformed GGUF files
3. Add property tests for bounds validation
4. Add edge case tests
5. Security review by auth-min team

**Week 3** (Sprint 1 - LT-005 Implementation):
1. Integrate validation into pre-load checks
2. Add audit logging for rejected files
3. Test with real malicious GGUF files (if available)
4. Final security review

---

## Alternative: Defer to M1 (NOT RECOMMENDED)

If you choose to defer:

**Requirements**:
1. Document known vulnerability in M0 release notes
2. Add security warning to README
3. Restrict M0 to trusted GGUF files only
4. Prioritize fix in M1 (P0 blocker)
5. Accept risk of security incidents during M0

**auth-min team position**: We **strongly recommend against deferral**. The cost is minimal (+1 day) and the risk is HIGH.

---

## Conclusion

This is a **memory safety bug** in foundational M0 code, not a "platform mode" security feature. It affects all deployment modes and should be fixed when the code is written (M0), not retrofitted later (M1+).

**Recommendation**: ✅ **APPROVE +1 DAY TO M0 FOR SECURITY FIX**

**Timeline Impact**: Negligible (1 day in 6-7 week timeline)  
**Risk Reduction**: HIGH → LOW  
**Cost**: Minimal now, expensive later

---

**Prepared by**: auth-min Security Team 🎭  
**Reviewed by**: llama-research Team (vulnerability discovery)  
**Status**: Awaiting PM approval  
**Decision Required By**: Before LT-001 implementation (Week 2 of M0)

---

## References

- **Security Alert**: `/home/vince/Projects/llama-orch/bin/shared-crates/auth-min/SECURITY_ALERT_GGUF_PARSING.md`
- **M0 Spec**: `/home/vince/Projects/llama-orch/bin/.specs/01_M0_worker_orcd.md`
- **System Spec**: `/home/vince/Projects/llama-orch/bin/.specs/00_llama-orch.md`
- **Research**: `/home/vince/Projects/llama-orch/bin/worker-orcd/.plan/llama-team/stories/LT-000-prep/REASERCH_pt1.md` (line 45)
- **External**: https://blog.huntr.com/gguf-file-format-vulnerabilities-a-guide-for-hackers

---

Security verified by auth-min Team 🎭
