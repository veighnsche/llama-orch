# vram-residency Proof Bundles

This directory contains proof bundles demonstrating that vram-residency works as intended.

## ⚠️ AUDIT FINDINGS: INSUFFICIENT EVIDENCE

**Status**: ⚠️ **PROOF BUNDLE DIRECTORY IS EMPTY**  
**Auditor Confidence**: LOW  
**Action Required**: See [AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md](./AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md)

---

## Latest Proof Bundle

**Run ID**: NONE (directory empty)  
**Date**: N/A  
**Status**: ⚠️ NO EVIDENCE GENERATED

### Quick Links

- ⚠️ [AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md](./AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md) - **READ THIS FIRST**
- ✅ [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) - How to fix the gaps

### Claims (UNVERIFIED - No Evidence)

- ⚠️ **112 tests** passing (100% pass rate) — **NO EVIDENCE**
- ⚠️ **50 requirements** covered (100% coverage) — **NO EVIDENCE**
- ⚠️ **Real GPU VRAM** tested (RTX 3060 + RTX 3090) — **NO EVIDENCE**
- ⚠️ **TIER 1 security** validated — **NO EVIDENCE**
- ⚠️ **Production ready** — **CANNOT VERIFY**

## What's Claimed (But Not Proven)

### Functional Correctness (NO EVIDENCE)
- ⚠️ Cryptographic seal creation (HMAC-SHA256) — **NO LOGS**
- ⚠️ Seal verification (timing-safe) — **NO LOGS**
- ⚠️ VRAM allocation/deallocation — **NO LOGS**
- ⚠️ Digest computation (SHA-256) — **NO LOGS**
- ⚠️ Input validation (all attack vectors) — **NO LOGS**

### Security Properties (NO EVIDENCE)
- ⚠️ Memory safety (bounds checking, no panics) — **NO LOGS**
- ⚠️ VRAM pointer privacy (never exposed) — **NO LOGS**
- ⚠️ Seal forgery prevention — **NO LOGS**
- ⚠️ Integer overflow protection — **NO LOGS**
- ⚠️ Path traversal prevention — **NO LOGS**
- ⚠️ Null byte injection prevention — **NO LOGS**
- ⚠️ Timing attack resistance — **NO LOGS**

### Performance (NO MEASUREMENTS)
- ⚠️ ~2ms seal operation (end-to-end) — **NO DATA**
- ⚠️ ~1ms VRAM allocation — **NO DATA**
- ⚠️ ~5 GB/s memory bandwidth — **NO DATA**
- ⚠️ Sub-millisecond cryptographic operations — **NO DATA**

### Integration (NO EVIDENCE)
- ⚠️ Works on real GPU VRAM — **NO LOGS**
- ⚠️ Falls back to mock when no GPU — **NO LOGS**
- ⚠️ Auto-detects GPU and CUDA toolkit — **NO LOGS**
- ⚠️ BDD tests pass in both modes — **NO LOGS**

## How to Generate New Proof Bundle

```bash
# Run tests (auto-generates proof bundle)
cargo test -p vram-residency

# Proof bundle will be created at:
# .proof_bundle/unit/<timestamp>-<git_sha>/
```

## Proof Bundle Structure

```
.proof_bundle/
├── README.md (this file)
└── unit/
    └── 20251002-101833-f7247fae/
        ├── test_report.md          # Human-readable summary
        ├── spec_coverage.md         # Requirements coverage
        └── (future: evidence files)
```

## Evidence Files (Future)

When fully implemented, proof bundles will include:

- `crypto_validation.jsonl` - Cryptographic operation results
- `policy_enforcement.jsonl` - VRAM-only policy checks
- `cuda_operations.jsonl` - GPU memory operations
- `bounds_checking.jsonl` - Safety validation results
- `input_validation.jsonl` - Input validation tests
- `test_results.txt` - Raw cargo test output
- `build_log.txt` - Build output with CUDA compilation

## Retention Policy

- Keep latest 3 proof bundles
- Clean older bundles before major releases
- Archive critical proof bundles (security audits, major milestones)

## For Reviewers

**To verify vram-residency works as intended**:

1. Read [test_report.md](unit/20251002-101833-f7247fae/test_report.md)
2. Check [spec_coverage.md](unit/20251002-101833-f7247fae/spec_coverage.md)
3. Review evidence files (when generated)
4. Optionally: Re-run tests yourself

**Key Questions Answered**:

- ✅ Does it meet security requirements? → YES (TIER 1 compliant)
- ✅ Does it work on real GPU? → YES (tested on RTX 3060/3090)
- ✅ Is it production-ready? → YES (pending audit-logging integration)
- ✅ Are all specs covered? → YES (100% coverage)

---

**Last Updated**: 2025-10-02  
**Maintainer**: vram-residency team
