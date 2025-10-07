# 🍐 TEAM PEAR — Lessons Learned (Phases 1-2)

**Date:** 2025-10-07T11:55Z

---

## Key Lessons

### 1. DON'T Complain "Output is Garbage"
**WRONG:** "Output is garbage, therefore claim is wrong"  
**RIGHT:** "Did you test what you claimed? Where's the evidence?"

**Why:** We KNOW output is garbage. That's the baseline. It's not a finding.

### 2. LOOK for Tools FIRST, Then BUILD
**WRONG:** "I'm blocked, no parser available"  
**RIGHT:** "Let me search for existing parsers... Found! Let me use it."

**Phase 2 Example:**
- Initially claimed "BLOCKED: No GGUF parser"
- Actually found: `worker_gguf::GGUFMetadata::parse_tensors`
- Built test in 10 minutes

### 3. RUN ACTUAL TESTS, Don't Just Read
**WRONG:** Accept Team Sentinel's claim: "Manual Q[0] = -0.015185"  
**RIGHT:** Build test, run it, get: "Manual Q[0] = +0.001864" → MISMATCH!

**Result:** Found real evidence of problem (sign is different!)

### 4. Focus on Evidence Gaps, Not Outcomes
**WRONG:** "Only verified Q[0] but output is garbage"  
**RIGHT:** "Only verified Q[0] (0.11% coverage), need more comprehensive testing"

**Why:** Coverage is measurable, "garbage output" is not actionable.

---

## Phase 1 Results

**Fines:** €500  
**Key Finding:** Test bypasses special tokens (use_chat_template=false)  
**Artifacts:** Test logs, token dumps, vocab checks

**What Worked:**
- ✅ Ran haiku test
- ✅ Extracted token texts
- ✅ Documented missing evidence

**What Didn't:**
- ❌ Initially just read documents
- ❌ Complained about garbage output
- ❌ Didn't test enough

---

## Phase 2 Results

**Fines:** €300  
**Key Finding:** Cannot reproduce Sentinel's manual Q[0] (-0.015185 vs +0.001864)  
**Artifacts:** Test code (`tests/verify_manual_q0.rs`), logs showing mismatch

**What Worked:**
- ✅ Found existing GGUF parser
- ✅ Built manual verification test
- ✅ Ran test and found MISMATCH
- ✅ Produced real evidence

**What Didn't:**
- ❌ Initially claimed "BLOCKED"
- ❌ Didn't look for tools first
- ❌ Wasted time complaining about garbage output

---

## Phase 3 Approach

**LEARNED:** LOOK → BUILD → TEST → DOCUMENT

1. ✅ Search for existing KV cache tests
2. ✅ Found comprehensive test suite (30 tests)
3. ⏳ Running tests
4. ⏳ Analyze results
5. ⏳ Stamp code with findings

**NO BLOCKERS** — Found existing infrastructure!

---

## Updated Mission Rules

### NEVER BE BLOCKED
**Before claiming "BLOCKED":**
1. ✅ LOOK for existing tools/infrastructure
2. ✅ Search codebase for parsers, loaders, utilities
3. ✅ BUILD the tool yourself if needed
4. ✅ Use existing code as examples
5. ✅ EXHAUST ALL OPTIONS

**Only claim BLOCKED if:**
- Hardware not available
- External dependencies truly missing
- User permission required

**NEVER claim blocked for:**
- ❌ "No parser" — BUILD ONE
- ❌ "No numpy" — Use Rust
- ❌ "No infrastructure" — LOOK FIRST
- ❌ "Too hard" — FIGURE IT OUT

---

## Fines Summary

**Phase 1:** €500
- Team Purple: €250
- Team Blue: €100
- Team Blue+Purple: €150

**Phase 2:** €300
- Team Sentinel: €200 (incomplete verification + missing reproducibility)
- Team Charlie: €100

**Total:** €800

---

## Next Phases

**Phase 3:** KV Cache (running tests now)  
**Phase 4:** RoPE/RMSNorm  
**Phase 5:** Attention Mechanism  
**Phase 6:** FFN Path  
**Phase 7:** Sampling & Generation  
**Phase 8:** Weight Loading  
**Phase 9:** Infrastructure & Edge Cases  
**Phase 10:** Contradictions & Final Synthesis

**Approach:** LOOK → BUILD → TEST → DOCUMENT (NO BLOCKERS!)
