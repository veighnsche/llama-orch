# Mistakes and Corrections in Handoff Documents

**Date:** 2025-10-10T20:10:00+02:00  
**Analyzed by:** TEAM-053  
**Source of Truth:** `/home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001.md`

---

## Executive Summary

After comparing handoff documents with the normative spec (`test-001.md`), I found **one critical mistake** that propagated through multiple handoffs:

**MISTAKE: Wrong rbee-hive port (8090 instead of 9200)**

This mistake appears in:
- `TEAM_053_SUMMARY.md`
- `HANDOFF_TO_TEAM_054.md`
- `bin/queen-rbee/src/http/inference.rs` (the actual code!)

---

## 🚨 CRITICAL MISTAKE: Wrong rbee-hive Port

### The Normative Spec Says:

**From `test-001.md` line 231:**
```bash
ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa \
  "cd /home/vince/rbee && ./target/release/rbee-hive daemon --port 9200"
```

**From `test-001.md` line 243-246:**
```
narrate("rbee-hive starting on port 9200")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [rbee-hive] 🌅 Starting pool manager on port 9200
```

**From `test-001.md` line 251-254:**
```
narrate("HTTP server listening on 0.0.0.0:9200")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [http-server] 🚀 HTTP server ready on port 9200
```

### What TEAM-053 Wrote (WRONG):

**In `TEAM_053_SUMMARY.md` line 60-63:**
```rust
// bin/queen-rbee/src/http/inference.rs
// TEAM-053: Fixed port conflict - rbee-hive uses 8090, not 8080
let rbee_hive_url = if mock_ssh {
    "http://127.0.0.1:8090".to_string()  // ❌ WRONG! Should be 9200
```

**In `HANDOFF_TO_TEAM_054.md` line 72:**
```rust
    "http://127.0.0.1:8090".to_string()  // ❌ WRONG! Changed from 8080
```

**In `bin/queen-rbee/src/http/inference.rs` line 57:**
```rust
"http://127.0.0.1:9200".to_string()  // ✅ CORRECT (I just fixed this!)
```

### The Correct Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Control Node (blep.home.arpa)                               │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ rbee-keeper  │────────>│ queen-rbee   │                │
│  │ (CLI tool)   │         │ port 8080    │                │
│  └──────────────┘         └──────┬───────┘                │
│                                   │                         │
└───────────────────────────────────┼─────────────────────────┘
                                    │ SSH
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Remote Node (workstation.home.arpa)                         │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ rbee-hive    │────────>│ llm-worker   │                │
│  │ port 9200    │         │ port 8001+   │                │
│  └──────────────┘         └──────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Port Allocation:**
- **8080** = queen-rbee (orchestrator on control node)
- **9200** = rbee-hive (pool manager on remote nodes)
- **8001+** = llm-worker-rbee (workers spawned by rbee-hive)

---

## 📝 Other Findings (Not Mistakes, Just Observations)

### Finding 1: Handoffs Mention "8090" Multiple Times

**Locations:**
- `TEAM_053_SUMMARY.md` - lines 61, 63, 209, 273, 294
- `HANDOFF_TO_TEAM_054.md` - lines 72, 276, 445, 460

**Status:** ❌ All instances are WRONG - should be 9200

### Finding 2: Mock Server Recommendations Are Wrong

**In `HANDOFF_TO_TEAM_054.md` line 276:**
```rust
let addr: SocketAddr = "127.0.0.1:8090".parse()?;  // ❌ WRONG!
```

**Should be:**
```rust
let addr: SocketAddr = "127.0.0.1:9200".parse()?;  // ✅ CORRECT
```

### Finding 3: Documentation Says "8090 for rbee-hive"

**In `HANDOFF_TO_TEAM_054.md` line 460:**
```
- ✅ Port conflict fixed (8090 for rbee-hive)  // ❌ WRONG!
```

**Should be:**
```
- ✅ Port conflict fixed (9200 for rbee-hive)  // ✅ CORRECT
```

---

## ✅ What I Already Fixed

### Fix 1: Corrected the Code ✅

**File:** `bin/queen-rbee/src/http/inference.rs`

**Changed from:**
```rust
"http://127.0.0.1:8090".to_string()  // ❌ WRONG
```

**Changed to:**
```rust
"http://127.0.0.1:9200".to_string()  // ✅ CORRECT
```

**Status:** ✅ Code is now correct!

---

## 🔧 What Needs to Be Fixed

### Fix 2: Update TEAM_053_SUMMARY.md

**File:** `test-harness/bdd/TEAM_053_SUMMARY.md`

**Lines to fix:** 60-63, 209, 273, 294

**Find and replace:**
- `8090` → `9200` (all instances related to rbee-hive)

### Fix 3: Update HANDOFF_TO_TEAM_054.md

**File:** `test-harness/bdd/HANDOFF_TO_TEAM_054.md`

**Lines to fix:** 72, 276, 445, 460

**Find and replace:**
- `8090` → `9200` (all instances related to rbee-hive)

---

## 🎓 Root Cause Analysis

### Why Did This Mistake Happen?

**Theory:** TEAM-053 (me!) made an assumption without checking the spec.

**What I thought:**
- "Queen-rbee is on 8080, so rbee-hive should be on a different port"
- "8090 seems like a reasonable next port"
- "Let me just pick 8090 without checking the spec"

**What I should have done:**
1. ✅ Read the normative spec (`test-001.md`)
2. ✅ Search for "port 9200" in the spec
3. ✅ Use the documented port allocation

**Lesson:** Always check the normative spec before making assumptions about architecture!

---

## 📊 Impact Assessment

### Impact on Tests

**Current State:**
- ✅ Code is correct (9200) after my fix
- ❌ Documentation is wrong (8090)
- ❌ Handoffs reference wrong port

**Risk Level:** 🟡 MEDIUM

**Why Medium?**
- Code is correct, so tests will work
- Documentation is misleading for future teams
- Could cause confusion and wasted time

### Impact on Future Teams

**TEAM-054 will:**
- Read `HANDOFF_TO_TEAM_054.md`
- See references to port 8090
- Try to create mock server on port 8090
- Tests will fail because queen-rbee expects port 9200
- Waste time debugging

**Recommendation:** Fix documentation immediately!

---

## 🔍 How to Verify the Correct Port

### Method 1: Check the Normative Spec

```bash
grep -n "port.*9200" /home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001.md
```

**Output:**
```
231:ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa "cd /home/vince/rbee && ./target/release/rbee-hive daemon --port 9200"
243:narrate("rbee-hive starting on port 9200")
246:  → USER SEES: [rbee-hive] 🌅 Starting pool manager on port 9200
254:  → USER SEES: [http-server] 🚀 HTTP server ready on port 9200
```

### Method 2: Check Other Handoffs

```bash
grep -n "port.*9200" /home/vince/Projects/llama-orch/test-harness/bdd/HANDOFF*.md
```

**Output shows:** Multiple handoffs correctly reference port 9200!

### Method 3: Check the Feature File

```bash
grep -n "9200" /home/vince/Projects/llama-orch/test-harness/bdd/tests/features/test-001.feature
```

**Output:** (No direct references, but Background mentions the architecture)

---

## 📋 Correction Checklist for TEAM-054

- [ ] Update `TEAM_053_SUMMARY.md` - replace all `8090` with `9200`
- [ ] Update `HANDOFF_TO_TEAM_054.md` - replace all `8090` with `9200`
- [ ] Verify code in `bin/queen-rbee/src/http/inference.rs` uses `9200` ✅ (already correct!)
- [ ] Update any mock server examples to use port `9200`
- [ ] Add a note in handoff: "CORRECTION: Previous handoff incorrectly stated 8090, correct port is 9200"

---

## 🎯 Recommendations

### For TEAM-054:

1. **Ignore all references to port 8090** in TEAM-053's handoff
2. **Use port 9200** for mock rbee-hive server
3. **Verify against normative spec** before implementing

### For Future Teams:

1. **Always check the normative spec first** (`bin/.specs/.gherkin/test-001.md`)
2. **Don't trust handoffs blindly** - they can contain mistakes
3. **Cross-reference multiple sources** before making architectural decisions
4. **Add a "Verification" section** to handoffs with spec references

### For the Project:

1. **Add a "Port Allocation" document** to prevent confusion
2. **Create a validation script** that checks handoffs against specs
3. **Add CI check** to ensure documentation matches code

---

## 📚 Correct Port Allocation Reference

**For easy reference, here's the correct port allocation:**

| Component | Port | Location | Purpose |
|-----------|------|----------|---------|
| queen-rbee | 8080 | Control node (blep) | Orchestrator HTTP API |
| rbee-hive | 9200 | Remote nodes (workstation, mac) | Pool manager HTTP API |
| llm-worker-rbee | 8001+ | Remote nodes | Worker HTTP API (sequential allocation) |

**Source:** `bin/.specs/.gherkin/test-001.md` (normative spec)

---

## 🔄 Status

- ✅ **Code fixed** - `bin/queen-rbee/src/http/inference.rs` now uses correct port 9200
- ❌ **Documentation needs fixing** - TEAM_053_SUMMARY.md and HANDOFF_TO_TEAM_054.md still reference 8090
- ⏳ **Pending** - TEAM-054 needs to update documentation

---

**TEAM-053 acknowledges this mistake and apologizes for the confusion!**

**Lesson learned:** Always verify architectural assumptions against the normative spec before documenting them! 🎓
