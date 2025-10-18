# 🚨 MANDATORY DEBUGGING RULES

**Version:** 1.0  
**Date:** 2025-10-18  
**Status:** MANDATORY - Violations result in FINES

---

## ⚠️ CRITICAL: The "Fixed" Claim Problem

**67 teams failed by ignoring rules. Don't make it worse by claiming victory prematurely.**

### The Catastrophic Pattern

```
❌ TEAM-XXX: "Fixed the question mark bug!"
   → Writes 4 markdown files documenting the "fix"
   → Creates test scripts
   → Updates handoff with "✅ FIXED"
   → Doesn't test end-to-end
   → Bug still exists
   → Now we need CLAIM AUDIT across all files
   → Wastes everyone's time cleaning up false claims
```

---

## 1. NO VICTORY CLAIMS WITHOUT END-TO-END VERIFICATION

### ❌ BANNED BEFORE END-TO-END TEST

- Writing "FIXED" in any document
- Writing "ROOT CAUSE FOUND" without verification
- Creating `*_FIX.md` or `*_SOLUTION.md` files
- Updating handoffs with ✅ checkmarks
- Writing "The bug is solved"
- Creating demo scripts showing the "fix"
- Committing code with "fix:" prefix in message

### ✅ REQUIRED BEFORE CLAIMING ANYTHING

1. **Reproduce the bug** - Confirm it actually exists
2. **Make your change** - Implement what you think fixes it
3. **Run end-to-end test** - Full integration test, not unit test
4. **Verify output** - Actually check the result matches expectations
5. **Test edge cases** - Try it with different inputs
6. **THEN and ONLY THEN** - Write documentation

### Example: Correct Process

```markdown
# TEAM-XXX Investigation Log

## Hypothesis
The tokenizer may be buffering punctuation tokens.

## Test
Created unit test: tokens encode correctly ✅

## Code Change
Removed is_alphanumeric() check in TokenOutputStream

## Unit Test Result
✅ Test passes - punctuation tokens return immediately

## End-to-End Test
❌ FAIL - Model still generates 0 tokens

## Conclusion
Hypothesis was WRONG. Tokenization works fine.
The bug is elsewhere (generation loop).

## Status
🔴 NOT FIXED - Investigation continues
```

---

## 2. DOCUMENTATION DISCIPLINE

### One Investigation = One Document MAX

**❌ BANNED:**
- Creating multiple `.md` files for ONE bug
- `ANALYSIS.md` + `FIX.md` + `SUMMARY.md` + `HANDOFF.md` for same issue
- Separate files for "hypothesis", "solution", "verification"

**✅ REQUIRED:**
- ONE investigation document
- Update it as you learn
- Status at top: 🔴 NOT FIXED / 🟡 TESTING / 🟢 VERIFIED FIXED

### Template

```markdown
# TEAM-XXX: [Bug Name] Investigation

**Status:** 🔴 NOT FIXED | 🟡 TESTING | 🟢 VERIFIED FIXED

## Problem
[What's broken]

## Hypothesis
[What you think causes it]

## Tests Run
- [ ] Reproduced bug
- [ ] Unit test
- [ ] Integration test  
- [ ] End-to-end test
- [ ] Edge cases

## Changes Made
[List of file changes]

## Verification
[End-to-end test results]

## Conclusion
[What actually happened]
```

---

## 3. FINE SCHEDULE

Violations result in **FINES** and your work being **REJECTED**.

### 🚨 TIER 1: MASTURBATORY VICTORY (€500)

**Offense:** Creating 3+ markdown files documenting a "fix" before end-to-end testing

**Examples:**
- `TEAM_XXX_BUG_FIX.md` (comprehensive doc)
- `TEAM_XXX_ROOT_CAUSE.md` (analysis)
- `TEAM_XXX_VERIFICATION.md` (fake verification)
- Demo script: `SHOW_THE_FIX.sh`

**Why highest fine:**
- Creates maximum cleanup work
- Maximum false confidence
- Maximum confusion for next team
- Shows complete disregard for testing discipline

**Penalty:**
- €500 fine
- Work REJECTED
- Handoff DELETED
- Cited in "teams that failed" list
- Must clean up ALL false claims before handing off

---

### 🚨 TIER 2: PREMATURE CLAIM (€250)

**Offense:** Writing "FIXED" or "✅" before end-to-end testing

**Examples:**
- Handoff: "✅ Fixed the question mark bug"
- Commit message: "fix: solved tokenization issue"
- Code comment: "// TEAM-XXX: Fixed this bug"

**Penalty:**
- €250 fine
- Work REJECTED
- Must rewrite handoff with honest status
- Cannot hand off until properly tested

---

### 🚨 TIER 3: UNTESTED CODE (€100)

**Offense:** Committing "fix" without ANY testing

**Examples:**
- Changed code
- Unit test passes
- Didn't run end-to-end test
- Handed off as "done"

**Penalty:**
- €100 fine
- Work REJECTED
- Must test before resubmitting

---

### 🚨 TIER 4: INCOMPLETE CLEANUP (€150)

**Offense:** Getting called out for false claims, but not cleaning up ALL of them

**Examples:**
- Deleted `FIX.md` but left "FIXED" in handoff
- Removed claim from code but left it in README
- Half-assed cleanup that requires CLAIM AUDIT

**Why this matters:**
- If you claim victory in 10 places and only clean up 7, we waste time finding the other 3
- Forces CLAIM AUDIT = additional work for everyone

**Penalty:**
- €150 fine (more than original claim fine)
- Work REJECTED until 100% clean
- Next team audits your work

---

## 4. THE CLAIM AUDIT PROBLEM

### What is a Claim Audit?

When a team makes false claims in multiple places, the next team must:

1. **Find all claims** - grep for "fix", "solved", "root cause", "✅"
2. **Verify each claim** - Test to see if it's true
3. **Remove false claims** - Update all files
4. **Document reality** - Write what actually happened

**This wastes 2-4 hours of the next team's time.**

### How to Avoid Causing Claim Audits

**SIMPLE: Don't claim anything until you've tested it.**

If you follow this one rule, you will never trigger a claim audit.

---

## 5. VERIFICATION CHECKLIST

Before writing ANYTHING that suggests a fix:

### Required Tests (ALL must pass)

- [ ] **Reproduced original bug** - Confirmed it exists
- [ ] **Unit test passes** - Isolated component works
- [ ] **Integration test passes** - Components work together
- [ ] **End-to-end test passes** - Full pipeline works
- [ ] **Manual test passes** - Actual usage scenario works
- [ ] **Edge cases pass** - Tried different inputs
- [ ] **Verified output** - Actually read the results
- [ ] **Performance check** - Didn't make it slower
- [ ] **No regressions** - Didn't break other stuff

### If ANY test fails:

🔴 Status = NOT FIXED  
❌ Do NOT claim victory  
🔄 Keep investigating

### Only when ALL tests pass:

🟢 Status = VERIFIED FIXED  
✅ Now you can document the fix  
📝 Write ONE document with verification evidence

---

## 6. HANDOFF DISCIPLINE

### ❌ BANNED in Handoffs

```markdown
# TEAM-XXX HANDOFF

## What I Fixed
- ✅ Question mark bug - SOLVED
- ✅ Root cause found and fixed
- ✅ All tests passing

## The Fix
I removed the is_alphanumeric() check which was buffering 
punctuation. This completely solves the issue.

## Next Team
You can move on to the next priority. This is done.
```

**Why banned:** No evidence of end-to-end testing. Just claims.

### ✅ REQUIRED in Handoffs

```markdown
# TEAM-XXX HANDOFF

## What I Investigated
Question mark bug - model generates 0 tokens

## What I Learned
- ✅ Tokenization works (verified with real GGUF)
- ❌ TokenOutputStream change didn't help
- 🔴 Bug is in generation loop (0 tokens produced)

## What I Changed
- Added debug logging in inference.rs (lines 284-316)
- Created tokenization test (proves tokenizer works)

## Status
🔴 NOT FIXED - Bug still exists

## Next Steps for Next Team
1. Run ./ASK_SKY_BLUE.sh
2. Check logs: grep "Sampled token" /tmp/rbee-hive.log
3. Find why generation loop exits immediately

## Tests Run
- ✅ Tokenization test (tokens encode correctly)
- ❌ End-to-end test (still 0 tokens)
```

**Why good:** Honest status, evidence provided, clear next steps.

---

## 7. REAL-WORLD EXAMPLE: TEAM-095 VIOLATION

### What Happened

**TEAM-095 investigating question mark bug:**

1. ✅ Created tokenization test - GOOD
2. ✅ Found TokenOutputStream buffers punctuation - GOOD observation
3. ❌ Removed `is_alphanumeric()` check - UNVERIFIED change
4. ❌ Wrote `TEAM_095_QUESTION_MARK_BUG_FIX.md` - PREMATURE
5. ❌ Created `ASK_WHY_SKY_BLUE.sh` demo script - PREMATURE
6. ❌ Wrote 200+ lines documenting the "fix" - MASTURBATORY
7. ❌ Updated all comments with "TEAM-095: Fixed" - FALSE CLAIMS
8. ❌ Didn't run end-to-end test until called out
9. ❌ End-to-end test showed 0 tokens (bug NOT fixed)
10. ✅ Had to delete everything and start over - WASTED TIME

### Proper Process Would Have Been

1. ✅ Created tokenization test
2. ✅ Found TokenOutputStream buffers punctuation
3. ✅ Removed `is_alphanumeric()` check
4. ✅ **RUN END-TO-END TEST IMMEDIATELY**
5. ❌ Test fails - 0 tokens still generated
6. ✅ Update investigation doc: "Hypothesis wrong, bug elsewhere"
7. ✅ Revert change (or keep for different reason)
8. ✅ Continue investigating the REAL bug

**Time saved:** 2 hours  
**Confusion prevented:** 100%  
**Claim audit prevented:** YES

### Fine Assessment

**Violations:**
- Created 3+ docs about "fix" before testing: **€500 (TIER 1)**
- Wrote "FIXED" in multiple places: **€250 (TIER 2)**

**Total: €750 fine + work REJECTED**

---

## 8. WHEN CAN YOU CLAIM VICTORY?

### The One Rule

**You can claim a fix is verified when someone OUTSIDE your team confirms it works.**

This means:

1. You test it yourself (end-to-end) ✅
2. You document the test results ✅
3. You hand off with evidence ✅
4. **Next team or user confirms it works** ✅

Until step 4, status is: 🟡 APPEARS FIXED - PENDING VERIFICATION

Only after step 4: 🟢 VERIFIED FIXED

### Why This Matters

- Your tests might be wrong
- You might be testing the wrong thing
- You might have confirmation bias
- Fresh eyes catch what you miss

**External verification is the gold standard.**

---

## 9. SUMMARY

### The Three Rules

1. **TEST BEFORE YOU CLAIM**
   - No "fixed" claims without end-to-end verification
   - One document max per investigation
   - Status: 🔴 NOT FIXED until proven otherwise

2. **VERIFY EXTERNALLY**
   - Your tests might be wrong
   - Get next team or user to confirm
   - Only then: 🟢 VERIFIED FIXED

3. **CLEAN UP COMPLETELY**
   - If you made false claims, remove ALL of them
   - Don't trigger claim audits
   - Leave codebase cleaner than you found it

### The Fine Structure

- €500 - Masturbatory victory docs (3+ files)
- €250 - Premature "FIXED" claims
- €150 - Incomplete cleanup
- €100 - Untested code

### The Bottom Line

**Test end-to-end BEFORE claiming anything.**

**If you can't verify it, you didn't fix it.**

**One investigation = one document = one honest status.**

---

## 10. CONSEQUENCES

Violating these rules results in:

1. **Fine** (€100-€500 depending on severity)
2. **Work REJECTED** (must redo)
3. **Handoff DELETED** (doesn't count)
4. **Cited in "failed teams" list**
5. **Claim audit** forced on next team (wastes their time)
6. **Loss of trust** (future work scrutinized more)

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────┐
│ DEBUGGING RULES QUICK REFERENCE                        │
├────────────────────────────────────────────────────────┤
│ Before claiming ANYTHING:                              │
│   ✅ Reproduce bug                                     │
│   ✅ Make change                                       │
│   ✅ Run end-to-end test                               │
│   ✅ Verify output matches expectations                │
│   ✅ Test edge cases                                   │
│   ✅ Get external verification                         │
│                                                        │
│ BANNED:                                                │
│   ❌ Writing "FIXED" before testing                    │
│   ❌ Multiple .md files for one bug                    │
│   ❌ Demo scripts before verification                  │
│   ❌ Committing with "fix:" before testing             │
│                                                        │
│ REQUIRED:                                              │
│   ✅ One investigation doc                             │
│   ✅ Status: 🔴 NOT FIXED / 🟡 TESTING / 🟢 VERIFIED   │
│   ✅ Evidence of end-to-end testing                    │
│   ✅ Honest handoff                                    │
│                                                        │
│ FINES:                                                 │
│   €500 - Masturbatory victory (3+ docs)                │
│   €250 - Premature claims                              │
│   €150 - Incomplete cleanup                            │
│   €100 - Untested code                                 │
└────────────────────────────────────────────────────────┘
```

---

**This is not optional. This is mandatory.**

**Test before you claim. Verify before you celebrate.**

**Don't be TEAM-095. Don't trigger claim audits.**

**€750 in fines is expensive. Testing takes 5 minutes.**
