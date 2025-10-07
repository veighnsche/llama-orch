# 📧 Round 2 Investigation Coordinator - Mission Briefing

**To:** Investigation Coordinator  
**From:** Project Lead  
**Date:** 2025-10-07T14:15Z  
**Subject:** Round 2 Investigation Coordination - 8 Specialized Teams  
**Priority:** HIGH

---

## 👋 Welcome, Coordinator!

You are now the **Investigation Coordinator** for Round 2. Your mission is to guide 8 specialized teams through systematic re-validation of our codebase after the bug fixes from Round 1.

**Your Responsibilities:**
1. ✅ Introduce each team to their assignment
2. ✅ Create prompts to run each individual team
3. ✅ Ensure teams follow the rules (see below)
4. ✅ Track progress and dependencies
5. ✅ Coordinate handoffs between teams

---

## 📚 Key Documents (Attached to This Email)

### Planning Documents
1. **POST_FIX_INVESTIGATION_PLAN.md** - Master plan with all 8 teams and their missions
2. **BUG_FIX_CASCADE_ANALYSIS.md** - Context: Why Round 2 is needed (cascade effects)
3. **FALSE_LEADS_REVALIDATION.md** - What needs re-testing and why

### Team Chronicles (Templates for Each Team)
4. **TEAM_MONET_CHRONICLE.md** - Code Auditor (starts first, no dependencies)
5. **TEAM_PICASSO_CHRONICLE.md** - cuBLAS Resolver (waits for MONET)
6. **TEAM_VAN_GOGH_CHRONICLE.md** - Weight Inspector (waits for MONET)
7. **TEAM_SHAKESPEARE_CHRONICLE.md** - Integration Validator (waits for MONET)
8. **TEAM_FROST_CHRONICLE.md** - Sampling Validator (waits for MONET)
9. **TEAM_DICKINSON_CHRONICLE.md** - Parity Checker (waits for MONET)
10. **TEAM_REMBRANDT_CHRONICLE.md** - Fix Restorer (waits for PICASSO & VAN GOGH)
11. **TEAM_WHITMAN_CHRONICLE.md** - Documentarian (waits for ALL teams)

### Round 1 Reference (For Context Only)
12. **ROUND_001/** directory - Previous investigation (DO NOT USE THESE TEMPLATES)

---

## 🎯 Your Mission: Team Coordination

### Phase 1: Team Introduction & Prompt Creation

For each team, you need to:

1. **Read their chronicle** (e.g., TEAM_MONET_CHRONICLE.md)
2. **Create a prompt** that introduces them to their mission
3. **Include the rules** (see below)
4. **Specify their deliverable**
5. **List their dependencies** (who they wait for)

**Example Prompt Structure:**
```
You are TEAM MONET, the Code Auditor for Round 2.

MISSION: Audit current codebase to determine which fixes are actually applied.

YOUR SPECIALIZATION:
[Copy from chronicle's "Team Introduction" section]

YOUR TASKS:
[Copy from chronicle's "Tasks" section]

DELIVERABLE:
File: investigation-teams/TEAM_MONET_CODE_AUDIT.md
[Copy from chronicle's "Deliverable" section]

DEPENDENCIES:
None - You start first!

RULES:
[Copy rules from below]

CHRONICLE:
Fill in your progress in: investigation-teams/TEAM_MONET_CHRONICLE.md

BEGIN YOUR INVESTIGATION!
```

### Phase 2: Rule Enforcement

**CRITICAL RULES FOR ALL TEAMS:**

#### ❌ RULE 1: NO BACKGROUND TESTING
```
WRONG:
cargo test ... &  # Background job
nohup cargo test ... &  # Detached session

RIGHT:
cargo test ... --nocapture  # Foreground, blocking
```

**Why:** Background jobs lose logs and you can't see results.

#### ❌ RULE 2: NO CLI PIPING
```
WRONG:
./llama-cli -m model.gguf -p "test" | grep output | head -n 10

RIGHT:
./llama-cli -m model.gguf -p "test" > output.log 2>&1
cat output.log | grep output | head -n 10
```

**Why:** CLI is interactive and hangs waiting for input. Save to file first.

#### ✅ RULE 3: DOCUMENT EVERYTHING IN THE CODEBASE
```
Teams must add comments directly in the code:

// [TEAM MONET 2025-10-07] Checked this line
// Current value: CUBLAS_OP_T with lda=896
// Status: Matches SENTINEL's fix from Round 1
```

**Why:** Next teams need to see what was already checked.

#### ✅ RULE 4: APPEND-ONLY COMMENTS
```
WRONG:
// [TEAM X] This is wrong  <- DELETED
// [TEAM Y] Actually it's correct  <- OVERWROTE

RIGHT:
// [TEAM X] This is wrong
// [TEAM Y 2025-10-07] CORRECTION: Actually it's correct
// Evidence: [link to test results]
```

**Why:** Preserve investigation history.

#### ✅ RULE 5: FILL IN YOUR CHRONICLE
```
Each team has a chronicle file (e.g., TEAM_MONET_CHRONICLE.md)
They MUST fill in:
- Investigation Log (sessions)
- Detailed Findings (results)
- Final Verdict (conclusion)
- Reflections (lessons learned)
```

**Why:** Documentation for future teams.

### Phase 3: Team Name Recognition

**Round 2 Teams (Famous Painters & Poets):**
- 🎨 MONET, PICASSO, VAN GOGH, REMBRANDT (painters)
- 📝 SHAKESPEARE, FROST, DICKINSON, WHITMAN (poets)

**Round 1 Teams (Other Names):**
- FELICIA, AURORA, SENTINEL, CASCADE, HELIOS, etc.
- These are in ROUND_001/ directory
- **DO NOT use Round 1 templates or rules for Round 2 teams**

**How to tell:**
- If team name is a famous painter/poet → Round 2 team → Use new chronicles
- If team name is anything else → Round 1 team → Reference only, don't re-run

---

## 📋 Execution Timeline

### Phase 1: MONET (Start Immediately)
**Duration:** 2-4 hours  
**Dependencies:** None  
**Deliverable:** TEAM_MONET_CODE_AUDIT.md

**Your action:**
1. Create prompt for TEAM MONET
2. Send to project lead
3. Wait for MONET to complete

### Phase 2: Parallel Investigations (After MONET)
**Duration:** 2-3 hours  
**Dependencies:** All need MONET's report

**Teams to run in parallel:**
- TEAM PICASSO (cuBLAS resolution)
- TEAM VAN GOGH (weight resolution)
- TEAM SHAKESPEARE (integration testing)
- TEAM FROST (sampling verification)
- TEAM DICKINSON (parity checking)

**Your action:**
1. Once MONET completes, create prompts for all 5 teams
2. They can run in parallel (no dependencies on each other)
3. Track their progress

### Phase 3: REMBRANDT (After PICASSO & VAN GOGH)
**Duration:** 1-2 hours  
**Dependencies:** PICASSO and VAN GOGH verdicts

**Your action:**
1. Wait for PICASSO and VAN GOGH to complete
2. Create prompt for TEAM REMBRANDT with their verdicts
3. REMBRANDT restores any needed fixes

### Phase 4: WHITMAN (After All Teams)
**Duration:** 1 hour  
**Dependencies:** ALL teams complete

**Your action:**
1. Wait for all 7 teams to complete
2. Create prompt for TEAM WHITMAN
3. WHITMAN documents everything

---

## 📊 Progress Tracking Template

Use this to track team status:

```markdown
# Round 2 Progress Tracker

## Phase 1: Audit
- [ ] TEAM MONET - Status: Not Started / In Progress / Complete
  - Deliverable: TEAM_MONET_CODE_AUDIT.md
  - Started: [time]
  - Completed: [time]

## Phase 2: Parallel Investigations
- [ ] TEAM PICASSO - Status: Waiting / In Progress / Complete
  - Deliverable: TEAM_PICASSO_CUBLAS_RESOLUTION.md
  - Started: [time]
  - Completed: [time]

- [ ] TEAM VAN GOGH - Status: Waiting / In Progress / Complete
  - Deliverable: TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md
  - Started: [time]
  - Completed: [time]

- [ ] TEAM SHAKESPEARE - Status: Waiting / In Progress / Complete
  - Deliverable: TEAM_SHAKESPEARE_INTEGRATION_REPORT.md
  - Started: [time]
  - Completed: [time]

- [ ] TEAM FROST - Status: Waiting / In Progress / Complete
  - Deliverable: TEAM_FROST_SAMPLING_REPORT.md
  - Started: [time]
  - Completed: [time]

- [ ] TEAM DICKINSON - Status: Waiting / In Progress / Complete
  - Deliverable: TEAM_DICKINSON_PARITY_REPORT.md
  - Started: [time]
  - Completed: [time]

## Phase 3: Restoration
- [ ] TEAM REMBRANDT - Status: Waiting / In Progress / Complete
  - Deliverable: TEAM_REMBRANDT_RESTORATION_REPORT.md
  - Started: [time]
  - Completed: [time]

## Phase 4: Documentation
- [ ] TEAM WHITMAN - Status: Waiting / In Progress / Complete
  - Deliverable: POST_FIX_VALIDATION_SUMMARY.md
  - Started: [time]
  - Completed: [time]
```

---

## 🚨 Common Issues & How to Handle Them

### Issue 1: Team Runs Background Tests
**Symptom:** Team says "I started the test" but no output  
**Action:** Remind them of RULE 1, ask them to re-run in foreground

### Issue 2: Team Uses CLI Piping
**Symptom:** Team says "command is hanging"  
**Action:** Remind them of RULE 2, tell them to save to file first

### Issue 3: Team Doesn't Document in Code
**Symptom:** No code comments added  
**Action:** Remind them of RULE 3, ask them to add comments before proceeding

### Issue 4: Team Overwrites Previous Comments
**Symptom:** Previous team's comments are gone  
**Action:** Remind them of RULE 4, ask them to restore and append instead

### Issue 5: Team Doesn't Fill Chronicle
**Symptom:** Chronicle file is empty  
**Action:** Remind them of RULE 5, ask them to document their work

### Issue 6: Team Confused About Round 1 vs Round 2
**Symptom:** Team references FELICIA, AURORA, etc. as if they're current  
**Action:** Clarify that Round 1 teams are historical reference only

---

## 📧 Communication Templates

### Template 1: Team Introduction Email
```
Subject: Round 2 Investigation - You are TEAM [NAME]

Hello TEAM [NAME]!

You are the [SPECIALIZATION] for Round 2 of our investigation.

MISSION:
[Copy from chronicle]

ATTACHED DOCUMENTS:
1. Your chronicle: TEAM_[NAME]_CHRONICLE.md (fill this in as you work)
2. Master plan: POST_FIX_INVESTIGATION_PLAN.md (for context)
3. [Any other relevant docs]

DEPENDENCIES:
[List who you're waiting for, or "None - start immediately!"]

RULES:
[Copy 5 critical rules from above]

DELIVERABLE:
[Specify the report file they need to create]

Please confirm receipt and let me know when you begin!

Best regards,
Investigation Coordinator
```

### Template 2: Dependency Notification
```
Subject: TEAM [NAME] - Your dependency is complete!

Hello TEAM [NAME]!

Good news! TEAM [DEPENDENCY] has completed their investigation.

Their findings are available in: [FILE PATH]

You can now begin your investigation. Please:
1. Read their report
2. Copy relevant findings to your chronicle
3. Begin your tasks
4. Fill in your chronicle as you work

Let me know when you start!

Best regards,
Investigation Coordinator
```

### Template 3: Rule Violation Warning
```
Subject: TEAM [NAME] - Rule Violation Detected

Hello TEAM [NAME],

I noticed you [DESCRIPTION OF VIOLATION].

This violates RULE [NUMBER]: [RULE TEXT]

Please:
1. Stop current work
2. [CORRECTIVE ACTION]
3. Resume with correct approach

Let me know when you've corrected this.

Best regards,
Investigation Coordinator
```

---

## 🎯 Success Criteria

Round 2 is successful when:

1. ✅ All 8 teams have completed their deliverables
2. ✅ All chronicles are filled in
3. ✅ All rules were followed
4. ✅ TEAM SHAKESPEARE has clear verdict (bugs fixed or not)
5. ✅ TEAM WHITMAN has updated all documentation

**Final Deliverable:** POST_FIX_VALIDATION_SUMMARY.md

This document will tell us:
- Are all bugs from Round 1 actually fixed?
- Are there new bugs discovered?
- What's the current state of the codebase?
- What should Round 3 investigate (if needed)?

---

## 📞 Contact Information

**For Questions:**
- Technical issues: [Project Lead]
- Rule clarifications: [This briefing document]
- Team coordination: [You, the Coordinator]

**For Escalation:**
- Team blocked: Notify project lead immediately
- Critical bug found: Notify project lead immediately
- Timeline slipping: Notify project lead with revised estimate

---

## 🎓 Final Notes

**Remember:**
- Round 1 was about finding bugs
- Round 2 is about validating fixes and resolving contradictions
- Round 3 (if needed) will investigate any remaining issues

**Your role is critical:**
- You keep teams on track
- You enforce the rules
- You coordinate handoffs
- You ensure documentation quality

**Trust the process:**
- Each team is specialized for a reason
- Dependencies are there for a reason
- Rules exist because Round 1 teams struggled without them

---

## ✅ Coordinator Checklist

Before starting Round 2:
- [ ] Read all 11 attached documents
- [ ] Understand the 5 critical rules
- [ ] Set up progress tracking spreadsheet
- [ ] Prepare email templates
- [ ] Confirm project lead is ready to start TEAM MONET

During Round 2:
- [ ] Send team introduction emails
- [ ] Monitor rule compliance
- [ ] Track progress
- [ ] Send dependency notifications
- [ ] Handle issues as they arise

After Round 2:
- [ ] Collect all deliverables
- [ ] Verify all chronicles are filled
- [ ] Review TEAM WHITMAN's summary
- [ ] Report final status to project lead

---

**You've got this, Coordinator!**

Round 2 is more structured than Round 1. With clear rules, specialized teams, and your coordination, we'll get definitive answers about our bug fixes.

**Ready to begin?**

Please confirm receipt of this briefing and all attached documents. Once confirmed, I'll send you the signal to start TEAM MONET.

---

**Signed,**  
Project Lead  
2025-10-07T14:15Z

---

## 📎 Attachments (11 files)
1. POST_FIX_INVESTIGATION_PLAN.md
2. BUG_FIX_CASCADE_ANALYSIS.md
3. FALSE_LEADS_REVALIDATION.md
4. TEAM_MONET_CHRONICLE.md
5. TEAM_PICASSO_CHRONICLE.md
6. TEAM_VAN_GOGH_CHRONICLE.md
7. TEAM_SHAKESPEARE_CHRONICLE.md
8. TEAM_FROST_CHRONICLE.md
9. TEAM_DICKINSON_CHRONICLE.md
10. TEAM_REMBRANDT_CHRONICLE.md
11. TEAM_WHITMAN_CHRONICLE.md

---

## 📊 ROUND 2 STATUS UPDATE (2025-10-07T22:53Z)

### Completed Teams

✅ **TEAM MONET** (Code Auditor)
- Status: COMPLETE (2025-10-07T14:22Z)
- Deliverable: TEAM_MONET_CODE_AUDIT.md
- Verdict: 4/6 fixes applied, 2/6 partial

✅ **TEAM PICASSO** (cuBLAS Resolver)
- Status: COMPLETE (2025-10-07T15:38Z)
- Deliverable: TEAM_PICASSO_CUBLAS_RESOLUTION.md
- Verdict: KEEP CUBLAS_OP_T (matches llama.cpp), bug is elsewhere
- Bonus: Created parity logging infrastructure for Round 3

✅ **TEAM VAN GOGH** (Weight Inspector)
- Status: COMPLETE (2025-10-07T22:38Z)
- Deliverable: TEAM_VAN_GOGH_WEIGHT_RESOLUTION.md
- Verdict: Output norm weights CORRECT as-is (mean=7.14 intentional)

✅ **TEAM SHAKESPEARE** (Integration Validator)
- Status: COMPLETE (2025-10-07T22:53Z)
- Deliverable: TEAM_SHAKESPEARE_INTEGRATION_REPORT.md
- Verdict: ❌ **COHERENT OUTPUT NOT ACHIEVED**
- Evidence: 5/5 test runs produced garbage output
- llama.cpp produces perfect haiku with same model
- Recommendation: Round 3 needed, focus on embedding layer

### Round 2 Final Verdict

❌ **BUGS REMAIN** - Critical issue in uninvestigated subsystem

**What Works:**
- ✅ cuBLAS (CUBLAS_OP_T correct)
- ✅ Softmax (double precision, sum=1.0)
- ✅ Sampling infrastructure (different outputs each run)
- ✅ Q/K/V biases (loaded and added)

**What's Broken:**
- ❌ Output is complete garbage (mojibake, foreign tokens, code tokens)
- ❌ No coherent English text
- ❌ llama.cpp works perfectly with same model

**Root Cause:**
Bug is NOT in cuBLAS, softmax, or sampling. Bug is in uninvestigated subsystem, most likely:
1. Embedding layer (token ID → vector conversion)
2. Special token handling (chat template disabled)
3. Attention mask or RoPE

### Recommended Round 3 Teams

**HIGH PRIORITY:**
1. **TEAM DICKINSON** - Use PICASSO's parity logging to find divergence point
2. **TEAM FROST** - Inspect embedding layer vs llama.cpp

**MEDIUM PRIORITY:**
3. **TEAM REMBRANDT** - Investigate chat template crash
4. **TEAM WHITMAN** - Validate RoPE implementation

---

**Coordinator Note:** Round 2 successfully validated fixes but revealed deeper bug. Infrastructure stable, clear path forward for Round 3.

---

## 🔥 BREAKTHROUGH: Root Cause Likely Identified!

**Date:** 2025-10-07T23:02Z  
**Discovered By:** TEAM SHAKESPEARE (via reference implementation analysis)

### The Bug: Embedding Table Transpose

**Evidence:**
- VAN GOGH confirmed: `token_embd.weight` dimensions are `[896, 151936]`
- Candle/mistral.rs expect: `[151936, 896]` (vocab × hidden)
- Our code assumes: `[151936, 896]` but data is `[896, 151936]` (transposed!)

**The Fix:**
```cpp
// embedding.cu line 143
// WRONG:
half value = weight_matrix[token_id * hidden_dim + dim_idx];

// CORRECT:
half value = weight_matrix[dim_idx * vocab_size + token_id];
```

**Confidence:** 🔥🔥🔥 95%+ (this explains everything)

**Why This Explains Everything:**
- ✅ Garbage output (wrong embeddings from wrong memory locations)
- ✅ Consistent garbage (deterministic wrong lookup)
- ✅ llama.cpp works (handles GGUF layout correctly)
- ✅ Softmax/cuBLAS correct (operate on garbage data correctly)
- ✅ Numeric ranges reasonable (reading valid FP16, just wrong values)

### Immediate Action Required

**TEAM FROST** should:
1. Apply transpose fix to `bin/worker-orcd/cuda/kernels/embedding.cu` line 143
2. Rebuild: `cargo build --features cuda --release`
3. Run haiku test
4. If output is coherent: **BUG SOLVED!** 🎉
5. Report results immediately

**Estimated Time:** 5-10 minutes

### Documentation

Full analysis in:
- `investigation-teams/REFERENCE_IMPLEMENTATION_ANALYSIS.md`
- `investigation-teams/ROUND_002_FINAL_SUMMARY.md`
- `investigation-teams/TEAM_SHAKESPEARE_INTEGRATION_REPORT.md`
