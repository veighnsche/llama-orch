# PEER REVIEW INVESTIGATION GUIDE

**Phase:** Investigation Peer Review  
**Duration:** 2-3 days  
**Status:** 🔍 CRITICAL QUALITY GATE

---

## 🎯 MISSION

**Verify, validate, and challenge every claim made by other teams.**

**Be CRITICAL. Find GAPS. Collect PROOF.**

**Your job is to find what the other team MISSED!**

**IMPORTANT:** 
- **Search for shared crates** in the codebase they're reviewing
- **Answer questions** left unanswered in their investigation reports

---

## 👥 PEER REVIEW ASSIGNMENTS

### Cross-Review Matrix

| Reviewing Team | Reviews Team 1 | Reviews Team 2 |
|----------------|----------------|----------------|
| **TEAM-131** (rbee-hive) | TEAM-134 (rbee-keeper) | TEAM-132 (queen-rbee) |
| **TEAM-132** (queen-rbee) | TEAM-131 (rbee-hive) | TEAM-133 (llm-worker-rbee) |
| **TEAM-133** (llm-worker-rbee) | TEAM-132 (queen-rbee) | TEAM-134 (rbee-keeper) |
| **TEAM-134** (rbee-keeper) | TEAM-131 (rbee-hive) | TEAM-133 (llm-worker-rbee) |

**Each team reviews 2 other teams' complete investigations.**

---

## 📋 WHAT TO REVIEW

### Each Team Must Review ALL Files:

#### Team 131 Files:
- [ ] `TEAM_131_rbee-hive_INVESTIGATION_REPORT.md`
- [ ] `TEAM_131_DEPENDENCY_GRAPH.md` (if exists)
- [ ] `TEAM_131_CRATE_PROPOSALS.md` (if exists)
- [ ] `TEAM_131_RISK_ANALYSIS.md` (if exists)
- [ ] Any other Team 131 deliverables

#### Team 132 Files:
- [ ] `TEAM_132_queen-rbee_INVESTIGATION_REPORT.md`
- [ ] `TEAM_132_DEPENDENCY_GRAPH.md` (if exists)
- [ ] `TEAM_132_CRATE_PROPOSALS.md` (if exists)
- [ ] `TEAM_132_RISK_ANALYSIS.md` (if exists)
- [ ] Any other Team 132 deliverables

#### Team 133 Files:
- [ ] `TEAM_133_llm-worker-rbee_INVESTIGATION_REPORT.md`
- [ ] `TEAM_133_REUSABILITY_MATRIX.md` (if exists)
- [ ] `TEAM_133_DEPENDENCY_GRAPH.md` (if exists)
- [ ] `TEAM_133_RISK_ANALYSIS.md` (if exists)
- [ ] Any other Team 133 deliverables

#### Team 134 Files:
- [ ] `TEAM_134_rbee-keeper_INVESTIGATION_REPORT.md`
- [ ] `TEAM_134_DEPENDENCY_GRAPH.md` (if exists)
- [ ] `TEAM_134_INVESTIGATION_COMPLETE.md` (if exists)
- [ ] `TEAM_134_RISK_ANALYSIS.md` (if exists)
- [ ] Any other Team 134 deliverables

---

## 🔍 PEER REVIEW PROCESS

### Step 1: Claim Inventory & Question Extraction (Day 1 Morning)

**Goal:** Extract every claim the other team made AND identify unanswered questions

**Process:**
1. Read ALL their documents
2. Extract EVERY claim they made
3. Extract EVERY question they left unanswered
4. Categorize claims:
   - Architecture claims (e.g., "module X depends on Y")
   - LOC claims (e.g., "file Z has 500 LOC")
   - Dependency claims (e.g., "uses crate A but not B")
   - Risk claims (e.g., "migration complexity is LOW")
   - Shared crate claims (e.g., "should use auth-min")
5. Categorize questions:
   - Unanswered questions (e.g., "How does X integrate with Y?")
   - TBD items (e.g., "Test coverage: TBD")
   - TODO notes (e.g., "TODO: Verify this")
   - Assumptions without verification (e.g., "Seems reasonable")

**Output:** Claim inventory list + Question inventory list

**Example:**
```markdown
## Claim Inventory: TEAM-132 queen-rbee

### Architecture Claims:
1. "Orchestrator depends on load-balancer and registry" (Report p.5)
2. "HTTP server is independent" (Report p.8)
3. "No circular dependencies" (Report p.12)

### LOC Claims:
1. "orchestrator.rs has 1,200 LOC" (Report p.3)
2. "Total LOC is 3,100" (Report p.2)

### Dependency Claims:
1. "Uses model-catalog for model info" (Report p.15)
2. "Does NOT use auth-min" (Report p.16)

### Risk Claims:
1. "Migration complexity: MEDIUM" (Report p.25)
2. "No breaking changes expected" (Report p.26)

## Question Inventory: TEAM-132 queen-rbee

### Unanswered Questions:
1. "How does queen-rbee handle worker failures?" (Report p.10 - NOT ANSWERED)
2. "What happens when all workers are busy?" (Report p.15 - NOT ANSWERED)
3. "Is there retry logic?" (Report p.18 - NOT ANSWERED)

### TBD Items:
1. "Test coverage: TBD" (Report p.22 - NOT INVESTIGATED)
2. "Integration points: TBD" (Report p.24 - NOT DOCUMENTED)
3. "Performance metrics: TBD" (Report p.27 - NOT MEASURED)

### TODO Notes:
1. "TODO: Verify auth implementation" (Report p.16 - NOT VERIFIED)
2. "TODO: Check error handling" (Report p.19 - NOT CHECKED)

### Assumptions Without Proof:
1. "Seems like it uses auth-min" (Report p.16 - NO PROOF PROVIDED)
2. "Probably handles timeouts correctly" (Report p.21 - NOT VERIFIED)
```

---

### Step 2: Claim Verification & Question Answering (Day 1 Afternoon + Day 2 Morning)

**Goal:** Verify EVERY claim with actual code AND answer unanswered questions

**Process:**
1. For each claim, find the actual code
2. Verify the claim is TRUE
3. If FALSE or INCOMPLETE, document it
4. Collect proof (file paths, line numbers, code snippets)
5. **For each unanswered question, investigate the codebase and answer it**
6. **Provide proof for your answers (code snippets, file locations)**

**Commands to Use:**
```bash
# Count LOC
cloc bin/[binary]/src --by-file

# Find dependencies
grep -r "use " bin/[binary]/src

# Find function calls
grep -rn "function_name" bin/[binary]/src

# Check if shared crate is used
grep -r "auth-min" bin/[binary]/Cargo.toml
grep -r "auth_min" bin/[binary]/src
```

**Output:** Claim verification report

**Example:**
```markdown
## Claim Verification: TEAM-132 queen-rbee

### Claim 1: "orchestrator.rs has 1,200 LOC"
- **Status:** ❌ INCORRECT
- **Actual:** 1,087 LOC (verified via cloc)
- **Proof:** 
  ```bash
  $ cloc bin/queen-rbee/src/orchestrator.rs
  Language  files  blank  comment  code
  Rust      1      203    45       1087
  ```
- **Impact:** LOC estimate off by ~10%, affects effort estimate

### Claim 2: "Uses model-catalog for model info"
- **Status:** ⚠️ PARTIALLY TRUE
- **Actual:** model-catalog is in Cargo.toml but barely used
- **Proof:**
  ```bash
  $ grep -r "model_catalog" bin/queen-rbee/src
  bin/queen-rbee/src/orchestrator.rs:5:use model_catalog::ModelInfo;
  bin/queen-rbee/src/orchestrator.rs:127:// TODO: Use ModelInfo
  ```
- **Impact:** Shared crate not fully utilized, opportunity missed

### Claim 3: "No circular dependencies"
- **Status:** ✅ VERIFIED
- **Proof:** Analyzed all `use` statements, no cycles found
- **Method:** Created dependency graph (see attached)

## Question Answering: TEAM-132 queen-rbee

### Question 1: "How does queen-rbee handle worker failures?"
- **Original Status:** NOT ANSWERED (Report p.10)
- **Our Answer:** ✅ ANSWERED
- **Finding:** Implemented in `src/orchestrator.rs:156-189`
- **Proof:**
  ```rust
  // File: bin/queen-rbee/src/orchestrator.rs
  // Lines: 156-189
  async fn handle_worker_failure(&self, worker_id: &str) -> Result<()> {
      warn!("Worker {} failed, removing from registry", worker_id);
      self.registry.deregister(worker_id).await?;
      Ok(())
  }
  ```
- **Impact:** Team 132 missed this critical error handling logic

### Question 2: "What happens when all workers are busy?"
- **Original Status:** NOT ANSWERED (Report p.15)
- **Our Answer:** ✅ ANSWERED
- **Finding:** Request queues with timeout in `src/load_balancer.rs:92-105`
- **Proof:** [code snippet showing queue implementation]
- **Impact:** Important architectural detail missed

### Question 3: "TODO: Verify auth implementation"
- **Original Status:** NOT VERIFIED (Report p.16)
- **Our Verification:** ❌ AUTH NOT IMPLEMENTED!
- **Finding:** auth-min is in Cargo.toml but NEVER USED
- **Proof:**
  ```bash
  $ grep -r "auth_min" bin/queen-rbee/src
  [no results]
  ```
- **Impact:** CRITICAL SECURITY GAP!
```

---

### Step 3: Gap Analysis (Day 2 Afternoon)

**Goal:** Find what the other team MISSED

**Areas to Check:**

#### 3.1 Missing Files
- [ ] Did they analyze ALL files?
- [ ] Did they count ALL LOC?
- [ ] Are there hidden modules?

**How to Check:**
```bash
# List all Rust files
find bin/[binary]/src -name "*.rs" -type f

# Compare with their file list
# Any files missing from their report?
```

#### 3.2 Missing Dependencies
- [ ] Did they audit ALL Cargo.toml dependencies?
- [ ] Did they check workspace dependencies?
- [ ] Did they find all `use` statements?

**How to Check:**
```bash
# List all dependencies
cat bin/[binary]/Cargo.toml | grep -A 100 "[dependencies]"

# Find all use statements
grep -rh "^use " bin/[binary]/src | sort | uniq
```

#### 3.3 Missing Shared Crate Opportunities ⚠️ CRITICAL!

**🔍 ACTIVELY SEARCH FOR ALL SHARED CRATES!**

- [ ] Did they check ALL 10+ shared crates?
- [ ] Did they find duplicate code?
- [ ] Did they identify consolidation opportunities?
- [ ] Did they verify shared crate usage claims?
- [ ] Did they find shared crates in Cargo.toml but not used?

**Complete Shared Crate List to Check:**
1. `hive-core` - Core hive types
2. `model-catalog` - Model management
3. `gpu-info` - GPU detection
4. `auth-min` - Authentication
5. `secrets-management` - Secrets
6. `input-validation` - Input validation
7. `audit-logging` - Audit logs
8. `deadline-propagation` - Deadlines
9. `narration-core` - Observability
10. `jwt-guardian` - JWT handling

**How to Check (DO THIS FOR EVERY SHARED CRATE!):**
```bash
# Step 1: List ALL shared crates
ls -la shared-crates/

# Step 2: For EACH shared crate, check if it's in Cargo.toml
grep -r "shared-crate-name" bin/[binary]/Cargo.toml

# Step 3: For EACH shared crate, check if it's actually USED in code
grep -r "shared_crate_name" bin/[binary]/src

# Step 4: Identify gaps
# - In Cargo.toml but NOT in src/ = UNUSED DEPENDENCY
# - NOT in Cargo.toml but SHOULD BE = MISSED OPPORTUNITY
# - In src/ but barely used = UNDERUTILIZED

# Example: Complete auth-min audit
echo "=== Checking auth-min ==="
grep -r "auth-min" bin/[binary]/Cargo.toml
echo "Cargo.toml result: $?"

grep -rn "auth_min" bin/[binary]/src
echo "Source usage result: $?"

grep -rn "use auth" bin/[binary]/src
echo "Auth import result: $?"

# Example: Search for authentication code that SHOULD use auth-min
grep -rn "bearer" bin/[binary]/src -i
grep -rn "token" bin/[binary]/src -i
grep -rn "authorization" bin/[binary]/src -i
# If you find auth code but NO auth-min usage = GAP!
```

**Questions to Answer for EACH Shared Crate:**

1. **Is it in Cargo.toml?** YES/NO
2. **Is it used in src/?** YES/NO  
3. **How many times is it used?** (count grep results)
4. **Is usage appropriate?** FULL/PARTIAL/NONE
5. **Should it be used more?** YES/NO
6. **Is there duplicate code it could replace?** YES/NO

**Document Your Findings:**
```markdown
### Shared Crate Audit: auth-min

**In Cargo.toml:** ✅ YES (line 23)
**Used in src/:** ❌ NO
**Usage count:** 0 occurrences
**Appropriateness:** ❌ NONE - Should be used!

**Evidence of missed opportunity:**
- Found manual bearer token parsing in src/http/middleware.rs:45
- Found manual auth header checking in src/http/routes.rs:89
- Found TODO comment "use auth-min here" in src/orchestrator.rs:127

**Impact:** CRITICAL - Auth not properly implemented!
**Recommendation:** MUST use auth-min for all authentication

---

### Shared Crate Audit: model-catalog

**In Cargo.toml:** ✅ YES (line 25)
**Used in src/:** ⚠️ PARTIALLY
**Usage count:** 2 occurrences (barely used)
**Appropriateness:** ⚠️ UNDERUTILIZED

**Evidence:**
- Only used for type import: `use model_catalog::ModelInfo`
- Model info is duplicated in src/types/mod.rs
- Should query catalog for model metadata

**Impact:** MEDIUM - Duplicate code exists
**Recommendation:** Use model-catalog more extensively
```

#### 3.4 Missing Integration Points
- [ ] Did they document how binary talks to other binaries?
- [ ] Did they check for shared types?
- [ ] Did they identify API contracts?

**How to Check:**
```bash
# Find HTTP client calls (integration points)
grep -rn "reqwest" bin/[binary]/src
grep -rn "Client::new" bin/[binary]/src

# Find shared types
grep -rn "::types::" bin/[binary]/src
```

#### 3.5 Missing Risks
- [ ] Did they identify ALL breaking changes?
- [ ] Did they assess test coverage?
- [ ] Did they consider rollback scenarios?

**How to Check:**
```bash
# Check test coverage
find bin/[binary] -name "*test*" -o -name "*tests*"

# Count test files vs source files
# Low ratio = HIGH RISK!
```

#### 3.6 Missing Circular Dependencies
- [ ] Did they actually verify no circular deps?
- [ ] Did they check module-level dependencies?
- [ ] Did they create a dependency graph?

**How to Check:**
```bash
# Extract all module dependencies
grep -rh "^use crate::" bin/[binary]/src | sort | uniq

# Build dependency graph manually
# Look for cycles!
```

**Output:** Gap analysis report

**Example:**
```markdown
## Gap Analysis: TEAM-132 queen-rbee

### Missing Files:
❌ **CRITICAL:** Team 132 missed `src/preflight/` directory!
- Files: `preflight/mod.rs`, `preflight/rbee_hive.rs`, `preflight/ssh.rs`
- Total LOC: ~200
- Impact: LOC estimate off by ~6%, affects architecture

### Missing Dependencies:
⚠️ Team 132 did NOT check workspace dependencies!
- Workspace deps: `tokio`, `reqwest`, `serde` (inherited)
- Impact: Incomplete dependency analysis

### Missing Shared Crate Opportunities:
❌ **MAJOR GAP:** `audit-logging` is in Cargo.toml but NEVER USED!
- Proof: 
  ```bash
  $ grep -r "audit_logging" bin/queen-rbee/Cargo.toml
  audit-logging = { path = "../shared-crates/audit-logging" }
  
  $ grep -r "audit_logging" bin/queen-rbee/src
  [no results]
  ```
- Recommendation: Either use it or remove it!

### Missing Integration Points:
⚠️ Team 132 did NOT document how queen-rbee talks to llm-worker-rbee!
- Found: HTTP client calls to worker endpoints (src/orchestrator.rs:245)
- Missing: Request/response format documentation
- Impact: Integration risks not assessed

### Missing Risks:
❌ **CRITICAL:** No test coverage analysis!
- Team 132 claimed "good test coverage" but provided NO PROOF
- Actual: Only 3 test files found in entire codebase
- Impact: Migration risk severely underestimated

### Missing Circular Dependencies:
✅ Team 132 correctly identified no circular dependencies
- Verified independently
- No gaps found in this area
```

---

### Step 4: Write Peer Review Report (Day 3)

**Goal:** Document all findings in a formal report

**File Naming:**
- `TEAM_131_PEER_REVIEW_OF_TEAM_132.md`
- `TEAM_131_PEER_REVIEW_OF_TEAM_134.md`
- etc.

**Report Template:**

```markdown
# TEAM-[X] PEER REVIEW OF TEAM-[Y]

**Reviewing Team:** TEAM-[X]  
**Reviewed Team:** TEAM-[Y]  
**Binary:** [binary name]  
**Date:** 2025-10-19  
**Reviewers:** [names]

---

## Executive Summary

**Overall Assessment:** PASS / PASS WITH CONCERNS / FAIL

**Key Findings:**
- [Major finding 1]
- [Major finding 2]
- [Major finding 3]

**Recommendation:** APPROVE / REQUEST REVISIONS / REJECT

---

## Documents Reviewed

- [ ] `TEAM_[Y]_[binary]_INVESTIGATION_REPORT.md` (XX pages)
- [ ] `TEAM_[Y]_DEPENDENCY_GRAPH.md` (if exists)
- [ ] `TEAM_[Y]_CRATE_PROPOSALS.md` (if exists)
- [ ] `TEAM_[Y]_RISK_ANALYSIS.md` (if exists)
- [ ] [List all reviewed documents]

**Total Documents Reviewed:** X  
**Total Pages Reviewed:** XX

---

## Claim Verification Results

### ✅ Verified Claims (Correct)

1. **Claim:** "[exact quote from report]"
   - **Location:** [Report page X]
   - **Verification:** [How we verified]
   - **Proof:** [Code location, command output]
   - **Status:** ✅ CORRECT

[List all correct claims]

### ❌ Incorrect Claims (Wrong)

1. **Claim:** "[exact quote from report]"
   - **Location:** [Report page X]
   - **Status:** ❌ INCORRECT
   - **Actual:** [What we found]
   - **Proof:** [Code location, command output, screenshots]
   - **Impact:** [How this affects the investigation]
   - **Recommendation:** [What they should do]

[List all incorrect claims]

### ⚠️ Incomplete Claims (Partially True)

1. **Claim:** "[exact quote from report]"
   - **Location:** [Report page X]
   - **Status:** ⚠️ PARTIALLY TRUE
   - **What's Correct:** [...]
   - **What's Missing:** [...]
   - **Proof:** [Code location, command output]
   - **Impact:** [How this affects the investigation]
   - **Recommendation:** [What they should add]

[List all incomplete claims]

---

## Gap Analysis

### Missing Files

❌ **CRITICAL GAPS:**
- [File/directory they missed]
- **LOC Impact:** [How much LOC they missed]
- **Proof:** [ls command output, file listing]
- **Recommendation:** [What they should do]

✅ **No Gaps:** [If all files were covered]

### Missing Dependencies

❌ **DEPENDENCY GAPS:**
- [Dependencies they missed]
- **Impact:** [How this affects crate proposals]
- **Proof:** [Cargo.toml excerpt, grep output]
- **Recommendation:** [What they should analyze]

### Missing Shared Crate Opportunities

❌ **SHARED CRATE GAPS:**
- [Shared crates they didn't check]
- [Shared crates in Cargo.toml but not used]
- [Duplicate code that should be in shared crates]
- **Proof:** [Code examples, grep output]
- **Recommendation:** [What they should investigate]

### Missing Integration Points

❌ **INTEGRATION GAPS:**
- [How binary talks to other binaries - not documented]
- [Shared types - not identified]
- [API contracts - not documented]
- **Proof:** [Code locations showing integration]
- **Recommendation:** [What they should document]

### Missing Risks

❌ **RISK GAPS:**
- [Risks they didn't identify]
- [Test coverage not assessed]
- [Breaking changes not documented]
- **Proof:** [Evidence of risks]
- **Recommendation:** [What risks they should assess]

### Missing Test Analysis

❌ **TEST COVERAGE GAPS:**
- [Claims about test coverage without proof]
- [Test files not counted]
- [Test gaps not identified]
- **Proof:** [Test file listing, coverage data]
- **Recommendation:** [What they should analyze]

---

## Crate Proposal Review

### Proposed Crates Assessment

For each proposed crate:

#### Crate 1: [crate-name]

**Proposal Summary:**
- **Purpose:** [from their report]
- **LOC:** [from their report]
- **Dependencies:** [from their report]

**Our Assessment:**
- **LOC Verification:** ✅ CORRECT / ❌ INCORRECT ([actual LOC])
- **Dependencies Verification:** ✅ CORRECT / ⚠️ INCOMPLETE
- **Boundary Justification:** ✅ STRONG / ⚠️ WEAK / ❌ POOR
- **Testability:** ✅ GOOD / ⚠️ ACCEPTABLE / ❌ POOR
- **Size Appropriateness:** ✅ GOOD / ⚠️ TOO LARGE / ❌ TOO SMALL

**Issues Found:**
- [Issue 1 with proof]
- [Issue 2 with proof]

**Recommendation:** APPROVE / REVISE / REJECT

[Repeat for each crate]

---

## Shared Crate Audit Review

### Their Findings:
- [List what they claimed about shared crates]

### Our Verification:

#### Shared Crate 1: [name]
- **Their Claim:** [what they said]
- **Our Finding:** ✅ CORRECT / ❌ INCORRECT / ⚠️ INCOMPLETE
- **Proof:** [our verification with code examples]
- **Gap:** [what they missed, if anything]

[Repeat for each shared crate]

### Additional Shared Crate Opportunities They Missed:
- [Opportunity 1 with proof]
- [Opportunity 2 with proof]

---

## Migration Strategy Review

### Their Plan:
- [Summary of their migration strategy]

### Our Assessment:

**Strengths:**
- [What they got right]

**Weaknesses:**
- [What's wrong or missing]

**Missing Steps:**
- [Steps they didn't include]

**Risk Underestimation:**
- [Risks they underestimated with proof]

**Recommendation:** APPROVE / REVISE / REJECT

---

## Risk Assessment Review

### Their Risks:
- [List risks they identified]

### Our Verification:

**Verified Risks:** ✅
- [Risks we agree with]

**Missing Risks:** ❌
- [Risks they didn't identify]
- **Proof:** [Evidence of these risks]
- **Severity:** LOW / MEDIUM / HIGH / CRITICAL

**Underestimated Risks:** ⚠️
- [Risks they marked LOW but should be HIGH]
- **Proof:** [Why we think it's higher risk]

---

## Detailed Findings

### Critical Issues (Must Fix)

#### Issue 1: [Title]
- **Severity:** CRITICAL
- **Location:** [Report page X]
- **Problem:** [Detailed description]
- **Proof:** [Code locations, command output, screenshots]
- **Impact:** [How this affects the project]
- **Recommendation:** [Specific fix required]

[Repeat for each critical issue]

### Major Issues (Should Fix)

[Same format as critical issues]

### Minor Issues (Nice to Fix)

[Same format as critical issues]

---

## Code Evidence

### Evidence 1: [Title]
```bash
# Command used
$ [command]

# Output
[actual output]
```

**Explanation:** [What this proves]

### Evidence 2: [Title]
```rust
// File: bin/[binary]/src/[file].rs
// Lines: [X-Y]

[code snippet]
```

**Explanation:** [What this proves]

[Include ALL evidence collected]

---

## Recommendations

### Required Changes (Must Do)

1. **[Recommendation 1]**
   - **Reason:** [Why this is required]
   - **Action:** [Specific steps they should take]
   - **Priority:** CRITICAL

[List all required changes]

### Suggested Improvements (Should Do)

[Same format]

### Optional Enhancements (Nice to Have)

[Same format]

---

## Overall Assessment

**Completeness:** [0-100%]
- Files analyzed: [X/Y]
- Dependencies checked: [X/Y]
- Shared crates audited: [X/Y]
- Risks identified: [X/estimated total]

**Accuracy:** [0-100%]
- Correct claims: [X/Y]
- Incorrect claims: [X/Y]
- Incomplete claims: [X/Y]

**Quality:** [0-100%]
- Documentation quality: [score]
- Evidence provided: [score]
- Justification strength: [score]

**Overall Score:** [0-100%]

**Decision:** APPROVE / REQUEST REVISIONS / REJECT

---

## Sign-off

**Reviewed by:**
- [Name 1] - [Date]
- [Name 2] - [Date]
- [Name 3] - [Date]

**Status:** COMPLETE

---

## Appendices

### Appendix A: Full Claim Inventory
[Complete list of all claims with locations]

### Appendix B: Dependency Graph (Our Version)
[Our independently created dependency graph]

### Appendix C: LOC Verification
[Complete LOC breakdown we verified]

### Appendix D: Test Coverage Analysis
[Our test coverage analysis]

### Appendix E: Code Evidence
[All code snippets, command outputs, screenshots]
```

---

## 🚨 CRITICAL RULES

### Be CRITICAL
- ❌ **DON'T** assume claims are correct
- ✅ **DO** verify EVERYTHING with actual code
- ❌ **DON'T** accept "seems reasonable"
- ✅ **DO** demand proof for every claim

### Collect PROOF
- ❌ **DON'T** say "we checked and it's wrong"
- ✅ **DO** say "we ran `cloc` and got 1,087 LOC not 1,200"
- ❌ **DON'T** say "they missed files"
- ✅ **DO** say "they missed src/preflight/ with 200 LOC (see ls output)"

### Find GAPS
- ❌ **DON'T** only verify what they checked
- ✅ **DO** look for what they DIDN'T check
- ❌ **DON'T** assume they found everything
- ✅ **DO** independently analyze the codebase

### Document EVERYTHING
- ❌ **DON'T** write vague findings
- ✅ **DO** include file paths, line numbers, command outputs
- ❌ **DON'T** say "good job"
- ✅ **DO** provide specific, actionable feedback

### Search for Shared Crates ⚠️ MANDATORY
- ❌ **DON'T** only verify what they claimed about shared crates
- ✅ **DO** audit ALL 10+ shared crates independently
- ❌ **DON'T** assume shared crates are being used properly
- ✅ **DO** grep for each shared crate in Cargo.toml AND src/
- ✅ **DO** find code that SHOULD use shared crates but doesn't

### Answer Their Questions ⚠️ MANDATORY
- ❌ **DON'T** skip unanswered questions
- ✅ **DO** answer every question they left unanswered
- ❌ **DON'T** leave TBD/TODO items uninvestigated
- ✅ **DO** investigate the codebase and provide answers with proof
- ✅ **DO** challenge assumptions without verification

---

## ✅ PEER REVIEW CHECKLIST

### Day 1 Morning: Claim Inventory & Question Extraction
- [ ] Read ALL documents from reviewed team
- [ ] Extract EVERY claim (aim for 50+ claims)
- [ ] **Extract EVERY unanswered question**
- [ ] **Extract EVERY TBD/TODO item**
- [ ] Categorize claims by type
- [ ] Categorize questions by type
- [ ] Document claim locations (report page numbers)

### Day 1 Afternoon: Start Verification
- [ ] Set up verification environment
- [ ] Verify LOC claims (run cloc)
- [ ] Verify file structure claims (run find)
- [ ] Verify dependency claims (check Cargo.toml)

### Day 2 Morning: Continue Verification & Answer Questions
- [ ] Verify architecture claims (analyze code)
- [ ] Verify integration claims (grep for HTTP calls)
- [ ] **Verify shared crate claims (grep usage for EACH shared crate)**
- [ ] Verify risk claims (analyze test coverage)
- [ ] **Answer unanswered questions from their investigation**
- [ ] **Investigate TBD/TODO items they left incomplete**

### Day 2 Afternoon: Gap Analysis & Shared Crate Audit
- [ ] Find missing files
- [ ] Find missing dependencies
- [ ] **🔍 AUDIT ALL 10+ SHARED CRATES (see section 3.3)**
- [ ] **Find shared crates in Cargo.toml but not used**
- [ ] **Find code that SHOULD use shared crates but doesn't**
- [ ] Find missing integration points
- [ ] Find missing risks
- [ ] Find missing test analysis

### Day 3 Morning: Write Report (Part 1)
- [ ] Executive summary
- [ ] Documents reviewed
- [ ] Claim verification results
- [ ] Gap analysis

### Day 3 Afternoon: Write Report (Part 2)
- [ ] Crate proposal review
- [ ] Shared crate audit review
- [ ] Migration strategy review
- [ ] Risk assessment review
- [ ] Detailed findings
- [ ] Code evidence
- [ ] Recommendations
- [ ] Overall assessment

### Day 3 End: Finalize
- [ ] Peer review within your team
- [ ] Ensure ALL proof is included
- [ ] Check report completeness
- [ ] Submit to reviewed team

---

## 📊 SUCCESS CRITERIA

### Peer Review Complete When:
- [ ] ALL documents from reviewed team analyzed
- [ ] EVERY claim verified with proof
- [ ] **EVERY unanswered question answered with proof**
- [ ] **ALL 10+ shared crates audited independently**
- [ ] ALL gaps identified and documented
- [ ] Complete peer review report written
- [ ] Report includes code evidence for every finding
- [ ] Report reviewed by your own team
- [ ] Report submitted to reviewed team

### Quality Gates:
- ✅ At least 50 claims verified per review
- ✅ At least 10 gaps identified per review
- ✅ At least 20 pieces of code evidence per review
- ✅ **At least 10 questions answered per review**
- ✅ **ALL 10+ shared crates audited (with grep proof)**
- ✅ All findings have proof (file paths, line numbers, commands)
- ✅ Report is 20+ pages per review

---

## 📞 COMMUNICATION

### With Reviewed Team:
- ❌ **DON'T** discuss findings during review
- ✅ **DO** submit formal written report
- ❌ **DON'T** argue about findings
- ✅ **DO** provide proof and let them respond

### Within Your Team:
- ✅ **DO** collaborate on verification
- ✅ **DO** peer review each other's findings
- ✅ **DO** ensure quality before submitting

### Slack Channels:
- Use `#team-13x-peer-review` for your team's review work
- Use `#crate-decomposition-all` for cross-team questions

---

## 🎯 DELIVERABLES

### Per Team (2 reviews each):

**TEAM-131 must create:**
- `TEAM_131_PEER_REVIEW_OF_TEAM_132.md`
- `TEAM_131_PEER_REVIEW_OF_TEAM_134.md`

**TEAM-132 must create:**
- `TEAM_132_PEER_REVIEW_OF_TEAM_131.md`
- `TEAM_132_PEER_REVIEW_OF_TEAM_133.md`

**TEAM-133 must create:**
- `TEAM_133_PEER_REVIEW_OF_TEAM_132.md`
- `TEAM_133_PEER_REVIEW_OF_TEAM_134.md`

**TEAM-134 must create:**
- `TEAM_134_PEER_REVIEW_OF_TEAM_131.md`
- `TEAM_134_PEER_REVIEW_OF_TEAM_133.md`

**Total: 8 peer review reports**

---

## ✅ READY TO REVIEW!

**This is a CRITICAL QUALITY GATE!**

**Your peer reviews will:**
- Find mistakes before they become problems
- Ensure investigations are thorough
- Improve quality across all teams
- Prevent bad assumptions from propagating

**Be thorough. Be critical. Collect proof. Find gaps.**

**The success of the entire project depends on investigation quality!**

---

**Let's ensure these investigations are BULLETPROOF! 🔍**
