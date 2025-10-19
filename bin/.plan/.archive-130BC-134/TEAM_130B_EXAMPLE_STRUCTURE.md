# FINAL INVESTIGATION FILE STRUCTURE

**📖 NOTE: This document is a TEMPLATE/REFERENCE for file structure.**

**👉 For when to write these files, see: `TEAM_130B_PHASED_APPROACH.md`**

---

Each binary has **3 SEPARATE FILES** to avoid token limits:

## Part 1: TEAM_130B_FINAL_[binary]_PART1_METRICS.md
## Part 2: TEAM_130B_FINAL_[binary]_PART2_LIBRARIES.md
## Part 3: TEAM_130B_FINAL_[binary]_PART3_MIGRATION.md

---

## PART 1: Metrics & Crate Decomposition (15-20 pages)

### 1. Executive Summary (2 pages)
- Binary stats, LOC, proposed crates count
- Effort estimates (hours/days)
- Risk level, recommendation

### 2. Ground Truth Metrics (3-4 pages)
- LOC verification with cloc output
- Reconciliation of investigations + peer reviews
- File structure verification
- Dependency verification with cargo tree

### 3. Proposed Crate Decomposition (8-10 pages)
**For EACH crate:**
- Purpose, LOC, files included
- Dependencies (external + shared + internal)
- Public API preview
- Justification (5 criteria: single responsibility, right size, testable, reusable, clear boundaries)
- Changes from original investigation
- External library recommendations (3-5 per crate)
- Approval status

---

## PART 2: Shared Crates & External Libraries (15-20 pages)

### 4. Shared Crate Usage Analysis (5-7 pages)
**For EACH of 10+ shared crates:**
- Current usage in THIS binary
- Assessment (FULL/PARTIAL/MINIMAL/NONE)
- Recommendations

**Missing opportunities:**
- Which shared crates should be used but aren't
- Evidence with code snippets
- Migration effort

**Cross-binary recommendations:**
- How shared crates should be used across ALL 4 binaries
- Action items for THIS binary

### 5. External Rust Library Recommendations (5-6 pages)
- **Add new:** rstest, clap, russh, indicatif, etc. (5+ recommendations)
- **Replace existing:** with better alternatives
- **Remove unused:** dependencies
- **Update outdated:** with migration notes

---

## PART 3: Migration & Approval (10-15 pages)

### 6. Architecture Decisions (2-3 pages)
- Module boundaries finalized
- Dependency flow (no circular deps)
- Integration points with other binaries

### 7. Migration Strategy (6-8 pages)
**Step-by-step with commands:**
- Phase 1: Skeletons (bash commands)
- Phase 2: Move code (order, validation)
- Phase 3: Testing (test commands)
- Phase 4: Cleanup (final checks)
**Each step:** time estimate, validation criteria

### 8. Risk Assessment (3-4 pages)
- High/Medium/Low risks
- Mitigation strategies
- Rollback plans

### 9. Test Coverage (2 pages)
- Current coverage %
- Recommended tests to add

### 10. Open Questions Answered (2-3 pages)
- Every question from investigations
- Answers with code proof

### 11. Recommendations Summary (1-2 pages)
- Critical/Important/Optional
- Owner, deadline, effort

### 12. Approval (1 page)
- Quality scores, readiness, sign-off

**Total per binary: 40-55 pages (split into 3 files)**
**Total for 4 binaries: 160-220 pages (12 files)**
**Plus cross-binary analysis: 20-25 pages (1 file)**
**Grand total: ~220 pages across 13 files**

---

## Key Focus Areas

### CRATE DESIGN (not binary functionality)
- Size, boundaries, dependencies
- Public APIs
- Reusability across project

### EXTERNAL LIBRARIES
- Suggest 5+ Rust crates per binary
- Concrete versions, code examples
- Migration effort

### CROSS-BINARY ANALYSIS
- Shared crate opportunities
- Standardization
- Code sharing

### ACTIONABILITY
- Step-by-step migration
- Exact commands
- Time estimates
- Validation criteria
