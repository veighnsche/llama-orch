# Dead Code Analysis: narration-core

**Created by:** TEAM-204  
**Date:** 2025-10-22  
**Crate Size:** 3,619 LOC (code) + 32 markdown docs  
**Status:** 🟡 MODERATE BLOAT

---

## TL;DR

**Code:** Mostly clean, ~200 LOC could be removed  
**Docs:** MASSIVE BLOAT - 32 markdown files, many obsolete  

**Recommendation:** Archive old team docs, keep redaction (still used in stderr)

---

## Code Analysis (3,619 LOC)

### File Breakdown

| File | LOC | Status | Notes |
|------|-----|--------|-------|
| `builder.rs` | 844 | ✅ ACTIVE | Core narration builder |
| `lib.rs` | 520 | ✅ ACTIVE | Main API |
| `sse_sink.rs` | 428 | ✅ ACTIVE | SSE broadcaster |
| `capture.rs` | 364 | ✅ ACTIVE | Test adapter (BDD) |
| `trace.rs` | 321 | ⚠️ FEATURE | Only used with `trace-enabled` |
| `http.rs` | 212 | ⚠️ FEATURE | HTTP header propagation |
| `redaction.rs` | 202 | ✅ ACTIVE | **Still used in stderr!** |
| `auto.rs` | 190 | ⚠️ FEATURE | Auto-injection (cloud profile) |
| `unicode.rs` | 158 | ✅ ACTIVE | Sanitization |
| `axum.rs` | 153 | ⚠️ FEATURE | Only with `axum` feature |
| `correlation.rs` | 126 | ✅ ACTIVE | Correlation IDs |
| `otel.rs` | 101 | ⚠️ FEATURE | Only with `otel` feature |

**Total:** 3,619 LOC

---

## What's Actually Dead?

### ❌ DEAD: Redaction in SSE (Removed by TEAM-204)

**What was removed:**
```rust
// REMOVED from sse_sink.rs:
use crate::{redact_secrets, RedactionPolicy};

// REMOVED from From<NarrationFields>:
let target = redact_secrets(&fields.target, ...);
let human = redact_secrets(&fields.human, ...);
```

**Impact:** -20 LOC in sse_sink.rs

---

### ✅ STILL ACTIVE: Redaction in stderr

**Location:** `lib.rs:433-440`

```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    // Apply redaction to human text (ORCH-3302)
    let human = redact_secrets(&fields.human, RedactionPolicy::default());
    let cute = fields.cute.as_ref().map(|c| redact_secrets(c, ...));
    let story = fields.story.as_ref().map(|s| redact_secrets(s, ...));
    
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
    // ...
}
```

**Why it's still there:**
- stderr goes to daemon logs
- Logs might be shared/stored
- Redaction prevents secrets in log files

**Should we remove it?**
- **NO** - stderr redaction is different from SSE redaction
- Logs are long-lived, SSE is ephemeral
- Compliance requirement for log storage

---

### ⚠️ QUESTIONABLE: Feature-gated code

**Modules only used with features:**

1. **`trace.rs` (321 LOC)** - Only with `trace-enabled` feature
   - Used by: BDD tests, dev builds
   - Production: Dead code
   - **Keep:** Useful for debugging

2. **`auto.rs` (190 LOC)** - Auto-injection for cloud profile
   - Used by: Cloud deployments
   - Homelab: Dead code
   - **Keep:** Future-proofing

3. **`axum.rs` (153 LOC)** - Axum integration
   - Used by: queen-rbee (has `axum` feature)
   - **Keep:** Active

4. **`otel.rs` (101 LOC)** - OpenTelemetry
   - Used by: Cloud profile
   - Homelab: Dead code
   - **Keep:** Future-proofing

5. **`http.rs` (212 LOC)** - HTTP header propagation
   - Used by: Distributed tracing
   - Current: Minimal use
   - **Keep:** Useful utility

**Total feature-gated:** ~977 LOC (27% of crate)

**Recommendation:** Keep all - they're properly feature-gated

---

## Documentation Bloat (32 files!)

### Active Docs (Keep)

1. ✅ `README.md` - Main documentation
2. ✅ `QUICK_START.md` - Getting started guide
3. ✅ `CHANGELOG.md` - Version history

**Total:** 3 files

---

### Team Summaries (Archive)

**From TEAMS 192-203:**
- `TEAM-192-SUMMARY.md`
- `TEAM-197-SUMMARY.md`
- `TEAM-197-ARCHITECTURE-REVIEW.md`
- `TEAM-199-SUMMARY.md`
- `TEAM-199-REDACTION-FIX.md`
- `TEAM-200-SUMMARY.md`
- `TEAM-200-JOB-SCOPED-SSE.md`
- `TEAM-201-SUMMARY.md`
- `TEAM-201-CENTRALIZED-FORMATTING.md`
- `TEAM-202-SUMMARY.md`
- `TEAM-202-HIVE-NARRATION.md`
- `TEAM-203-SUMMARY.md`
- `TEAM-203-VERIFICATION.md`
- `TEAM-203-REFACTORING-PLAN.md`

**Total:** 14 files  
**Recommendation:** Move to `.archive/teams/`

---

### Architecture Docs (Consolidate)

**Multiple overlapping docs:**
- `NARRATION_SSE_ARCHITECTURE_TEAM_198.md` (SUPERSEDED)
- `NARRATION_ARCHITECTURE_FINAL.md` (Current)
- `SSE_FORMATTING_ISSUE.md` (Historical)
- `START_HERE_TEAMS_199_203.md` (Historical)
- `IMPLEMENTATION_COMPLETE_TEAMS_199_203.md` (Historical)

**Recommendation:** Keep `NARRATION_ARCHITECTURE_FINAL.md`, archive rest

---

### TEAM-204 Docs (Keep Recent, Archive Old)

**Created by TEAM-204:**
1. ✅ `INCIDENT_REPORT_GLOBAL_CHANNEL.md` - **KEEP** (compliance)
2. ✅ `ISOLATION_ANALYSIS.md` - **KEEP** (explains current design)
3. ⚠️ `CRITICAL_REVIEW_BUGS.md` - Archive (superseded)
4. ⚠️ `FIXES_APPLIED.md` - Archive (superseded)
5. ⚠️ `REVIEW_SUMMARY.md` - Archive (superseded)
6. ⚠️ `CHANGES_MADE.md` - Archive (superseded)
7. ✅ `FINAL_CRITICAL_REVIEW.md` - **KEEP** (final verdict)
8. ✅ `SECURITY_FIX_GLOBAL_CHANNEL_REMOVED.md` - **KEEP** (security doc)
9. ✅ `TEAM-204-SUMMARY.md` - **KEEP** (team summary)
10. ✅ `TEAM-204-CLEANUP-COMPLETE.md` - **KEEP** (cleanup summary)

**Keep:** 6 files  
**Archive:** 4 files

---

## Recommended Actions

### 1. Archive Team Docs (18 files)

```bash
mkdir -p .archive/teams
mv TEAM-{192,197,199,200,201,202,203}-*.md .archive/teams/
mv START_HERE_TEAMS_199_203.md .archive/teams/
mv IMPLEMENTATION_COMPLETE_TEAMS_199_203.md .archive/teams/
mv CRITICAL_REVIEW_BUGS.md .archive/teams/
mv FIXES_APPLIED.md .archive/teams/
mv REVIEW_SUMMARY.md .archive/teams/
mv CHANGES_MADE.md .archive/teams/
```

**Savings:** 18 files moved to archive

---

### 2. Archive Superseded Architecture Docs (3 files)

```bash
mkdir -p .archive/architecture
mv NARRATION_SSE_ARCHITECTURE_TEAM_198.md .archive/architecture/
mv SSE_FORMATTING_ISSUE.md .archive/architecture/
```

**Savings:** 3 files moved to archive

---

### 3. Keep Active Docs (11 files)

**Core:**
- `README.md`
- `QUICK_START.md`
- `CHANGELOG.md`
- `NARRATION_ARCHITECTURE_FINAL.md`

**TEAM-204 (Security & Design):**
- `INCIDENT_REPORT_GLOBAL_CHANNEL.md`
- `SECURITY_FIX_GLOBAL_CHANNEL_REMOVED.md`
- `ISOLATION_ANALYSIS.md`
- `FINAL_CRITICAL_REVIEW.md`
- `TEAM-204-SUMMARY.md`
- `TEAM-204-CLEANUP-COMPLETE.md`

**Total:** 10 files (down from 32)

---

### 4. Code Cleanup (Optional)

**No dead code to remove!**

All code is either:
- ✅ Actively used
- ⚠️ Feature-gated (properly)
- ✅ Redaction still needed for stderr

**Recommendation:** Leave code as-is

---

## Summary

### Current State

| Category | Count | Size |
|----------|-------|------|
| Source files | 12 | 3,619 LOC |
| Markdown docs | 32 | ~250 KB |
| Total | 44 files | - |

### After Cleanup

| Category | Count | Size | Change |
|----------|-------|------|--------|
| Source files | 12 | 3,619 LOC | No change |
| Active docs | 10 | ~100 KB | -22 files |
| Archived docs | 21 | ~150 KB | Moved |
| Total | 10 active | - | **-68% docs** |

---

## Detailed Recommendations

### ✅ Keep All Code

**Reason:** No dead code found
- Feature-gated code is properly conditional
- Redaction still used in stderr path
- All modules have active consumers

### ✅ Archive 21 Documentation Files

**Categories:**
1. Team summaries (14 files) → `.archive/teams/`
2. Superseded architecture (3 files) → `.archive/architecture/`
3. Superseded TEAM-204 docs (4 files) → `.archive/teams/`

**Benefits:**
- Cleaner root directory
- Easier to find current docs
- Historical context preserved

### ✅ Keep 10 Active Docs

**Essential:**
- README, QUICK_START, CHANGELOG
- Current architecture doc
- Security incident report (compliance)
- TEAM-204 final summaries

---

## Why Redaction Stays

### Common Misconception

> "We removed global channel, so we don't need redaction anymore"

**WRONG!** Redaction serves two purposes:

1. **SSE redaction** (REMOVED) - Was for global channel privacy
2. **stderr redaction** (KEPT) - For log file security

### stderr Redaction is Still Needed

**Scenario:**
```rust
NARRATE
    .action("auth")
    .human("Connecting with api_key=sk-secret123")
    .emit();
```

**Without redaction:**
```
# /var/log/queen-rbee.log
[queen] auth: Connecting with api_key=sk-secret123
```

**With redaction:**
```
# /var/log/queen-rbee.log
[queen] auth: Connecting with api_key=[REDACTED]
```

**Why this matters:**
- Log files are stored long-term
- Logs might be shared with support
- Compliance requirements (GDPR, PCI-DSS)
- Different from ephemeral SSE streams

---

## Conclusion

**Code:** ✅ Clean, no dead code  
**Docs:** 🟡 Bloated, 21 files should be archived  

**Action Items:**
1. Archive 21 documentation files
2. Keep all code as-is
3. Update README to point to archived docs

**Estimated cleanup time:** 15 minutes  
**Benefit:** 68% reduction in active documentation

---

**END OF ANALYSIS**
