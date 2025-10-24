# TEAM-277 Instructions - Corrections Applied

**Date:** Oct 23, 2025  
**Status:** ✅ COMPLETE

---

## Summary

Applied 5 corrections to TEAM-277 instruction documents based on accuracy review.

---

## Corrections Applied

### 1. ✅ Fixed Part 2, Step 2.3: Clarified Package Operations Routing

**File:** `TEAM_277_INSTRUCTIONS_PART_2.md`, lines 103-112

**Issue:** Unclear what happens to package operations that aren't forwarded.

**Fix:** Added clarification that package operations are handled directly in queen-rbee's job_router.rs, not forwarded to hive.

**Before:**
```markdown
**Important:** Package operations are NOT forwarded. Update doc comment to mention this.
```

**After:**
```markdown
**Important:** Package operations are NOT forwarded to hive via `should_forward_to_hive()`.
They are handled directly in queen-rbee's job_router.rs (see Phase 3, Step 3.6).

Update the doc comment in `should_forward_to_hive()` to mention:
- Package operations are handled by queen (orchestration)
- Worker/Model operations are forwarded to hive (execution)
```

---

### 2. ✅ Fixed Part 3, Step 3.3: Added SSH Helper Reference

**File:** `TEAM_277_INSTRUCTIONS_PART_3.md`, lines 60-71

**Issue:** Worker installation pattern didn't reference existing SSH helpers.

**Fix:** Added import and reference to existing `ssh_helper.rs` module.

**Before:**
```rust
pub async fn install_worker_binary(
    hive: &HiveConfig,
    worker: &WorkerConfig,
) -> Result<()> {
    // 1. Connect via SSH
    let ssh = SshClient::connect(hive).await?;
```

**After:**
```rust
// TEAM-277: Use existing SSH helper from hive-lifecycle
use queen_rbee_hive_lifecycle::ssh_helper::SshClient;

pub async fn install_worker_binary(
    hive: &HiveConfig,
    worker: &WorkerConfig,
) -> Result<()> {
    // 1. Connect via SSH (reuse existing pattern)
    // See bin/15_queen_rbee_crates/hive-lifecycle/src/ssh_helper.rs for reference
    let ssh = SshClient::connect(hive).await?;
```

---

### 3. ✅ Fixed Part 3, Step 3.6: Added Config Loading Details

**File:** `TEAM_277_INSTRUCTIONS_PART_3.md`, lines 140-170

**Issue:** Referenced undefined `load_config()` function.

**Fix:** Added import and clarified `HivesConfig::load()` usage with conditional path handling.

**Before:**
```rust
Operation::PackageSync { config_path, dry_run, remove_extra, force } => {
    let config = load_config(config_path)?;
```

**After:**
```rust
// Add import at top:
use rbee_config::declarative::HivesConfig;

Operation::PackageSync { config_path, dry_run, remove_extra, force } => {
    // Load declarative config
    let config = if let Some(path) = config_path {
        HivesConfig::load_from(&path)?
    } else {
        HivesConfig::load()?  // Loads from ~/.config/rbee/hives.conf
    };
```

---

### 4. ✅ Fixed Part 4, Phase 4.1: Clarified Stub Files Should Not Be Deleted

**File:** `TEAM_277_INSTRUCTIONS_PART_4.md`, lines 13-21

**Issue:** Instructed to delete `install.rs` and `uninstall.rs`, but these are API stubs that should remain.

**Fix:** Changed from deletion to documentation update.

**Before:**
```markdown
### Step 4.1: Delete Worker Installation Files

```bash
rm bin/25_rbee_hive_crates/worker-lifecycle/src/install.rs
rm bin/25_rbee_hive_crates/worker-lifecycle/src/uninstall.rs
```
```

**After:**
```markdown
### Step 4.1: Worker Installation is Already Stubbed

**Note:** `install.rs` and `uninstall.rs` are already stubs (TEAM-276).
They delegate to worker-catalog and don't perform actual installation.

**Action:** Update documentation in these files to clarify that:
- Queen will handle worker installation via SSH (new TEAM-277 pattern)
- These stubs remain for API consistency
- No deletion needed - they're part of the public API
```

---

### 5. ✅ Fixed Part 4, Step 5.2: Corrected Job Client API

**File:** `TEAM_277_INSTRUCTIONS_PART_4.md`, lines 97-130

**Issue:** Referenced non-existent `QueenClient` struct.

**Fix:** Updated to use correct `job_client::submit_and_stream_job()` function.

**Before:**
```rust
use crate::queen_client::QueenClient;

pub async fn sync(...) -> Result<()> {
    let client = QueenClient::new("http://localhost:8500");
    
    let operation = Operation::PackageSync { ... };
    
    client.submit_and_stream(operation).await?;
```

**After:**
```rust
use crate::job_client::submit_and_stream_job;
use crate::config::Config;

pub async fn sync(...) -> Result<()> {
    let config = Config::load()?;
    let queen_url = config.queen_url();
    
    let operation = Operation::PackageSync { ... };
    
    submit_and_stream_job(&queen_url, operation).await?;
```

---

## Additional Improvements

### Created Comprehensive Checklist

**File:** `TEAM_277_CHECKLIST.md`

Created a detailed, actionable checklist with:
- 150+ individual tasks across all 6 phases
- Checkbox format for easy tracking
- Status markers (⏳ TODO, 🔄 IN PROGRESS, ✅ DONE)
- Success criteria for each phase
- Blockers section
- Cumulative progress tracker
- Quick reference section

---

## Verification

All corrections have been applied and verified:

- ✅ Part 2: Package operations routing clarified
- ✅ Part 3: SSH helper reference added
- ✅ Part 3: Config loading clarified
- ✅ Part 4: Stub files clarification added
- ✅ Part 4: Job client API corrected
- ✅ Checklist created with all tasks

---

## Files Modified

1. `.docs/TEAM_277_INSTRUCTIONS_PART_2.md` - Step 2.3 clarification
2. `.docs/TEAM_277_INSTRUCTIONS_PART_3.md` - Steps 3.3 and 3.6 improvements
3. `.docs/TEAM_277_INSTRUCTIONS_PART_4.md` - Steps 4.1, 4.2, and 5.2 corrections

## Files Created

1. `.docs/TEAM_277_CHECKLIST.md` - Complete task checklist (150+ items)
2. `.docs/TEAM_277_CORRECTIONS_APPLIED.md` - This document

---

## Next Steps for TEAM-277

1. Read all instruction documents in order (Parts 1-4)
2. Use `TEAM_277_CHECKLIST.md` to track progress
3. Check off items as you complete them
4. Update status markers and progress tracker
5. Follow the corrected patterns for:
   - SSH client usage
   - Config loading
   - Job submission
   - Worker lifecycle stubs

**Good luck! 🚀**
