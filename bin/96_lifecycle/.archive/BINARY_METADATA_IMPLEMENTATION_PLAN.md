# Binary Metadata Implementation Plan

**Goal:** Embed build mode (debug/release) directly in binaries to enable stateless dev/prod detection.

**Status:** Planning Phase  
**Date:** 2025-10-31

---

## Overview

Instead of maintaining state files, we'll embed metadata at compile-time and read it at runtime to determine which binary to use.

### **Key Principle: RULE ZERO**
- ✅ Break existing code if needed (pre-1.0)
- ✅ Delete deprecated patterns immediately
- ✅ One way to do things, not multiple

---

## Current State Analysis

### **Affected Binaries**
1. `queen-rbee` - Orchestrator daemon
2. `rbee-hive` - Worker manager daemon

### **Current Build Flow**
```
lifecycle-local::install_daemon()
  ↓
resolve_binary_path() → build_daemon()
  ↓
lifecycle-shared::build_daemon()
  ↓
#[cfg(debug_assertions)] → cargo build
#[cfg(not(debug_assertions))] → cargo build --release
```

### **Current Start Flow**
```
lifecycle-local::start_daemon()
  ↓
find_binary_command() → searches:
  1. target/debug/{daemon}
  2. target/release/{daemon}
  3. ~/.local/bin/{daemon}
  ↓
Returns FIRST match (problem: ambiguous!)
```

### **The Problem**
- If both `target/debug/queen-rbee` AND `~/.local/bin/queen-rbee` exist
- Current code picks `target/debug` (first match)
- But user installed production → should use `~/.local/bin`
- **No way to know which mode was installed!**

---

## Solution Architecture

### **Embed Metadata at Compile Time**

Each binary will contain:
- `BUILD_PROFILE` - "debug" or "release"
- `BUILD_TIMESTAMP` - When it was built
- Accessible via `--build-info` flag

### **Read Metadata at Runtime**

When starting a daemon:
1. Check if `~/.local/bin/{daemon}` exists
2. If yes, execute `{daemon} --build-info` to get mode
3. If mode == "release", use it (production install)
4. Otherwise, fall back to `target/debug` (development)

### **Data Flow**

```
Install Production:
  cargo build --release
  → Binary has BUILD_PROFILE="release" embedded
  → Copy to ~/.local/bin/
  → Done (no state file!)

Start:
  Check ~/.local/bin/queen-rbee exists? YES
  → Execute: ~/.local/bin/queen-rbee --build-info
  → Output: "release"
  → Use ~/.local/bin/queen-rbee ✅

Install Dev:
  cargo build
  → Binary has BUILD_PROFILE="debug" embedded
  → Keep in target/debug/
  → Done (no state file!)

Start:
  Check ~/.local/bin/queen-rbee exists? NO
  → Use target/debug/queen-rbee ✅
```

---

## Implementation Phases

### **Phase 1: Add Build Metadata** ✅
- Add `build.rs` to `queen-rbee` and `rbee-hive`
- Embed `BUILD_PROFILE` and `BUILD_TIMESTAMP`
- Add `--build-info` CLI flag to both binaries

**Files:**
- `bin/10_queen_rbee/build.rs` (NEW)
- `bin/10_queen_rbee/src/main.rs` (MODIFY)
- `bin/20_rbee_hive/build.rs` (NEW)
- `bin/20_rbee_hive/src/main.rs` (MODIFY)

### **Phase 2: Binary Mode Detection** ✅
- Create `get_binary_mode()` function
- Execute binary with `--build-info` flag
- Parse output to determine debug/release

**Files:**
- `bin/96_lifecycle/lifecycle-local/src/utils/binary.rs` (MODIFY)

### **Phase 3: Smart Binary Selection** ✅
- Update `find_binary_command()` logic
- Prefer production binary if installed
- Fall back to dev binary otherwise

**Files:**
- `bin/96_lifecycle/lifecycle-shared/src/lib.rs` (MODIFY)
- `bin/96_lifecycle/lifecycle-local/src/start.rs` (MODIFY)

### **Phase 4: Install Logic Update** ✅
- Production install → copy to `~/.local/bin/`
- Dev install → keep in `target/debug/`
- Remove any state file logic

**Files:**
- `bin/96_lifecycle/lifecycle-local/src/install.rs` (MODIFY)

### **Phase 5: Testing & Verification** ✅
- Test dev install + start
- Test prod install + start
- Test switching between modes
- Verify no state files created

---

## Detailed Phase Breakdown

Each phase has its own detailed document:
- `PHASE_1_BUILD_METADATA.md` - Embed metadata in binaries
- `PHASE_2_MODE_DETECTION.md` - Read metadata from binaries
- `PHASE_3_SMART_SELECTION.md` - Choose correct binary
- `PHASE_4_INSTALL_UPDATE.md` - Update install logic
- `PHASE_5_TESTING.md` - Comprehensive testing

---

## Breaking Changes (RULE ZERO Compliant)

### **What We're Breaking**
1. ❌ **DELETE** ambiguous binary search order
2. ❌ **DELETE** "first match wins" logic
3. ❌ **CHANGE** install behavior (prod → ~/.local/bin, dev → target/debug)

### **Why It's OK**
- ✅ Pre-1.0 software (v0.1.0)
- ✅ Compiler will catch all issues
- ✅ Better than permanent tech debt
- ✅ One clear way to do things

### **Migration Path**
- Users must reinstall after update
- Clear error messages if binary not found
- Documentation updated

---

## Success Criteria

### **Functional Requirements**
- ✅ Dev install keeps binary in `target/debug/`
- ✅ Prod install copies binary to `~/.local/bin/`
- ✅ Start uses correct binary based on install mode
- ✅ No state files created
- ✅ Binary metadata readable via `--build-info`

### **Non-Functional Requirements**
- ✅ No performance degradation (metadata check is fast)
- ✅ Clear error messages
- ✅ Backwards compatible with existing installs (graceful fallback)

---

## Risk Assessment

### **Low Risk**
- ✅ Metadata embedding (compile-time, can't fail at runtime)
- ✅ CLI flag addition (backwards compatible)

### **Medium Risk**
- ⚠️ Binary selection logic change (needs thorough testing)
- ⚠️ Install behavior change (users need to reinstall)

### **Mitigation**
- Comprehensive testing in Phase 5
- Clear narration messages
- Fallback to old behavior if metadata unavailable

---

## Timeline Estimate

- **Phase 1:** 30 minutes (build.rs + CLI flags)
- **Phase 2:** 20 minutes (mode detection function)
- **Phase 3:** 30 minutes (smart selection logic)
- **Phase 4:** 20 minutes (install updates)
- **Phase 5:** 40 minutes (testing)

**Total:** ~2.5 hours

---

## Next Steps

1. Read `PHASE_1_BUILD_METADATA.md`
2. Implement Phase 1
3. Test Phase 1
4. Move to Phase 2
5. Repeat until Phase 5 complete

---

**Let's build this! 🚀**
