# TEAM-334: Desktop Entry - Rule Zero Fix

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE  
**Rule:** RULE ZERO - Breaking changes > backwards compatibility

## The Violation

I initially created `scripts/launch-rbee-dev.sh` - a **duplicate launch script** that reimplemented functionality already present in the existing `rbee` wrapper script (TEAM-162).

### What Was Wrong

```bash
# ❌ ENTROPY - Created duplicate script
scripts/launch-rbee-dev.sh:
- Checks if binary exists
- Auto-builds if source changed
- Launches rbee-keeper

# ✅ ALREADY EXISTS - Root rbee script (TEAM-162)
rbee:
- Calls xtask rbee
- xtask handles auto-build and launch
- Single source of truth
```

**Problems:**
1. **Duplication:** Same functionality in two places
2. **Maintenance burden:** Two scripts to update when logic changes
3. **Ignored existing solution:** Didn't check for existing scripts first
4. **Entropy:** Created permanent technical debt

## The Fix (Rule Zero)

**DELETED** the duplicate script. Updated desktop entry to use existing `rbee` wrapper.

```diff
# Desktop entry before
- Exec=/home/vince/Projects/llama-orch/scripts/launch-rbee-dev.sh

# Desktop entry after
+ Exec=/home/vince/Projects/llama-orch/rbee
```

## Why This Is Correct

1. **Single source of truth:** Only `rbee` script (TEAM-162)
2. **No duplication:** Removed unnecessary wrapper
3. **Existing solution:** `rbee` → `xtask rbee` already handles auto-build
4. **Less maintenance:** One script to maintain

## Files

**Created:**
- `~/.local/share/applications/rbee-dev.desktop` (desktop entry)
- `DESKTOP_ENTRY.md` (documentation)

**Deleted:**
- ❌ `scripts/launch-rbee-dev.sh` (duplicate, entropy)

**Uses:**
- ✅ `rbee` (existing wrapper script, TEAM-162)

## Rule Zero Principle

> **Breaking changes are temporary. Entropy is forever.**

Instead of creating a new script "because it's easier," I should have:
1. **Searched for existing solutions** (found `rbee` script)
2. **Used the existing script** (single source of truth)
3. **Avoided duplication** (no new files needed)

The fix: Delete the duplicate, use what exists. Done.

## Desktop Entry Usage

**From Application Menu:**
1. Press Super key
2. Type "rbee"
3. Click "rbee (Development)"

**From Command Line:**
```bash
./rbee  # Uses existing wrapper
```

## Team Signatures

- TEAM-334: Desktop entry + Rule Zero fix (deleted duplicate script)
- TEAM-162: Original `rbee` wrapper script (reused correctly)
