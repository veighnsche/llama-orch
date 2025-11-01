# TEAM-352: Queen UI Migration - Complete Guide

**Mission:** Migrate Queen UI from duplicate code to shared packages created by TEAM-356  
**Expected Time:** 4-6 hours  
**Priority:** HIGH (Validates shared packages before Hive/Worker use them)

---

## Quick Navigation

**⚠️ READ FIRST:** [Rule Zero Compliance](TEAM_352_RULE_ZERO_COMPLIANCE.md) - NO WRAPPERS, NO BACKWARD COMPATIBILITY

1. **[Step 1: SDK Loader Migration](TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md)** (45-60 min)
2. **[Step 2: Hooks Migration](TEAM_352_STEP_2_HOOKS_MIGRATION.md)** (60-90 min)
3. **[Step 3: Narration Migration](TEAM_352_STEP_3_NARRATION_MIGRATION.md)** (30-45 min)
4. **[Step 4: Config Cleanup](TEAM_352_STEP_4_CONFIG_CLEANUP.md)** (20-30 min)
5. **[Step 5: Testing & Verification](TEAM_352_STEP_5_TESTING.md)** (45-60 min)
6. **[Final Summary](TEAM_352_FINAL_SUMMARY.md)** (Documentation)

---

## Overview

### What We're Doing

Replacing Queen UI's duplicate code with 5 shared packages:

| Package | What it replaces | LOC saved |
|---------|------------------|-----------|
| @rbee/sdk-loader | Custom loader + globalSlot | ~125 LOC |
| @rbee/react-hooks | Custom async state management | ~153 LOC |
| @rbee/shared-config | Hardcoded URLs | ~12 LOC |
| @rbee/narration-client | Custom SSE parsing | ~91 LOC |
| @rbee/dev-utils | Manual logging | ~12 LOC |
| **TOTAL** | | **~393 LOC** |

**Expected code reduction: 60%**

---

## Prerequisites

### TEAM-356 Must Be Complete

Before starting TEAM-352:

- [ ] @rbee/sdk-loader package exists and builds (34 tests passing)
- [ ] @rbee/react-hooks package exists and builds (19 tests passing)
- [ ] @rbee/shared-config package exists and builds
- [ ] @rbee/narration-client package exists and builds
- [ ] @rbee/dev-utils package exists and builds
- [ ] All packages installed: `pnpm install` in monorepo root

**Verify:**
```bash
cd frontend/packages
ls -d */  # Should see: sdk-loader/ react-hooks/ shared-config/ narration-client/ dev-utils/

# Test each package builds:
cd sdk-loader && pnpm build && cd ..
cd react-hooks && pnpm build && cd ..
cd shared-config && pnpm build && cd ..
cd narration-client && pnpm build && cd ..
cd dev-utils && pnpm build && cd ..
```

**If ANY package missing or fails to build: STOP and complete TEAM-356 first.**

---

## Step-by-Step Migration

### Step 1: SDK Loader Migration

**Time:** 45-60 minutes  
**File:** [TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md](TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md)

**What you'll do:**
- Add @rbee/sdk-loader dependency
- Replace `loader.ts` (~120 LOC → ~15 LOC)
- Delete `globalSlot.ts` (no longer needed)
- Verify SDK still loads

**Success criteria:**
- ✅ Package builds
- ✅ SDK loads successfully
- ✅ ~125 LOC removed

### Step 2: Hooks Migration

**Time:** 60-90 minutes  
**File:** [TEAM_352_STEP_2_HOOKS_MIGRATION.md](TEAM_352_STEP_2_HOOKS_MIGRATION.md)

**What you'll do:**
- Add @rbee/react-hooks and @rbee/shared-config dependencies
- Replace `useHeartbeat.ts` (~94 LOC → ~35 LOC)
- Replace `useRhaiScripts.ts` (~274 LOC → ~180 LOC)
- Remove hardcoded URLs from hooks

**Success criteria:**
- ✅ Package builds
- ✅ Heartbeat works
- ✅ RHAI IDE works
- ✅ ~153 LOC removed

### Step 3: Narration Migration

**Time:** 30-45 minutes  
**File:** [TEAM_352_STEP_3_NARRATION_MIGRATION.md](TEAM_352_STEP_3_NARRATION_MIGRATION.md)

**What you'll do:**
- Add @rbee/narration-client dependency
- Replace `narrationBridge.ts` (~111 LOC → ~20 LOC)
- Remove custom SSE parsing
- Remove custom postMessage logic

**Success criteria:**
- ✅ Package builds
- ✅ Narration flows Queen → Keeper
- ✅ [DONE] markers handled
- ✅ ~91 LOC removed

### Step 4: Config Cleanup

**Time:** 20-30 minutes  
**File:** [TEAM_352_STEP_4_CONFIG_CLEANUP.md](TEAM_352_STEP_4_CONFIG_CLEANUP.md)

**What you'll do:**
- Add @rbee/dev-utils dependency
- Replace startup logging in App.tsx
- Search for remaining hardcoded URLs
- Update Keeper UI (if needed)

**Success criteria:**
- ✅ No hardcoded ports remain
- ✅ Startup logs use shared utility
- ✅ ~12 LOC removed

### Step 5: Testing & Verification

**Time:** 45-60 minutes  
**File:** [TEAM_352_STEP_5_TESTING.md](TEAM_352_STEP_5_TESTING.md)

**What you'll do:**
- Test dev mode (all features)
- Test prod mode (embedded build)
- Test narration flow
- Test hot reload
- Document results

**Success criteria:**
- ✅ All dev tests pass
- ✅ All prod tests pass
- ✅ No regressions
- ✅ Performance acceptable

### Final Summary

**File:** [TEAM_352_FINAL_SUMMARY.md](TEAM_352_FINAL_SUMMARY.md)

**What you'll do:**
- Document total code reduction
- Document test results
- Create handoff for TEAM-353
- Verify all acceptance criteria

---

## Expected Results

### Code Reduction Summary

| Component | Before | After | Saved | % |
|-----------|--------|-------|-------|---|
| SDK Loader | 140 | 15 | 125 | 89% |
| Hooks | 368 | 215 | 153 | 42% |
| Narration | 111 | 20 | 91 | 82% |
| Config | 13 | 1 | 12 | 92% |
| **TOTAL** | **632** | **251** | **381** | **60%** |

### Quality Improvements

✅ Single source of truth for common patterns  
✅ Battle-tested code (70+ tests in shared packages)  
✅ Consistent error handling  
✅ Better TypeScript types  
✅ HMR-safe (hot reload works)  
✅ No duplicate code

---

## What Can Go Wrong

### Common Issues

1. **Shared packages not installed**
   - Fix: `pnpm install` in monorepo root
   - Verify each package builds

2. **TypeScript errors after migration**
   - Fix: Check import paths
   - Rebuild shared packages
   - Check types match

3. **Narration doesn't flow**
   - Fix: Verify SERVICES.queen exists in @rbee/narration-client
   - Check origins match
   - See TEAM_352_STEP_3 troubleshooting

4. **Hot reload broken**
   - Fix: Verify global slot is HMR-safe
   - Check for duplicate SDK loads

5. **Build fails**
   - Fix: Clean node_modules and rebuild
   - Check dependencies are correct
   - Verify all shared packages build first

---

## Engineering Rules Compliance

### 🔥 CRITICAL: Follow Rule Zero

**BREAKING CHANGES > BACKWARDS COMPATIBILITY**

**❌ BANNED PATTERNS (will cause rejection):**
```typescript
// ❌ Creating wrapper exports "for backward compatibility"
export const loadSDKOnce = queenSDKLoader.loadOnce  // WRONG!

// ❌ Re-exporting from shared packages
export { parseNarrationLine } from '@rbee/narration-client'  // WRONG!

// ❌ Keeping old code alongside new
export function loadSDK_old() { /* old code */ }  // WRONG!
export function loadSDK_new() { /* new code */ }  // WRONG!
```

**✅ REQUIRED PATTERNS:**
```typescript
// ✅ Import directly from shared packages in hooks
import { createSDKLoader } from '@rbee/sdk-loader'
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

// ✅ Delete old files or make them minimal (types only)
// loader.ts is now just: export type { RbeeSDK } from './types'

// ✅ Let compiler find all call sites, fix them
// Compilation errors are TEMPORARY, technical debt is PERMANENT
```

**When migrating:**
- ✅ **JUST UPDATE THE EXISTING FILES** - Don't create wrappers
- ✅ **DELETE deprecated code immediately** - Don't keep old code "for compatibility"
- ✅ **Fix compilation errors** - That's what the compiler is for
- ✅ **One way to do things** - Not 3 different APIs
- ❌ **NO WRAPPERS EVER** - Import directly from shared packages

**See:** [TEAM_352_RULE_ZERO_COMPLIANCE.md](TEAM_352_RULE_ZERO_COMPLIANCE.md) for detailed examples

### TEAM-352 Signatures

Add to ALL modified files:

```typescript
// TEAM-352: Migrated to use @rbee/[package-name]
// Old implementation: ~XXX LOC of [what it did]
// New implementation: ~YY LOC using shared package
// Reduction: ~ZZ LOC (XX%)
```

### No TODO Markers

❌ **BANNED:**
```typescript
// TODO: Migrate this to shared package
```

✅ **REQUIRED:**
- Migrate it NOW
- Delete old code
- No deferring to future teams

---

## When to Stop and Ask for Help

**STOP if:**

1. **TEAM-356 not complete** - Shared packages missing or broken
2. **Tests failing** - More than 1-2 test failures across shared packages
3. **Build errors** - Can't resolve after 30 min troubleshooting
4. **Narration broken** - Can't fix after following troubleshooting guides
5. **Unsure about pattern** - Better to ask than guess

**Ask TEAM-351 or TEAM-356 for help if:**
- Shared packages don't export what you need
- Types don't match
- API doesn't work as documented

---

## Success Checklist

Before declaring TEAM-352 complete:

- [ ] All 5 steps completed in order
- [ ] All modified files have TEAM-352 signatures
- [ ] No TODO markers in TEAM-352 code
- [ ] ~380 LOC removed (60% reduction)
- [ ] Queen UI builds without errors
- [ ] All features work (dev + prod modes)
- [ ] Narration flows correctly
- [ ] Hot reload works
- [ ] No regressions detected
- [ ] All tests documented in TEAM_352_STEP_5
- [ ] Final summary document complete
- [ ] Handoff to TEAM-353 ready

**If ALL boxes checked: TEAM-352 is COMPLETE!** ✅

---

## Time Tracking

Track your time for each step:

| Step | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Step 1 | 45-60 min | ____ min | |
| Step 2 | 60-90 min | ____ min | |
| Step 3 | 30-45 min | ____ min | |
| Step 4 | 20-30 min | ____ min | |
| Step 5 | 45-60 min | ____ min | |
| **TOTAL** | **4-6 hours** | **____ hours** | |

**Efficiency:** ____% (actual/estimated)

---

## Next Team (TEAM-353)

After TEAM-352 completes:

**Mission:** Implement Hive UI using same shared packages

**Expected:**
- Same 5 shared packages
- Similar code reduction (~60%)
- Follow TEAM-352 step-by-step guides
- **NO duplicate code!**

**Estimated time:** 5-7 hours (similar to Queen)

---

## Documentation Tree

```
TEAM_352_INDEX.md (YOU ARE HERE)
├── TEAM_352_RULE_ZERO_COMPLIANCE.md ⚠️ READ FIRST
├── TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md
├── TEAM_352_STEP_2_HOOKS_MIGRATION.md
├── TEAM_352_STEP_3_NARRATION_MIGRATION.md
├── TEAM_352_STEP_4_CONFIG_CLEANUP.md
├── TEAM_352_STEP_5_TESTING.md
└── TEAM_352_FINAL_SUMMARY.md
```

**⚠️ Read Rule Zero Compliance first, then start with Step 1 and follow in order.** ✅

---

**TEAM-352: Ready to migrate Queen UI! Let's prove the pattern works!** 🚀
