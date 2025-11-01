# TEAM-351: Correction Checklist

**Status:** 🔴 REQUIRES FIXES  
**Date:** Oct 29, 2025  
**Priority:** HIGH - Must fix before TEAM-352 migration

---

## CRITICAL: Fix Before Proceeding

### ❌ Issue 1: Port Configuration Duplication (CRITICAL)

**Problem:** Ports hardcoded in both `shared-config` and `narration-client`, violating "single source of truth" principle.

**Current State:**
```typescript
// shared-config/src/ports.ts
export const PORTS = {
  queen: { dev: 7834, prod: 7833, backend: 7833 },
  // ...
}

// narration-client/src/config.ts (DUPLICATE!)
export const SERVICES = {
  queen: { devPort: 7834, prodPort: 7833, ... },
  // ...
}
```

**Required Fix:**
- [ ] Add `@rbee/shared-config` as dependency to `narration-client/package.json`
- [ ] Import `PORTS` from `@rbee/shared-config` in `narration-client/src/config.ts`
- [ ] Derive `SERVICES` ports from `PORTS` constant
- [ ] Remove hardcoded port numbers from `SERVICES`
- [ ] Verify build still passes
- [ ] Update narration-client README with new dependency

**Example Fix:**
```typescript
// narration-client/src/config.ts
import { PORTS } from '@rbee/shared-config'

export const SERVICES: Record<ServiceName, ServiceConfig> = {
  queen: {
    name: 'queen-rbee',
    devPort: PORTS.queen.dev,      // Import from shared-config
    prodPort: PORTS.queen.prod,    // Import from shared-config
    keeperDevPort: PORTS.keeper.dev,
    keeperProdOrigin: '*',
  },
  // ... same pattern for hive and worker
}
```

**Verification:**
```bash
cd frontend/packages/narration-client
pnpm build  # Must pass
pnpm test   # When tests exist
```

---

## HIGH PRIORITY: Required for "Production-Ready" Claim

### ❌ Issue 2: Zero Tests Written

**Problem:** Claiming "production-ready" and "100% type safety" without any tests.

**Required Fix:**
- [ ] Create `shared-config/src/__tests__/ports.test.ts`
- [ ] Create `narration-client/src/__tests__/parser.test.ts`
- [ ] Create `iframe-bridge/src/__tests__/validator.test.ts`
- [ ] Create `dev-utils/src/__tests__/environment.test.ts`
- [ ] Add test scripts to all package.json files
- [ ] Install vitest or jest as dev dependency
- [ ] Run all tests and verify they pass
- [ ] Update documentation to remove "Testing Recommendations" and add "Tests Included"

**Minimum Test Coverage:**
- [ ] `getAllowedOrigins()` - returns unique origins, handles duplicates
- [ ] `getIframeUrl()` - throws error for keeper prod, handles null ports
- [ ] `getParentOrigin()` - validates port range, returns correct origin
- [ ] `parseNarrationLine()` - handles [DONE], parses JSON, handles errors
- [ ] `validateOrigin()` - handles wildcards, validates URLs
- [ ] `isDevelopment()` - works in different environments

**Test Script Example:**
```json
// package.json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage"
  },
  "devDependencies": {
    "vitest": "^1.0.0"
  }
}
```

---

### ❌ Issue 3: No Integration with Existing Code

**Problem:** Packages exist but aren't used anywhere. No proof they work.

**Required Fix:**
- [ ] Migrate Queen UI to use `@rbee/shared-config` (replace hardcoded ports)
- [ ] Migrate Queen UI to use `@rbee/narration-client` (replace narrationBridge)
- [ ] Test Queen UI in dev mode - verify narration works
- [ ] Test Queen UI in prod mode - verify narration works
- [ ] Document actual LOC reduction (not estimates)
- [ ] Update TEAM_350 documentation with new import paths
- [ ] Remove old narrationBridge code after migration

**Files to Migrate:**
```
bin/10_queen_rbee/ui/app/src/App.tsx
bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts
bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx
bin/00_rbee_keeper/ui/src/utils/narrationListener.ts
```

**Verification:**
- [ ] Queen UI starts without errors
- [ ] Narration flows from Queen → Keeper
- [ ] Hot reload still works
- [ ] No console errors
- [ ] No TypeScript errors

---

## MEDIUM PRIORITY: Code Quality Issues

### ⚠️ Issue 4: Over-Engineering - Remove Unnecessary Validation

**Problem:** Validating hardcoded constants at runtime adds overhead with zero benefit.

**Required Fix:**
- [ ] Remove module-load port validation in `shared-config/src/ports.ts` (lines 55-64)
- [ ] Remove service name validation in `narration-client/src/config.ts` (lines 44-50)
- [ ] Keep validation only for user input (e.g., `getParentOrigin(port)`)
- [ ] Update bug fix documentation to reflect this wasn't a "critical bug"

**Rationale:** TypeScript catches invalid hardcoded values at compile time. Runtime validation of constants is redundant.

**What to Keep:**
- Input validation: `getParentOrigin(port)` validates `port` parameter ✅
- Type guards: `isValidPort()` for external values ✅

**What to Remove:**
- Validating `PORTS` constant at module load ❌
- Validating `SERVICES` keys at module load ❌

---

### ⚠️ Issue 5: Remove Statistics Tracking Overhead

**Problem:** Tracking parse statistics adds memory overhead with no proven use case.

**Required Fix:**
- [ ] Remove `ParseStats` interface from `narration-client/src/parser.ts`
- [ ] Remove statistics tracking from `parseNarrationLine()`
- [ ] Remove `getParseStats()` and `resetParseStats()` functions
- [ ] Or: Make statistics opt-in via debug flag (disabled by default)
- [ ] Update documentation to remove statistics features

**Alternative:** If you want to keep statistics:
- [ ] Add `enableStats: boolean` parameter to `createStreamHandler()`
- [ ] Default: `false` (no overhead in production)
- [ ] Only track when explicitly enabled

---

### ⚠️ Issue 6: Documentation Bloat

**Problem:** 22+ markdown files with massive repetition for ~400 LOC of utility code.

**Required Fix:**
- [ ] Consolidate into 5 documents max:
  - `TEAM_351_IMPLEMENTATION_GUIDE.md` (comprehensive, one file)
  - `TEAM_351_HANDOFF.md` (2 pages max)
  - Package READMEs (4 files, one per package)
- [ ] Delete redundant step-by-step documents:
  - `TEAM_351_STEP_1_BUG_FIXES.md` → Merge into implementation guide
  - `TEAM_351_STEP_1_IMPROVEMENTS_SUMMARY.md` → Delete
  - `TEAM_351_STEP_1_COMPLETE.md` → Delete
  - Repeat for steps 2, 3, 4
- [ ] Delete marketing documents:
  - `TEAM_351_ALL_STEPS_COMPLETE.md` → Merge summary into handoff
  - `TEAM_351_TYPE_SAFETY_VERIFICATION.md` → Include in tests instead
- [ ] Keep only actionable, non-redundant documentation

**Documentation Structure:**
```
bin/.plan/
├── TEAM_351_IMPLEMENTATION_GUIDE.md  (all fixes, features, decisions)
└── TEAM_351_HANDOFF.md                (2 pages: summary + next steps)

frontend/packages/
├── shared-config/README.md            (usage examples)
├── narration-client/README.md         (usage examples)
├── iframe-bridge/README.md            (usage examples)
└── dev-utils/README.md                (usage examples)
```

---

### ⚠️ Issue 7: Misleading Metrics

**Problem:** Inflating "bug fixes" and "features" counts by calling basic functionality "bugs".

**Required Fix:**
- [ ] Rewrite metric claims to be accurate:
  - "40 bugs fixed" → "10 improvements implemented"
  - "35 features added" → "15 features added"
  - "100% backwards compatible" → "New packages (no prior version)"
- [ ] Separate actual bugs from features:
  - Bug: Port duplication between packages (fix this!)
  - Feature: HTTPS support (opt-in parameter)
  - Feature: Port validation for user input
  - Not a bug: Using a Set to prevent duplicates (basic)
- [ ] Remove self-congratulatory language ("production-ready", "mission accomplished")
- [ ] Use factual statements: "Packages built and ready for integration testing"

---

## LOW PRIORITY: Nice to Have

### 💡 Issue 8: Rust Codegen Could Be Simpler

**Current:** Parses TypeScript source with regex, extracts PORTS constant.

**Alternative:** Use JSON as intermediate format:
```bash
# Generate ports.json from TypeScript
pnpm build:ports-json

# Generate Rust from ports.json
pnpm generate:rust
```

**Benefit:** More reliable than regex parsing, easier to maintain.

**Decision:** Keep current implementation (works fine), but document this alternative.

---

### 💡 Issue 9: iframe-bridge May Be Over-Abstraction

**Current:** Wraps `postMessage` with factories and validators (~80 LOC).

**Alternative:** Inline utilities or simpler helpers:
```typescript
// Instead of factories, just export functions
export function sendMessage(message, origin) { ... }
export function validateOrigin(origin, allowed) { ... }
```

**Decision:** Keep current implementation if tests prove it's useful. Remove if Queen migration shows it adds no value.

---

## Verification Checklist

After completing fixes, verify:

### Build Verification
- [ ] `cd frontend/packages/shared-config && pnpm build` → ✅ PASS
- [ ] `cd frontend/packages/narration-client && pnpm build` → ✅ PASS
- [ ] `cd frontend/packages/iframe-bridge && pnpm build` → ✅ PASS
- [ ] `cd frontend/packages/dev-utils && pnpm build` → ✅ PASS
- [ ] No TypeScript errors in any package
- [ ] No circular dependency warnings

### Test Verification
- [ ] All packages have test files
- [ ] `pnpm test` passes in all packages
- [ ] Test coverage > 80% (or justify lower coverage)
- [ ] Tests include edge cases (null, invalid input, etc.)

### Integration Verification
- [ ] Queen UI uses shared-config for all port references
- [ ] Queen UI uses narration-client for SSE handling
- [ ] No hardcoded ports in Queen UI
- [ ] Narration flows correctly in dev mode
- [ ] Narration flows correctly in prod mode
- [ ] Hot reload still works

### Documentation Verification
- [ ] ≤ 5 markdown files total (not 22+)
- [ ] Each package has one README with examples
- [ ] No redundant/repetitive content
- [ ] Metrics are factual, not inflated
- [ ] "Production-ready" claim backed by tests

### Dependency Verification
- [ ] `narration-client` depends on `shared-config` (package.json)
- [ ] No port duplication anywhere
- [ ] `pnpm install` works without errors
- [ ] No unused dependencies

---

## Priority Order

**Week 1 (CRITICAL):**
1. Fix port duplication (Issue 1) - BLOCKER
2. Write minimum tests (Issue 2) - REQUIRED
3. Integrate with Queen UI (Issue 3) - VALIDATION

**Week 2 (HIGH):**
4. Remove over-engineering (Issue 4)
5. Fix metrics/documentation (Issues 6, 7)

**Week 3 (MEDIUM):**
6. Remove statistics overhead (Issue 5)
7. Verify everything works end-to-end

**Optional (LOW):**
8. Simplify Rust codegen (Issue 8)
9. Simplify iframe-bridge (Issue 9)

---

## Success Criteria

✅ No port duplication (single source of truth)  
✅ Tests written and passing (>80% coverage)  
✅ Integrated with Queen UI (real usage validation)  
✅ Documentation < 10 pages total  
✅ Metrics are factual  
✅ No over-engineering (removed unnecessary validation)  
✅ All builds pass  
✅ No TypeScript errors

---

## Handoff to Next Team

**After TEAM-351 completes these fixes:**

TEAM-352 can proceed with confidence:
- Packages are tested and proven to work
- No hidden duplication issues
- Real LOC reduction measured (not estimated)
- Pattern validated for Hive/Worker migration

**Before handoff:**
- [ ] All CRITICAL fixes complete
- [ ] All HIGH priority fixes complete
- [ ] Queen UI migration complete and tested
- [ ] Documentation reduced and accurate
- [ ] Tests passing
- [ ] Handoff document updated with actual metrics

---

## Notes

- This checklist is not optional. These are real issues that will cause problems.
- "Production-ready" requires tests. No exceptions.
- Port duplication defeats the entire purpose of shared-config.
- Documentation should help, not overwhelm. Quality > quantity.
- Metrics should be honest. Inflating numbers damages credibility.

---

**TEAM-351: Please complete this checklist before declaring the work "COMPLETE".**
