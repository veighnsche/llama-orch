# TEAM-356: Extraction Extravaganza Checklist

**Mission:** Create 2 shared packages and migrate Queen UI  
**Estimated Time:** 6-9 hours

---

## Phase 1: Create @rbee/sdk-loader

### Package Structure
- [ ] Create `frontend/packages/sdk-loader/` directory
- [ ] Create `package.json` with correct metadata
- [ ] Create `tsconfig.json` with strict mode
- [ ] Create `vitest.config.ts` for testing
- [ ] Add to `pnpm-workspace.yaml`

### Source Files
- [ ] Create `src/types.ts` - LoadOptions, SDKLoadResult, GlobalSlot
- [ ] Create `src/utils.ts` - sleep, addJitter, withTimeout, calculateBackoff
- [ ] Create `src/singleflight.ts` - getGlobalSlot, clearGlobalSlot
- [ ] Create `src/loader.ts` - loadSDK, loadSDKOnce, createSDKLoader
- [ ] Create `src/index.ts` - Export all public APIs

### Tests
- [ ] Create `src/loader.test.ts` - 20 tests
  - [ ] Test retry logic with exponential backoff
  - [ ] Test timeout handling
  - [ ] Test export validation
  - [ ] Test environment guards (SSR, WebAssembly)
  - [ ] Test error handling
- [ ] Create `src/singleflight.test.ts` - 10 tests
  - [ ] Test global slot creation
  - [ ] Test concurrent loads (only one executes)
  - [ ] Test error caching
  - [ ] Test slot clearing
- [ ] Create `src/utils.test.ts` - 10 tests
  - [ ] Test sleep function
  - [ ] Test jitter calculation
  - [ ] Test timeout wrapper
  - [ ] Test backoff calculation

### Documentation
- [ ] Create `README.md` with usage examples
- [ ] Add JSDoc comments to all public functions
- [ ] Document all types and interfaces

### Verification
- [ ] Run `pnpm install` - no errors
- [ ] Run `pnpm build` - compiles successfully
- [ ] Run `pnpm test` - 40/40 tests passing
- [ ] No TypeScript errors
- [ ] No `any` types (except controlled cases)

---

## Phase 2: Create @rbee/react-hooks

### Package Structure
- [ ] Create `frontend/packages/react-hooks/` directory
- [ ] Create `package.json` with React peer dependency
- [ ] Create `tsconfig.json` with strict mode
- [ ] Create `vitest.config.ts` for testing
- [ ] Add to `pnpm-workspace.yaml`

### Source Files
- [ ] Create `src/useAsyncState.ts`
  - [ ] AsyncStateOptions interface
  - [ ] AsyncStateResult interface
  - [ ] useAsyncState hook implementation
  - [ ] Mounted ref for cleanup
  - [ ] Refetch function
- [ ] Create `src/useSSEWithHealthCheck.ts`
  - [ ] Monitor interface
  - [ ] SSEHealthCheckOptions interface
  - [ ] SSEHealthCheckResult interface
  - [ ] useSSEWithHealthCheck hook implementation
  - [ ] Health check before SSE
  - [ ] Auto-retry logic
- [ ] Create `src/index.ts` - Export all hooks

### Tests
- [ ] Create `src/useAsyncState.test.ts` - 15 tests
  - [ ] Test async data loading
  - [ ] Test loading state transitions
  - [ ] Test error handling
  - [ ] Test cleanup on unmount
  - [ ] Test refetch functionality
  - [ ] Test skip option
  - [ ] Test callbacks (onSuccess, onError)
- [ ] Create `src/useSSEWithHealthCheck.test.ts` - 15 tests
  - [ ] Test health check before SSE
  - [ ] Test SSE connection
  - [ ] Test connection state
  - [ ] Test error handling
  - [ ] Test auto-retry logic
  - [ ] Test cleanup on unmount
  - [ ] Test retry function

### Documentation
- [ ] Create `README.md` with usage examples
- [ ] Add JSDoc comments to all hooks
- [ ] Document all options and return types

### Verification
- [ ] Run `pnpm install` - no errors
- [ ] Run `pnpm build` - compiles successfully
- [ ] Run `pnpm test` - 30/30 tests passing
- [ ] No TypeScript errors
- [ ] No React warnings

---

## Phase 3: Migrate Queen UI

### Update Dependencies
- [ ] Add `@rbee/sdk-loader` to Queen package.json
- [ ] Add `@rbee/react-hooks` to Queen package.json
- [ ] Run `pnpm install`

### Migrate SDK Loader
- [ ] Open `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/loader.ts`
- [ ] Replace custom loader with `createSDKLoader()`
- [ ] Import from `@rbee/sdk-loader`
- [ ] Update `loadSDK` export to use `queenSDKLoader.loadOnce`
- [ ] Delete old loader implementation (~100 lines)
- [ ] Delete `globalSlot.ts` (no longer needed)
- [ ] Add TEAM-356 signature to modified files

### Migrate useHeartbeat Hook
- [ ] Open `src/hooks/useHeartbeat.ts`
- [ ] Import `useSSEWithHealthCheck` from `@rbee/react-hooks`
- [ ] Import `getServiceUrl` from `@rbee/shared-config`
- [ ] Replace hardcoded URL with `getServiceUrl('queen', 'prod')`
- [ ] Replace custom implementation with `useSSEWithHealthCheck()`
- [ ] Delete old implementation (~60 lines)
- [ ] Add TEAM-356 signature

### Migrate useRhaiScripts Hook
- [ ] Open `src/hooks/useRhaiScripts.ts`
- [ ] Import `useAsyncState` from `@rbee/react-hooks`
- [ ] Import `getServiceUrl` from `@rbee/shared-config`
- [ ] Replace hardcoded URL with `getServiceUrl('queen', 'prod')`
- [ ] Replace list/load logic with `useAsyncState()`
- [ ] Keep save/delete/select functions (business logic)
- [ ] Delete old async boilerplate (~150 lines)
- [ ] Add TEAM-356 signature

### Fix All Hardcoded URLs
- [ ] Search for `'http://localhost:7833'` in Queen UI
- [ ] Replace with `getServiceUrl('queen', 'prod')`
- [ ] Search for `'http://localhost:7834'` in Queen UI
- [ ] Replace with `getServiceUrl('queen', 'dev')`
- [ ] Verify no hardcoded ports remain

---

## Phase 4: Testing & Verification

### Build Verification
- [ ] `cd bin/10_queen_rbee/ui/app && pnpm build` - success
- [ ] No TypeScript errors
- [ ] No build warnings
- [ ] Bundle size acceptable

### Dev Mode Testing
- [ ] Start Queen Vite dev server (`pnpm dev`)
- [ ] Start Queen backend (`cargo run --bin queen-rbee`)
- [ ] Start Keeper UI
- [ ] Navigate to Queen page
- [ ] Verify narration works
- [ ] Verify hot reload works
- [ ] Verify heartbeat monitoring works
- [ ] Verify RHAI IDE works
- [ ] Check console for errors

### Prod Mode Testing
- [ ] Build Queen UI (`pnpm build`)
- [ ] Start Queen backend in release mode
- [ ] Start Keeper UI
- [ ] Navigate to Queen page
- [ ] Verify narration works
- [ ] Verify heartbeat monitoring works
- [ ] Verify RHAI IDE works
- [ ] Check console for errors

### Regression Testing
- [ ] Test narration flow (Backend → Queen → Keeper)
- [ ] Test function name extraction
- [ ] Test [DONE] marker handling
- [ ] Test error boundaries
- [ ] Test SSE reconnection
- [ ] Test script save/load/delete
- [ ] Test script execution

---

## Phase 5: Documentation

### Code Documentation
- [ ] All new files have TEAM-356 signatures
- [ ] All functions have JSDoc comments
- [ ] All types documented
- [ ] No TODO markers

### Migration Documentation
- [ ] Create migration summary document
- [ ] Document before/after code examples
- [ ] Document lines saved (actual, not estimated)
- [ ] Document breaking changes (if any)

### Handoff Documentation
- [ ] Create TEAM_356_HANDOFF.md (≤2 pages)
- [ ] Summary of packages created
- [ ] Summary of Queen migration
- [ ] Verification checklist (all boxes checked)
- [ ] Next steps for TEAM-357 (Hive UI)

---

## Quality Gates

### Must Pass Before Declaring Complete

**Package Quality:**
- [ ] All 70 tests passing (40 sdk-loader + 30 react-hooks)
- [ ] 100% TypeScript compilation
- [ ] No `any` types (except controlled)
- [ ] READMEs with examples
- [ ] JSDoc on all public APIs

**Migration Quality:**
- [ ] Queen UI builds without errors
- [ ] Queen UI runs in dev mode
- [ ] Queen UI runs in prod mode
- [ ] All functionality preserved
- [ ] No regressions detected
- [ ] ~300-400 lines removed from Queen

**Documentation Quality:**
- [ ] Handoff ≤2 pages
- [ ] Code examples included
- [ ] Actual metrics (not estimates)
- [ ] All checklists complete

---

## Success Metrics

### Code Metrics
- **Packages created:** 2
- **Tests written:** 70
- **Lines in packages:** ~500
- **Lines removed from Queen:** ~300-400
- **Net savings:** Positive (prevents 800+ lines in Hive/Worker)

### Time Metrics
- **Estimated:** 6-9 hours
- **Actual:** ___ hours (fill in)
- **Efficiency:** ___ (actual/estimated)

### Quality Metrics
- **Tests passing:** 70/70
- **TypeScript errors:** 0
- **Build errors:** 0
- **Regressions:** 0

---

## Handoff Criteria

**Before handing off to TEAM-357:**

✅ Both packages created and tested  
✅ Queen UI migrated successfully  
✅ All tests passing  
✅ No regressions  
✅ Documentation complete  
✅ Handoff document ≤2 pages  
✅ All checklist items complete

**If ANY item is incomplete, TEAM-356 is NOT done.**

---

## Notes

### Common Pitfalls
- Don't forget to add packages to `pnpm-workspace.yaml`
- Don't forget peer dependencies (React for hooks)
- Don't skip tests - they validate the pattern works
- Don't skip Queen migration - it proves packages work
- Don't create new documentation files - update existing ones

### Testing Tips
- Use `@testing-library/react-hooks` for hook testing
- Mock `window.location` for environment tests
- Use `vi.fn()` for mocking callbacks
- Test cleanup with `unmount()`
- Test error cases, not just happy path

### Migration Tips
- Migrate one hook at a time
- Test after each migration
- Keep git commits small and focused
- Document lines removed (actual count)
- Don't delete old code until new code works

---

**TEAM-356: Complete this checklist before declaring victory!** ✅
