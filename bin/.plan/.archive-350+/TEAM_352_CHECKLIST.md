# TEAM-352 Missing Work Checklist

## 🔴 CRITICAL - Code Deletions

- [ ] **Delete `narrationBridge.ts` entirely**
  - File: `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts`
  - Current: Throws error at runtime
  - Required: DELETE THE FILE (let compiler catch imports)

- [ ] **Delete `loader.ts` entirely**
  - File: `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/loader.ts`
  - Current: Throws error at runtime
  - Required: DELETE THE FILE (let compiler catch imports)

- [ ] **Delete `utils.ts` entirely**
  - File: `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils.ts`
  - Status: Already deleted in diff ✅
  - Verify: No imports remain

- [ ] **Delete `globalSlot.ts` entirely**
  - File: `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/globalSlot.ts`
  - Status: Already deleted in diff ✅
  - Verify: No imports remain

---

## 🔴 CRITICAL - Dependency Cleanup

- [ ] **Remove unused React Query from queen-rbee-react**
  - File: `bin/10_queen_rbee/ui/packages/queen-rbee-react/package.json`
  - Remove: `"@tanstack/react-query": "^5.0.0"`
  - Remove: `"@tanstack/react-query-devtools": "^5.0.0"`
  - Reason: Not imported anywhere in the package

- [ ] **Verify React Query is actually used in react-hooks**
  - File: `frontend/packages/react-hooks/package.json`
  - Check: Is `@tanstack/react-query` used in `useSSEWithHealthCheck`?
  - If NO: Remove from react-hooks too
  - If YES: Document where and why

---

## 🟡 HIGH PRIORITY - Runtime Testing

- [ ] **Test HeartbeatMonitor SSE connection**
  ```bash
  # Start queen-rbee backend
  cargo run --bin queen-rbee
  
  # Start UI dev server
  cd bin/10_queen_rbee/ui/app
  pnpm dev
  
  # Verify in browser:
  # 1. SSE connection opens to /v1/heartbeat
  # 2. Heartbeat data updates every 5s
  # 3. Connection status shows "connected"
  # 4. No CORS errors in console
  ```

- [ ] **Test RHAI script execution**
  ```bash
  # In UI:
  # 1. Open RHAI IDE
  # 2. Write simple script: `print("test")`
  # 3. Click "Run Script"
  # 4. Verify SSE stream shows execution
  # 5. Verify narration events appear
  ```

- [ ] **Test health check before SSE**
  ```bash
  # Stop queen-rbee backend
  # Reload UI
  # Verify:
  # 1. Health check fails gracefully
  # 2. No CORS errors
  # 3. Error message shows "Queen is offline"
  # 4. Retry logic kicks in after 5s
  ```

- [ ] **Test narration bridge to parent**
  ```bash
  # If queen-rbee is embedded in rbee-keeper:
  # 1. Load queen UI in iframe
  # 2. Trigger narration event (run script)
  # 3. Verify parent receives postMessage
  # 4. Verify narration appears in keeper's panel
  ```

---

## 🟡 HIGH PRIORITY - Bundle Analysis

- [ ] **Measure bundle size BEFORE migration**
  ```bash
  git stash
  cd bin/10_queen_rbee/ui/app
  pnpm build
  # Record: dist/assets/index-*.js size
  git stash pop
  ```

- [ ] **Measure bundle size AFTER migration**
  ```bash
  cd bin/10_queen_rbee/ui/app
  pnpm build
  # Record: dist/assets/index-*.js size
  ```

- [ ] **Compare before/after**
  - Expected: Similar or smaller (removed ~300 LOC)
  - If larger: Investigate why
  - Document in single summary file

- [ ] **Analyze what's in the 622 KB bundle**
  ```bash
  cd bin/10_queen_rbee/ui/app
  pnpm build --mode production
  # Use rollup-plugin-visualizer or similar
  # Identify largest dependencies
  # Check for duplicates
  ```

- [ ] **Verify tree-shaking works**
  - Check: Is unused React Query code included?
  - Check: Are all shared packages tree-shakeable?
  - Check: Any duplicate React/ReactDOM?

---

## 🟡 HIGH PRIORITY - Documentation Cleanup

- [ ] **Delete excessive completion docs**
  - Keep: [TEAM_352_RULE_ZERO_FIX.md](cci:7://file:///home/vince/Projects/llama-orch/bin/.plan/TEAM_352_RULE_ZERO_FIX.md:0:0-0:0) (main summary)
  - Delete: `TEAM_352_STEP_1_COMPLETE.md`
  - Delete: `TEAM_352_STEP_2_COMPLETE.md`
  - Delete: `TEAM_352_STEP_3_COMPLETE.md`
  - Delete: `TEAM_352_STEP_4_COMPLETE.md`
  - Delete: [TEAM_352_STEP_5_TEST_RESULTS.md](cci:7://file:///home/vince/Projects/llama-orch/bin/.plan/TEAM_352_STEP_5_TEST_RESULTS.md:0:0-0:0)

- [ ] **Consolidate into single 2-page summary**
  - File: `bin/.plan/TEAM_352_MIGRATION_SUMMARY.md`
  - Include:
    - What changed (files modified/deleted)
    - Why (RULE ZERO compliance)
    - Before/after LOC
    - Before/after bundle size
    - Runtime test results
    - Breaking changes (import paths)
  - Max: 2 pages (engineering rules)

---

## 🟢 MEDIUM PRIORITY - Code Quality

- [ ] **Verify useSSEWithHealthCheck is actually generic**
  - File: `frontend/packages/react-hooks/src/useSSEWithHealthCheck.ts`
  - Check: Does it work with non-HeartbeatMonitor SSE sources?
  - Check: Is the API actually reusable?
  - If NO: Rename to `useHeartbeatMonitor` and move to queen-rbee-react

- [ ] **Check for hardcoded URLs**
  ```bash
  cd bin/10_queen_rbee/ui
  grep -r "localhost:783" --include="*.ts" --include="*.tsx"
  # Acceptable: Default params in hooks
  # Not acceptable: Hardcoded in components
  ```

- [ ] **Verify all imports resolve**
  ```bash
  cd bin/10_queen_rbee/ui/packages/queen-rbee-react
  pnpm build
  # Should pass without errors
  
  cd ../app
  pnpm build
  # Should pass without errors
  ```

- [ ] **Run TypeScript strict mode**
  ```bash
  cd bin/10_queen_rbee/ui/packages/queen-rbee-react
  tsc --noEmit --strict
  # Fix any strict mode violations
  ```

---

## 🟢 MEDIUM PRIORITY - Shared Package Verification

- [ ] **Verify @rbee/sdk-loader is tested**
  - File: `frontend/packages/sdk-loader/`
  - Check: Does it have unit tests?
  - Check: Does singleflight pattern work?
  - Check: Does retry logic work?

- [ ] **Verify @rbee/react-hooks is tested**
  - File: `frontend/packages/react-hooks/`
  - Check: Does useSSEWithHealthCheck have tests?
  - Check: Does health check work?
  - Check: Does retry work?

- [ ] **Verify @rbee/narration-client is tested**
  - File: `frontend/packages/narration-client/`
  - Check: Does createStreamHandler work?
  - Check: Does SERVICES.queen exist?
  - Check: Does it handle [DONE] marker?

- [ ] **Verify @rbee/dev-utils is tested**
  - File: `frontend/packages/dev-utils/`
  - Check: Does logStartupMode work?
  - Check: Is it actually shared (multiple consumers)?

---

## 🟢 LOW PRIORITY - Nice to Have

- [ ] **Add JSDoc to exported hooks**
  - File: `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts`
  - Add: Parameter descriptions
  - Add: Return value descriptions
  - Add: Usage examples

- [ ] **Add error boundary for SDK loading**
  ```tsx
  // In App.tsx or DashboardPage.tsx
  <ErrorBoundary fallback={<SDKLoadError />}>
    <DashboardPage />
  </ErrorBoundary>
  ```

- [ ] **Add loading state UI**
  - Currently: Just shows nothing while SDK loads
  - Better: Show spinner or skeleton

- [ ] **Add retry UI for failed connections**
  - Currently: Silent retry in background
  - Better: Show "Reconnecting..." with countdown

---

## 📋 Verification Checklist

After completing above tasks:

- [ ] **All files compile**
  ```bash
  cd bin/10_queen_rbee/ui
  pnpm build
  # Exit code: 0
  ```

- [ ] **No TypeScript errors**
  ```bash
  tsc --noEmit
  # Exit code: 0
  ```

- [ ] **No runtime errors in browser console**
  - Start backend + frontend
  - Open browser DevTools
  - Check: No red errors
  - Check: SSE connection works

- [ ] **Bundle size documented**
  - Before: XXX KB
  - After: XXX KB
  - Delta: +/- XXX KB
  - Explanation: Why it changed

- [ ] **Breaking changes documented**
  - Old import: `from '@rbee/queen-rbee-react/loader'`
  - New import: `from '@rbee/sdk-loader'`
  - Migration guide: How to update

- [ ] **Single summary doc created**
  - File: `TEAM_352_MIGRATION_SUMMARY.md`
  - Length: ≤ 2 pages
  - Contains: All critical info

---

## 🎯 Success Criteria

**This migration is COMPLETE when:**

1. ✅ Error-throwing files are DELETED (not throwing errors)
2. ✅ Unused dependencies are REMOVED
3. ✅ Runtime tests PASS (SSE, health check, narration)
4. ✅ Bundle size is DOCUMENTED (before/after comparison)
5. ✅ Documentation is CONSOLIDATED (1 file, max 2 pages)
6. ✅ All builds PASS (TypeScript, Vite, no errors)
7. ✅ UI works in browser (manual verification)

**Current status: 2/7 complete (builds pass, TypeScript passes)**