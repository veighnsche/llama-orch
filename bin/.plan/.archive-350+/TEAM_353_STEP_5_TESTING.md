# TEAM-353 Step 5: Hive UI - Testing & Verification

**Estimated Time:** 30-45 minutes  
**Priority:** CRITICAL  
**Previous Step:** TEAM_353_STEP_4_CONFIG_CLEANUP.md  
**Next Step:** None (final step)

---

## Mission

Comprehensive testing to ensure no regressions after migration.

**Location:** `bin/20_rbee_hive/ui/`

---

## Deliverables Checklist

- [ ] All builds pass
- [ ] Development mode works
- [ ] Production mode works
- [ ] Narration flows correctly
- [ ] No console errors
- [ ] No TypeScript errors
- [ ] Performance acceptable
- [ ] Test results documented

---

## Phase 1: Build Verification

### Step 1: Clean Build

```bash
cd bin/20_rbee_hive/ui

# Remove build artifacts
rm -rf packages/rbee-hive-react/dist
rm -rf app/dist

# Clean install
pnpm install
```

### Step 2: Build All Packages

```bash
# Build React package
cd packages/rbee-hive-react
pnpm build

# Build app
cd ../../app
pnpm build
```

**Expected:** ✅ All builds succeed with no errors

---

## Phase 2: Development Mode Testing

### Step 1: Start Services

**Terminal 1:** Hive backend
```bash
cargo run --bin rbee-hive
```

**Terminal 2:** Hive UI
```bash
cd bin/20_rbee_hive/ui/app
pnpm dev
```

**Terminal 3:** Keeper UI
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

### Step 2: Test Features

**Open:** http://localhost:7836

**Test checklist:**
- [ ] Page loads without errors
- [ ] useModels hook loads models
- [ ] useWorkers hook loads workers
- [ ] Auto-refetch works (workers update every 2s)
- [ ] Error states display correctly
- [ ] Loading states display correctly
- [ ] Hot reload works (make a change, verify update)

### Step 3: Test Narration Flow

**In Keeper (http://localhost:5173):**
- [ ] Navigate to Hive page
- [ ] Hive iframe loads
- [ ] Trigger an operation
- [ ] Narration appears in Keeper console
- [ ] Narration appears in Keeper narration panel

**Expected console logs:**
```
[Hive] Narration event: { actor: "rbee_hive", ... }
[rbee-hive] Sending to parent: ...
[Keeper] Received narration: { actor: "rbee_hive", ... }
```

---

## Phase 3: Production Mode Testing

### Step 1: Build for Production

```bash
cd bin/20_rbee_hive/ui/app
pnpm build
```

### Step 2: Preview Production Build

```bash
pnpm preview
```

**Open:** http://localhost:7836

### Step 3: Test Production Features

- [ ] Page loads without errors
- [ ] All features work
- [ ] Narration flows correctly
- [ ] No console errors
- [ ] Performance acceptable

---

## Phase 4: TypeScript Verification

```bash
cd bin/20_rbee_hive/ui

# Check React package
cd packages/rbee-hive-react
tsc --noEmit

# Check app
cd ../../app
tsc --noEmit
```

**Expected:** ✅ No TypeScript errors

---

## Phase 5: Code Quality Checks

### Check for Hardcoded URLs

```bash
cd bin/20_rbee_hive/ui/app
grep -r "localhost:[0-9]" src --include="*.ts" --include="*.tsx"
```

**Expected:** No matches (or only in console.log)

### Check for TODO Markers

```bash
grep -r "TODO" src --include="*.ts" --include="*.tsx"
```

**Expected:** Only intentional TODOs with context

### Check for TEAM-353 Signatures

```bash
grep -r "TEAM-353" src --include="*.ts" --include="*.tsx"
grep -r "TEAM-353" ../packages/rbee-hive-react/src --include="*.ts"
```

**Expected:** All modified files have TEAM-353 signatures

---

## Phase 6: Performance Testing

### Metrics to Check

**Development mode:**
- [ ] Initial load < 2 seconds
- [ ] Hot reload < 1 second
- [ ] Worker refetch < 500ms

**Production mode:**
- [ ] Initial load < 1 second
- [ ] Bundle size reasonable
- [ ] No memory leaks

---

## Test Results Documentation

**File:** `bin/.plan/TEAM_353_TEST_RESULTS.md`

```markdown
# TEAM-353 Hive UI Migration - Test Results

**Date:** [Date]  
**Tester:** TEAM-353  
**Status:** [PASS/FAIL]

## Build Tests

- [ ] rbee-hive-react builds: [PASS/FAIL]
- [ ] Hive UI app builds: [PASS/FAIL]
- [ ] TypeScript checks pass: [PASS/FAIL]

## Development Mode

- [ ] Page loads: [PASS/FAIL]
- [ ] useModels works: [PASS/FAIL]
- [ ] useWorkers works: [PASS/FAIL]
- [ ] Auto-refetch works: [PASS/FAIL]
- [ ] Hot reload works: [PASS/FAIL]

## Production Mode

- [ ] Production build succeeds: [PASS/FAIL]
- [ ] All features work: [PASS/FAIL]
- [ ] Performance acceptable: [PASS/FAIL]

## Narration Flow

- [ ] Narration appears in Hive console: [PASS/FAIL]
- [ ] Narration appears in Keeper console: [PASS/FAIL]
- [ ] Narration appears in Keeper panel: [PASS/FAIL]

## Code Quality

- [ ] No hardcoded URLs: [PASS/FAIL]
- [ ] No TypeScript errors: [PASS/FAIL]
- [ ] TEAM-353 signatures present: [PASS/FAIL]

## Issues Found

[List any issues discovered during testing]

## Conclusion

[Summary of test results and next steps]
```

---

## Critical Test Checklist

**Must verify ALL of these:**

✅ All builds pass  
✅ useModels hook works  
✅ useWorkers hook works  
✅ Auto-refetch works  
✅ Narration flows to Keeper  
✅ No console errors  
✅ No TypeScript errors  
✅ No hardcoded URLs  
✅ Hot reload works  
✅ Production build works  
✅ TEAM-353 signatures present

---

## Troubleshooting

### Issue: Narration doesn't appear in Keeper

**Check:**
1. SERVICES.hive config exists in @rbee/narration-client
2. Keeper's narrationListener includes Hive origins
3. Message type is 'NARRATION_EVENT'
4. Origins match (7835/7836)

### Issue: Workers don't auto-refetch

**Check:**
1. refetchInterval is set in useQuery
2. QueryClient is properly configured
3. No errors in console

### Issue: TypeScript errors

**Check:**
1. All shared packages are built
2. pnpm install ran successfully
3. tsconfig.json is correct

---

## Success Criteria

✅ All builds pass  
✅ All features work in dev mode  
✅ All features work in prod mode  
✅ Narration flows correctly  
✅ No console errors  
✅ No TypeScript errors  
✅ Performance acceptable  
✅ Test results documented  
✅ TEAM-353 signatures everywhere

---

## Final Handoff

**Document:** `bin/.plan/TEAM_353_MIGRATION_COMPLETE.md`

**Include:**
- Summary of changes
- Code savings analysis
- Test results
- Known issues (if any)
- Next steps (if any)

---

**TEAM-353 Step 5: Testing complete! Hive UI migration successful!** ✅
