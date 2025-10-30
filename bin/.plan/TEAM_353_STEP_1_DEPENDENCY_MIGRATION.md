# TEAM-353 Step 1: Hive UI - Add Shared Package Dependencies

**Estimated Time:** 15-20 minutes  
**Priority:** CRITICAL  
**Previous Step:** None (first step)  
**Next Step:** TEAM_353_STEP_2_HOOKS_MIGRATION.md

---

## Mission

Add all shared package dependencies to the existing Hive UI packages.

**Location:** `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui`

---

## Deliverables Checklist

- [ ] Dependencies added to rbee-hive-react/package.json
- [ ] Dependencies added to app/package.json
- [ ] pnpm install succeeds
- [ ] TEAM-353 signatures added

---

## Step 1: Update rbee-hive-react Package

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/package.json`

**Current dependencies:**
```json
{
  "dependencies": {
    "@rbee/rbee-hive-sdk": "workspace:*"
  }
}
```

**Add these dependencies:**
```json
{
  "dependencies": {
    "@rbee/rbee-hive-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@tanstack/react-query": "^5.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.1.16",
    "@tanstack/react-query-devtools": "^5.0.0",
    "typescript": "^5.2.2"
  }
}
```

---

## Step 2: Update Hive App Package

**File:** `bin/20_rbee_hive/ui/app/package.json`

**Add these dependencies if not present:**
```json
{
  "dependencies": {
    "@rbee/rbee-hive-react": "workspace:*",
    "@rbee/rbee-hive-sdk": "workspace:*",
    "@rbee/ui": "workspace:*",
    "@rbee/dev-utils": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

---

## Step 3: Install Dependencies

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

**Expected output:**
```
Packages: +X
Progress: resolved XXX, reused XXX, downloaded X, added XXX
```

---

## Step 4: Verify Shared Packages Build

```bash
# Build shared packages if not already built
cd frontend/packages/sdk-loader
pnpm build

cd ../react-hooks
pnpm build

cd ../narration-client
pnpm build

cd ../shared-config
pnpm build

cd ../dev-utils
pnpm build
```

---

## Testing Checklist

- [ ] `pnpm install` - no errors
- [ ] All shared packages built
- [ ] No dependency conflicts
- [ ] TypeScript recognizes new imports

---

## Success Criteria

✅ All shared package dependencies added  
✅ pnpm install succeeds  
✅ No dependency conflicts  
✅ Ready for Step 2 (hooks migration)  
✅ TEAM-353 signatures added

---

## Next Step

Continue to **TEAM_353_STEP_2_HOOKS_MIGRATION.md** to migrate the hooks to use shared packages.

---

**TEAM-353 Step 1: Dependencies added!** ✅
