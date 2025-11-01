# TEAM-354 Step 1: Worker UI - Add Shared Package Dependencies

**Estimated Time:** 15-20 minutes  
**Priority:** CRITICAL  
**Previous Step:** None (first step)  
**Next Step:** TEAM_354_STEP_2_HOOKS_MIGRATION.md

---

## Mission

Add all shared package dependencies to the existing Worker UI packages.

**Location:** `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/ui`

---

## Deliverables Checklist

- [ ] Dependencies added to rbee-worker-react/package.json
- [ ] Dependencies added to app/package.json
- [ ] pnpm install succeeds
- [ ] TEAM-354 signatures added

---

## Step 1: Check Current Structure

```bash
ls -la bin/30_llm_worker_rbee/ui/packages/
```

**Expected:**
- `rbee-worker-react/` (React hooks package)
- `rbee-worker-sdk/` (WASM SDK package)

---

## Step 2: Update rbee-worker-react Package

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/package.json`

**Add these dependencies:**
```json
{
  "name": "@rbee/rbee-worker-react",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch"
  },
  "peerDependencies": {
    "react": "^18.0.0 || ^19.0.0"
  },
  "dependencies": {
    "@rbee/rbee-worker-sdk": "workspace:*",
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

## Step 3: Update Worker App Package

**File:** `bin/30_llm_worker_rbee/ui/app/package.json`

**Add these dependencies if not present:**
```json
{
  "name": "@rbee/worker-ui",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 7838",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview --port 7838"
  },
  "dependencies": {
    "@rbee/rbee-worker-react": "workspace:*",
    "@rbee/rbee-worker-sdk": "workspace:*",
    "@rbee/ui": "workspace:*",
    "@rbee/dev-utils": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@tanstack/react-query": "^5.0.0",
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "lucide-react": "^0.545.0",
    "next-themes": "^0.4.6",
    "react": "^19.1.1",
    "react-dom": "^19.1.1",
    "tailwind-merge": "^3.3.1",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@repo/eslint-config": "workspace:*",
    "@repo/typescript-config": "workspace:*",
    "@tailwindcss/vite": "^4.1.14",
    "@types/node": "^24.6.0",
    "@types/react": "^19.1.16",
    "@types/react-dom": "^19.1.9",
    "@vitejs/plugin-react": "^5.0.4",
    "babel-plugin-react-compiler": "^19.1.0-rc.3",
    "eslint": "^9.36.0",
    "tailwindcss": "^4.1.14",
    "typescript": "~5.9.3",
    "vite": "npm:rolldown-vite@7.1.14",
    "vite-plugin-top-level-await": "^1.6.0",
    "vite-plugin-wasm": "^3.5.0"
  }
}
```

---

## Step 4: Install Dependencies

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

## Step 5: Verify Shared Packages Build

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
✅ TEAM-354 signatures added

---

## Next Step

Continue to **TEAM_354_STEP_2_HOOKS_MIGRATION.md** to migrate the hooks to use shared packages.

---

**TEAM-354 Step 1: Dependencies added!** ✅
