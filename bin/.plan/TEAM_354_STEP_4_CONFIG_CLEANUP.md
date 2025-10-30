# TEAM-354 Step 4: Worker UI - Remove Hardcoded URLs

**Estimated Time:** 15-20 minutes  
**Priority:** HIGH  
**Previous Step:** TEAM_354_STEP_3_NARRATION_INTEGRATION.md  
**Next Step:** TEAM_354_STEP_5_TESTING.md

---

## Mission

Remove all hardcoded URLs and use @rbee/shared-config instead.

**Location:** `bin/30_llm_worker_rbee/ui/app/src/`

---

## Deliverables Checklist

- [ ] All hardcoded URLs found
- [ ] Replaced with @rbee/shared-config
- [ ] App builds successfully
- [ ] No hardcoded URLs remain
- [ ] TEAM-354 signatures added

---

## Step 1: Find Hardcoded URLs

```bash
cd bin/30_llm_worker_rbee/ui/app
grep -r "localhost:[0-9]" src --include="*.ts" --include="*.tsx"
```

**Common patterns to look for:**
- `http://localhost:8080`
- `http://localhost:7838`
- Hardcoded port numbers
- Hardcoded origins

---

## Step 2: Replace with Shared Config

### For Iframe URLs

**❌ WRONG (hardcoded):**
```typescript
const workerUrl = "http://localhost:8080"
```

**✅ RIGHT (shared config):**
```typescript
import { getIframeUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV
const workerUrl = getIframeUrl('worker', isDev)
```

### For Allowed Origins

**❌ WRONG (hardcoded):**
```typescript
const allowedOrigins = [
  "http://localhost:8080",
  "http://localhost:7838",
]
```

**✅ RIGHT (shared config):**
```typescript
import { getAllowedOrigins } from '@rbee/shared-config'

const allowedOrigins = getAllowedOrigins()
```

### For Service URLs

**❌ WRONG (hardcoded):**
```typescript
const baseUrl = "http://localhost:8080"
```

**✅ RIGHT (shared config):**
```typescript
import { getServiceUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV
const baseUrl = getServiceUrl('worker', isDev ? 'dev' : 'prod')
```

---

## Step 3: Update App Startup Logging

**File:** `bin/30_llm_worker_rbee/ui/app/src/App.tsx`

**❌ WRONG (hardcoded):**
```typescript
console.log('🔧 [WORKER UI] Running in DEVELOPMENT mode')
console.log('   - Vite dev server active')
console.log('   - Running on: http://localhost:7838')
```

**✅ RIGHT (shared logging):**
```typescript
import { logStartupMode } from '@rbee/dev-utils'

logStartupMode("WORKER UI", import.meta.env.DEV, 7838)
```

---

## Step 4: Update useInferenceWithNarration Hook

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/hooks/useInferenceWithNarration.ts`

**Add default URL from config:**

```typescript
import { getServiceUrl } from '@rbee/shared-config'

export function useInferenceWithNarration(
  baseUrl?: string // Make it optional
): UseInferenceWithNarrationResult {
  // TEAM-354: Use shared config for default URL
  const isDev = (import.meta as any).env?.DEV ?? false
  const defaultUrl = getServiceUrl('worker', isDev ? 'dev' : 'prod')
  const url = baseUrl || defaultUrl

  // ... rest of implementation
}
```

---

## Step 5: Build and Verify

```bash
# Build package
cd bin/30_llm_worker_rbee/ui/packages/rbee-worker-react
pnpm build

# Build app
cd ../app
pnpm build
```

---

## Step 6: Verify No Hardcoded URLs

```bash
cd bin/30_llm_worker_rbee/ui/app
grep -r "localhost:[0-9]" src --include="*.ts" --include="*.tsx"
```

**Expected:** Only console.log statements (acceptable) or none

---

## Testing Checklist

- [ ] All hardcoded URLs found
- [ ] Replaced with @rbee/shared-config
- [ ] `pnpm build` (rbee-worker-react) - success
- [ ] `pnpm build` (app) - success
- [ ] grep shows no hardcoded URLs (except logs)
- [ ] App starts correctly
- [ ] URLs resolve correctly in dev/prod
- [ ] TEAM-354 signatures added

---

## Success Criteria

✅ All hardcoded URLs removed  
✅ Uses @rbee/shared-config  
✅ App builds successfully  
✅ URLs work in dev and prod modes  
✅ TEAM-354 signatures added

---

## Next Step

Continue to **TEAM_354_STEP_5_TESTING.md** for comprehensive testing.

---

**TEAM-354 Step 4: Config cleanup complete!** ✅
