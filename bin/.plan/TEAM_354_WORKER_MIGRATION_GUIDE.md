# TEAM-354: Worker UI Migration to Shared Packages

**Status:** 📋 READY FOR MIGRATION  
**Estimated Time:** 2-3 hours  
**Priority:** HIGH  
**Pattern:** Follow TEAM-352 (Queen migration)

---

## ⚠️ CRITICAL: Packages Already Exist!

**Location:** `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/ui`

**Existing structure:**
```
bin/30_llm_worker_rbee/ui/
├── app/                        # Worker UI app (already exists)
└── packages/
    ├── rbee-worker-react/      # React hooks (already exists)
    └── rbee-worker-sdk/        # WASM SDK (already exists)
```

**This is a MIGRATION, not a new implementation!**

---

## Migration Steps

### Step 1: Add Shared Package Dependencies

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/package.json`

**Add these dependencies:**
```json
{
  "dependencies": {
    "@rbee/rbee-worker-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

### Step 2: Migrate Hooks to TanStack Query

**Replace manual state management with TanStack Query**

### Step 3: Add Narration Support

**Use @rbee/narration-client for inference operations:**

```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

export function useInference() {
  const infer = async (prompt: string) => {
    const narrationHandler = createStreamHandler(SERVICES.worker, (event) => {
      console.log('[Worker] Narration:', event)
    }, {
      debug: true,
      validate: true,
    })

    // Use narration handler with inference
    // ...
  }

  return { infer }
}
```

### Step 4: Update App to Use QueryClient

**File:** `bin/30_llm_worker_rbee/ui/app/src/App.tsx`

```typescript
import { QueryClientProvider } from '@tanstack/react-query'
import { logStartupMode } from '@rbee/dev-utils'

logStartupMode("WORKER UI", import.meta.env.DEV, 7838)

// Add QueryClient setup
```

### Step 5: Remove Hardcoded URLs

**Use @rbee/shared-config for all URLs**

---

## Testing Checklist

- [ ] `pnpm install` - dependencies added
- [ ] `pnpm build` (rbee-worker-react) - success
- [ ] `pnpm build` (worker-ui app) - success
- [ ] `pnpm dev` - app starts
- [ ] Inference works
- [ ] Narration flows to Keeper
- [ ] No TypeScript errors
- [ ] TEAM-354 signatures added

---

## Reference

**Follow the same pattern as Queen migration:**
- `bin/.plan/TEAM_352_QUEEN_MIGRATION_PHASE.md`
- `bin/.plan/TEAM_353_HIVE_MIGRATION_GUIDE.md`

---

**TEAM-354: Migrate existing Worker UI to use shared packages!** 🚀
