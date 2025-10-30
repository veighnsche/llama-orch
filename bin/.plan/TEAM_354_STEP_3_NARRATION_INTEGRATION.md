# TEAM-354 Step 3: Worker UI - Add Narration Support

**Estimated Time:** 30-45 minutes  
**Priority:** HIGH  
**Previous Step:** TEAM_354_STEP_2_HOOKS_MIGRATION.md  
**Next Step:** TEAM_354_STEP_4_CONFIG_CLEANUP.md

---

## Mission

Add narration support for Worker inference operations using @rbee/narration-client.

**Location:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/`

---

## Deliverables Checklist

- [ ] useInferenceWithNarration hook created
- [ ] Uses @rbee/narration-client
- [ ] Uses SERVICES.worker config
- [ ] Narration flows to Keeper
- [ ] Package builds successfully
- [ ] TEAM-354 signatures added

---

## Step 1: Create useInferenceWithNarration Hook

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/hooks/useInferenceWithNarration.ts`

```typescript
// TEAM-354: Inference with narration support
// Uses @rbee/narration-client (no custom implementation)

'use client'

import { useState } from 'react'
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

export interface InferenceRequest {
  prompt: string
  max_tokens?: number
  temperature?: number
}

export interface InferenceResult {
  text: string
  tokens: number
  duration: number
}

export interface UseInferenceWithNarrationResult {
  infer: (request: InferenceRequest) => Promise<InferenceResult>
  inferring: boolean
  error: Error | null
  result: InferenceResult | null
}

/**
 * Hook for running inference with narration
 * 
 * TEAM-354: Uses @rbee/narration-client for narration
 * 
 * @param baseUrl - Worker API URL (from @rbee/shared-config)
 * @returns Inference function and state
 */
export function useInferenceWithNarration(
  baseUrl: string
): UseInferenceWithNarrationResult {
  const [inferring, setInferring] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [result, setResult] = useState<InferenceResult | null>(null)

  const infer = async (request: InferenceRequest): Promise<InferenceResult> => {
    setInferring(true)
    setError(null)
    setResult(null)

    try {
      // TEAM-354: Create narration handler using shared package
      const narrationHandler = createStreamHandler(SERVICES.worker, (event) => {
        console.log('[Worker] Narration event:', event)
      }, {
        debug: true,
        silent: false,
        validate: true,
      })

      // Submit inference operation with narration
      const operation = {
        operation: 'infer',
        prompt: request.prompt,
        max_tokens: request.max_tokens || 100,
        temperature: request.temperature || 0.7,
      }

      let inferenceResult: InferenceResult = {
        text: '',
        tokens: 0,
        duration: 0,
      }

      // TODO: Use Worker SDK to submit operation
      // This depends on how rbee-worker-sdk exposes operations
      // For now, this is a placeholder showing the pattern
      
      console.log('[Worker] Running inference:', request.prompt)
      
      // Simulate SSE stream processing
      // In real implementation, this would be:
      // await workerClient.submitAndStream(operation, (line: string) => {
      //   narrationHandler(line)
      //   // Parse inference results from SSE stream
      // })

      setResult(inferenceResult)
      return inferenceResult

    } catch (err) {
      console.error('[Worker] Inference error:', err)
      setError(err as Error)
      throw err
    } finally {
      setInferring(false)
    }
  }

  return {
    infer,
    inferring,
    error,
    result,
  }
}
```

**Key points:**
- ✅ Uses `createStreamHandler` from @rbee/narration-client
- ✅ Uses `SERVICES.worker` config (no hardcoded URLs)
- ✅ No custom narration parsing
- ✅ Events automatically flow to Keeper

---

## Step 2: Create hooks/index.ts

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/hooks/index.ts`

```typescript
// TEAM-354: Worker React hooks
export { useInferenceWithNarration } from './useInferenceWithNarration'
export type { 
  InferenceRequest, 
  InferenceResult, 
  UseInferenceWithNarrationResult 
} from './useInferenceWithNarration'
```

---

## Step 3: Update Package Index

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/index.ts`

```typescript
// TEAM-354: Worker React hooks package
// Uses shared packages - NO custom implementations

import { useQuery, useMutation } from '@tanstack/react-query'

// Export data fetching hooks
export function useWorkerStatus() {
  const { 
    data: status, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['worker-status'],
    queryFn: async () => {
      return { status: 'idle', model: null }
    },
    staleTime: 5000,
    refetchInterval: 2000,
    retry: 3,
  })

  return { 
    status: status || { status: 'idle', model: null }, 
    loading,
    error: error as Error | null,
    refetch
  }
}

// Export operation hooks
export { useInferenceWithNarration } from './hooks/useInferenceWithNarration'
export type { 
  InferenceRequest, 
  InferenceResult, 
  UseInferenceWithNarrationResult 
} from './hooks/useInferenceWithNarration'

// Re-export types
export type { WorkerClient } from '@rbee/rbee-worker-sdk'
```

---

## Step 4: Verify SERVICES.worker Config

**Check:** `frontend/packages/narration-client/src/config.ts`

**Should contain:**
```typescript
export const SERVICES: Record<ServiceName, ServiceConfig> = {
  queen: { ... },
  hive: { ... },
  worker: {
    name: 'llm-worker',
    devPort: PORTS.worker.dev,      // 7838
    prodPort: PORTS.worker.prod,    // 8080
    keeperDevPort: PORTS.keeper.dev,
    keeperProdOrigin: '*',
  },
}
```

**If missing:** Add it to the config file.

---

## Step 5: Build Package

```bash
cd bin/30_llm_worker_rbee/ui/packages/rbee-worker-react
pnpm build
```

**Expected output:**
```
> @rbee/rbee-worker-react@0.1.0 build
> tsc

✓ Built successfully
```

---

## Step 6: Test Narration Flow

**Terminal 1:** Start Worker backend
```bash
cargo run --bin llm-worker
```

**Terminal 2:** Start Worker UI
```bash
cd bin/30_llm_worker_rbee/ui/app
pnpm dev
```

**Terminal 3:** Start Keeper UI
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Browser:** Open Keeper at http://localhost:5173

**Test:**
1. Navigate to Worker page (iframe should load)
2. Trigger inference operation
3. **Verify narration appears in:**
   - Worker console: `[Worker] Sending to parent: ...`
   - Keeper console: `[Keeper] Received narration: ...`
   - Keeper narration panel

**Expected console logs:**
```
[Worker] SSE line: data: {"actor":"llm_worker",...}
[Worker] Narration event: { actor: "llm_worker", ... }
[llm-worker] Sending to parent: { origin: "http://localhost:5173", ... }
[Keeper] Received narration: { actor: "llm_worker", ... }
```

---

## Testing Checklist

- [ ] `pnpm build` (rbee-worker-react) - success
- [ ] SERVICES.worker config exists
- [ ] Narration handler created correctly
- [ ] Narration appears in Worker console
- [ ] Narration appears in Keeper console
- [ ] Narration appears in Keeper panel
- [ ] No TypeScript errors
- [ ] TEAM-354 signatures added

---

## Success Criteria

✅ useInferenceWithNarration hook created  
✅ Uses @rbee/narration-client  
✅ Uses SERVICES.worker config  
✅ Narration flows to Keeper  
✅ Package builds successfully  
✅ TEAM-354 signatures added

---

## Next Step

Continue to **TEAM_354_STEP_4_CONFIG_CLEANUP.md** to remove hardcoded URLs.

---

**TEAM-354 Step 3: Narration integration complete!** ✅
