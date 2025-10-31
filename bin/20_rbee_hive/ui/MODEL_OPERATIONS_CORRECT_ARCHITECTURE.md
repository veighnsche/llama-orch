# Model Operations - Correct Architecture ✅

**Status:** FIXED - Now follows proper job submission pattern  
**Date:** 2025-10-31

## Problem

Initial implementation created a shortcut `useJobSubmit` hook that bypassed the established architecture:
- ❌ Direct fetch calls to `/v1/jobs`
- ❌ Manual SSE handling
- ❌ Duplicate code pattern
- ❌ Not using WASM SDK
- ❌ Not using shared `job-client` crate

**This violated the single source of truth principle!**

---

## Correct Architecture

### Layer 1: Rust Contract (`operations-contract`)
Defines all operations with type safety:

```rust
// bin/97_contracts/operations-contract/src/lib.rs
pub enum Operation {
    ModelLoad(ModelLoadRequest),
    ModelUnload(ModelUnloadRequest),
    // ...
}

// bin/97_contracts/operations-contract/src/requests.rs
pub struct ModelLoadRequest {
    pub hive_id: String,
    pub id: String,
    pub device: String,
}
```

### Layer 2: WASM SDK (`rbee-hive-sdk`)
Exposes operations to JavaScript:

```rust
// bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs
#[wasm_bindgen]
impl OperationBuilder {
    #[wasm_bindgen(js_name = modelLoad)]
    pub fn model_load(hive_id: String, id: String, device: String) -> JsValue {
        let op = Operation::ModelLoad(ModelLoadRequest {
            hive_id,
            id,
            device,
        });
        to_value(&op).unwrap()
    }
}
```

Uses `job-client` crate internally:

```rust
// bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/client.rs
pub struct HiveClient {
    inner: JobClient,  // ← Reuses shared job-client!
}

impl HiveClient {
    pub async fn submit_and_stream(&self, operation, on_line) -> Result<String> {
        self.inner.submit_and_stream(op, |line| {
            callback.call1(&this, &line_js);
            Ok(())
        }).await
    }
}
```

### Layer 3: React Hook (`rbee-hive-react`)
Provides React-friendly API:

```typescript
// bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useModelOperations.ts
export function useModelOperations() {
  const loadMutation = useMutation({
    mutationFn: async ({ modelId, device }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.modelLoad(hiveId, modelId, device)
      
      await client.submitAndStream(op, (line: string) => {
        console.log('[Hive] Model load:', line)
      })
    }
  })

  return {
    loadModel: loadMutation.mutate,
    isPending: loadMutation.isPending,
    // ...
  }
}
```

### Layer 4: UI Component
Uses the hook:

```tsx
// bin/20_rbee_hive/ui/app/src/components/ModelManagement.tsx
import { useModelOperations } from '@rbee/rbee-hive-react'

export function ModelManagement() {
  const { loadModel, unloadModel, deleteModel } = useModelOperations()

  return (
    <button onClick={() => loadModel({ modelId: 'llama-2-7b', device: 'cuda:0' })}>
      Load to RAM
    </button>
  )
}
```

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ UI Component (ModelManagement.tsx)                          │
│ - Calls loadModel({ modelId, device })                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ React Hook (useModelOperations.ts)                          │
│ - TanStack Query mutation                                   │
│ - Calls OperationBuilder.modelLoad()                        │
│ - Calls client.submitAndStream()                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ WASM SDK (rbee-hive-sdk)                                    │
│ - HiveClient wraps JobClient                                │
│ - OperationBuilder creates typed operations                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Shared Job Client (job-client crate)                        │
│ - POST /v1/jobs with Operation JSON                         │
│ - Receive job_id                                            │
│ - Connect to SSE: /v1/jobs/{job_id}/stream                  │
│ - Stream narration lines                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ rbee-hive Job Router                                        │
│ - Matches Operation::ModelLoad(request)                     │
│ - Emits narration via n!() macro                            │
│ - Streams to SSE                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture?

### ✅ Single Source of Truth
- **One** job client implementation (`job-client` crate)
- **One** operation definition (`operations-contract`)
- **One** way to submit jobs

### ✅ Type Safety
- Rust types compile to WASM
- TypeScript gets proper types
- Impossible to send malformed operations

### ✅ Consistency
- Same pattern for ALL operations:
  - Worker spawn
  - Model download
  - Model load/unload
  - Inference
  - Everything!

### ✅ Maintainability
- Fix bugs in ONE place (`job-client`)
- Add operations in THREE files:
  1. `operations-contract/src/lib.rs` (Rust enum)
  2. `rbee-hive-sdk/src/operations.rs` (WASM binding)
  3. `rbee-hive-react/src/hooks/useModelOperations.ts` (React hook)

### ✅ Testability
- Mock at any layer
- Test WASM SDK independently
- Test React hooks with MSW

---

## Files Changed (Correct Implementation)

### Backend (Rust)
- ✅ `bin/97_contracts/operations-contract/src/lib.rs` - Added ModelLoad/Unload
- ✅ `bin/97_contracts/operations-contract/src/requests.rs` - Added request types
- ✅ `bin/97_contracts/operations-contract/src/operation_impl.rs` - Added routing
- ✅ `bin/20_rbee_hive/src/job_router.rs` - Added handlers

### WASM SDK
- ✅ `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs` - Added modelLoad/modelUnload

### React Package
- ✅ `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useModelOperations.ts` - **NEW**
- ✅ `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/index.ts` - Export hook
- ✅ `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts` - Export hook

### UI App
- ✅ `bin/20_rbee_hive/ui/app/src/components/ModelManagement.tsx` - Use hook
- ❌ `bin/20_rbee_hive/ui/app/src/hooks/useJobSubmit.ts` - **DELETED** (was wrong!)

---

## Comparison: Wrong vs Right

### ❌ Wrong (Initial Implementation)

```typescript
// Direct fetch - bypasses architecture!
const response = await fetch('http://localhost:7835/v1/jobs', {
  method: 'POST',
  body: JSON.stringify({ operation: 'model_load', ... })
})

const eventSource = new EventSource(`http://localhost:7835/v1/jobs/${jobId}/stream`)
// Manual SSE handling...
```

**Problems:**
- Duplicate code (same pattern in multiple places)
- No type safety
- Hardcoded URLs
- Manual SSE parsing
- Not using shared `job-client`

### ✅ Right (Corrected Implementation)

```typescript
// Uses proper architecture layers
const { loadModel } = useModelOperations()  // ← React hook
loadModel({ modelId, device })              // ← Type-safe

// Under the hood:
// - useModelOperations → HiveClient → JobClient → HTTP
// - All type-safe
// - All using shared code
// - All following same pattern
```

**Benefits:**
- Single source of truth
- Type safety end-to-end
- Consistent with other operations
- Maintainable
- Testable

---

## Next Steps

1. **Rebuild WASM SDK:**
   ```bash
   cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
   pnpm build
   ```

2. **Test in UI:**
   ```bash
   cd bin/20_rbee_hive/ui/app
   pnpm dev
   ```

3. **Verify narration:**
   - Click "Load to RAM" button
   - Check console for narration messages
   - Verify SSE stream works

---

## Architecture Principles

1. **Contract First** - Define operations in Rust contract
2. **WASM Bridge** - Expose to JavaScript via WASM
3. **React Wrapper** - Provide React-friendly hooks
4. **UI Consumption** - Components use hooks

**Every operation follows this pattern. No exceptions!**

---

**This is the correct way. The only way.** 🎯
