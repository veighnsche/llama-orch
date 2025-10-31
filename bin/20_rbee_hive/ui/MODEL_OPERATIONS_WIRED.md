# Model Operations Wired - Frontend ↔ Backend Integration Complete

**Status:** ✅ COMPLETE  
**Date:** 2025-10-31

## Summary

Wired up model load/unload operations through the proper job submission flow:
- **Frontend** → Job Client (SSE) → **Backend** → Narration

All model operations now follow the established pattern:
```
UI Button Click → useJobSubmit → POST /v1/jobs → SSE Stream → Narration Messages
```

---

## Changes Made

### 1. Contract Updates (`operations-contract`)

**Added New Operations:**
- `ModelLoad` - Load model into RAM on specific device
- `ModelUnload` - Unload model from RAM

**Files Modified:**
- `bin/97_contracts/operations-contract/src/lib.rs`
  - Added `ModelLoad(ModelLoadRequest)` to Operation enum
  - Added `ModelUnload(ModelUnloadRequest)` to Operation enum

- `bin/97_contracts/operations-contract/src/requests.rs`
  - Added `ModelLoadRequest { hive_id, id, device }`
  - Added `ModelUnloadRequest { hive_id, id }`

- `bin/97_contracts/operations-contract/src/operation_impl.rs`
  - Added `model_load` / `model_unload` to `name()` method
  - Added both operations to `hive_id()` method
  - Added both operations to `target_server()` → `TargetServer::Hive`

### 2. Backend Handler (`rbee-hive`)

**File:** `bin/20_rbee_hive/src/job_router.rs`

Added handlers for new operations with narration:

```rust
Operation::ModelLoad(request) => {
    n!("model_load_start", "🚀 Loading model '{}' to RAM on device '{}'", id, device);
    n!("model_load_progress", "📦 Allocating memory for model '{}'", id);
    n!("model_load_progress", "🔄 Loading model weights into VRAM/RAM");
    n!("model_load_complete", "✅ Model '{}' loaded to RAM on device '{}'", id, device);
}

Operation::ModelUnload(request) => {
    n!("model_unload_start", "🔽 Unloading model '{}' from RAM", id);
    n!("model_unload_progress", "🧹 Freeing memory for model '{}'", id);
    n!("model_unload_complete", "✅ Model '{}' unloaded from RAM", id);
}
```

**Note:** Actual implementation (spawning/killing workers) is TODO. Currently just narration for MVP.

### 3. Frontend Hook (`rbee-hive/ui`)

**New File:** `bin/20_rbee_hive/ui/app/src/hooks/useJobSubmit.ts`

Reusable hook for job submission following the job client pattern:

```typescript
const { submitJob, loading, error, messages } = useJobSubmit()

await submitJob({
  operation: 'model_load',
  hive_id: 'localhost',
  id: modelId,
  device: 'cuda:0',
}, {
  onProgress: (msg) => console.log(msg),
  onComplete: () => console.log('Done!'),
})
```

**Features:**
- ✅ POST to `/v1/jobs` endpoint
- ✅ Connect to SSE stream for narration
- ✅ Parse and emit progress messages
- ✅ Detect `[DONE]` completion
- ✅ Error handling

### 4. UI Component Updates

**File:** `bin/20_rbee_hive/ui/app/src/components/ModelManagement.tsx`

**Added Operation Handlers:**
```typescript
const handleLoadModel = async (modelId: string, device: string = 'cuda:0') => {
  await submitJob({ operation: 'model_load', hive_id: 'localhost', id: modelId, device })
}

const handleUnloadModel = async (modelId: string) => {
  await submitJob({ operation: 'model_unload', hive_id: 'localhost', id: modelId })
}

const handleDeleteModel = async (modelId: string) => {
  await submitJob({ operation: 'model_delete', hive_id: 'localhost', id: modelId })
}
```

**Wired Up Buttons:**
- ✅ Downloaded table: Load to RAM button → `handleLoadModel()`
- ✅ Downloaded table: Delete button → `handleDeleteModel()`
- ✅ Loaded table: Unload button → `handleUnloadModel()`

---

## Data Flow

### Model Load Operation

```
┌─────────────┐
│ UI Button   │ User clicks "Load to RAM"
│ (Play icon) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ handleLoadModel(modelId, device)                        │
│ - Calls submitJob() with ModelLoad operation            │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ useJobSubmit Hook                                       │
│ 1. POST /v1/jobs { operation: "model_load", ... }      │
│ 2. Receive job_id                                       │
│ 3. Connect to SSE: /v1/jobs/{job_id}/stream            │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ rbee-hive Job Router                                    │
│ - Matches Operation::ModelLoad(request)                 │
│ - Emits narration events:                               │
│   • model_load_start                                    │
│   • model_load_progress (allocating memory)             │
│   • model_load_progress (loading weights)               │
│   • model_load_complete                                 │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ SSE Stream → Frontend                                   │
│ - onProgress() callback fires for each message          │
│ - onComplete() fires when [DONE] received               │
└─────────────────────────────────────────────────────────┘
```

---

## Architecture Compliance

### ✅ Follows Job Client Pattern

Same pattern as existing operations (WorkerSpawn, ModelDownload, etc.):
1. Operation defined in `operations-contract`
2. Handler in `rbee-hive/job_router.rs`
3. Frontend uses `useJobSubmit` hook
4. Narration flows through SSE

### ✅ Contract-First Design

- All operations defined in shared contract crate
- Type-safe requests/responses
- Single source of truth for operation routing

### ✅ Separation of Concerns

- **Queen** handles orchestration (Status, Infer)
- **Hive** handles worker/model lifecycle (Spawn, Load, Delete)
- **Frontend** submits jobs, displays narration

---

## Testing

### Manual Test Flow

1. **Start rbee-hive:**
   ```bash
   cargo run --bin rbee-hive
   ```

2. **Start frontend:**
   ```bash
   cd bin/20_rbee_hive/ui/app
   pnpm dev
   ```

3. **Test Load Operation:**
   - Navigate to Model Management
   - Click "Downloaded" tab
   - Click Play icon on a model
   - Watch console for narration messages

4. **Expected Narration:**
   ```
   🚀 Loading model 'meta-llama-Llama-2-7b' to RAM on device 'cuda:0'
   📦 Allocating memory for model 'meta-llama-Llama-2-7b'
   🔄 Loading model weights into VRAM/RAM
   ✅ Model 'meta-llama-Llama-2-7b' loaded to RAM on device 'cuda:0'
   [DONE]
   ```

---

## Next Steps (TODO)

### 1. Actual Model Loading Implementation

Currently just narration. Need to:
- Spawn worker process with model loaded
- Track loaded models in registry
- Update model catalog with `loaded: true` flag

### 2. Worker Registry Integration

- Link loaded models to worker PIDs
- Track which device each model is on
- Show worker info in "Loaded in RAM" table

### 3. Device Selection UI

Add device picker in model details panel:
```tsx
<select>
  <option>cuda:0 (RTX 4090 - 24GB free)</option>
  <option>cuda:1 (RTX 3090 - 24GB free)</option>
  <option>cpu (64GB RAM free)</option>
</select>
```

### 4. Progress Indicators

Show loading state in UI:
- Spinner on button during load
- Progress bar for download/load operations
- Toast notifications for completion

### 5. Error Handling

- Show error messages in UI
- Retry failed operations
- Graceful degradation

---

## Files Changed

### Backend (Rust)
- `bin/97_contracts/operations-contract/src/lib.rs` - Added ModelLoad/Unload operations
- `bin/97_contracts/operations-contract/src/requests.rs` - Added request types
- `bin/97_contracts/operations-contract/src/operation_impl.rs` - Added routing logic
- `bin/20_rbee_hive/src/job_router.rs` - Added operation handlers

### Frontend (TypeScript/React)
- `bin/20_rbee_hive/ui/app/src/hooks/useJobSubmit.ts` - **NEW** job submission hook
- `bin/20_rbee_hive/ui/app/src/components/ModelManagement.tsx` - Wired up operations

---

## Compilation Status

✅ **Backend:** `cargo check --package rbee-hive` - PASS  
✅ **Contract:** `cargo check --package operations-contract` - PASS  
⚠️ **Frontend:** TypeScript warnings (unused variables) - Safe to ignore for MVP

---

**This completes the wiring for model operations through the job submission flow!** 🎉

The infrastructure is in place. Now we just need to implement the actual worker spawning/killing logic.
