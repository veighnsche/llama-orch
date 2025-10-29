# TEAM-286: Operations Implementation Checklist

**Date:** Oct 24, 2025  
**Total Operations:** 17  
**Status:** 17/17 Builders Complete, 1/17 Convenience Methods Complete

---

## System Operations (1)

- [x] **Status** - Get system status (all hives and workers)
  - Builder: ✅ `OperationBuilder::status()`
  - Convenience: ⏳ `client.status(callback)`

---

## Hive Operations (4)

- [x] **HiveList** - List all configured hives
  - Builder: ✅ `OperationBuilder::hive_list()`
  - Convenience: ⏳ `client.list_hives(callback)`

- [x] **HiveGet** - Get details for a specific hive
  - Builder: ✅ `OperationBuilder::hive_get(alias)`
  - Convenience: ⏳ `client.get_hive(alias, callback)`

- [x] **HiveStatus** - Check hive health endpoint
  - Builder: ✅ `OperationBuilder::hive_status(alias)`
  - Convenience: ⏳ `client.get_hive_status(alias, callback)`

- [x] **HiveRefreshCapabilities** - Refresh device capabilities
  - Builder: ✅ `OperationBuilder::hive_refresh_capabilities(alias)`
  - Convenience: ⏳ `client.refresh_hive_capabilities(alias, callback)`

---

## Worker Process Operations (4)

These manage worker PROCESSES on the hive (local ps, not registry).

- [x] **WorkerSpawn** - Spawn a worker process on hive
  - Builder: ✅ `OperationBuilder::worker_spawn(params)`
  - Convenience: ⏳ `client.spawn_worker(params, callback)`
  - Params: `{ hive_id, model, worker, device }`

- [x] **WorkerProcessList** - List worker processes on hive
  - Builder: ✅ `OperationBuilder::worker_process_list(params)`
  - Convenience: ⏳ `client.list_worker_processes(hive_id, callback)`
  - Params: `{ hive_id }`

- [x] **WorkerProcessGet** - Get worker process details
  - Builder: ✅ `OperationBuilder::worker_process_get(params)`
  - Convenience: ⏳ `client.get_worker_process(hive_id, pid, callback)`
  - Params: `{ hive_id, pid }`

- [x] **WorkerProcessDelete** - Kill a worker process
  - Builder: ✅ `OperationBuilder::worker_process_delete(params)`
  - Convenience: ⏳ `client.delete_worker_process(hive_id, pid, callback)`
  - Params: `{ hive_id, pid }`

---

## Active Worker Operations (3)

These query queen's registry of workers sending heartbeats.

- [x] **ActiveWorkerList** - List active workers from registry
  - Builder: ✅ `OperationBuilder::active_worker_list()`
  - Convenience: ⏳ `client.list_active_workers(callback)`

- [x] **ActiveWorkerGet** - Get active worker details from registry
  - Builder: ✅ `OperationBuilder::active_worker_get(worker_id)`
  - Convenience: ⏳ `client.get_active_worker(worker_id, callback)`

- [x] **ActiveWorkerRetire** - Retire an active worker
  - Builder: ✅ `OperationBuilder::active_worker_retire(worker_id)`
  - Convenience: ⏳ `client.retire_active_worker(worker_id, callback)`

---

## Model Operations (4)

- [x] **ModelDownload** - Download a model to hive
  - Builder: ✅ `OperationBuilder::model_download(params)`
  - Convenience: ⏳ `client.download_model(params, callback)`
  - Params: `{ hive_id, model }`

- [x] **ModelList** - List models on hive
  - Builder: ✅ `OperationBuilder::model_list(params)`
  - Convenience: ⏳ `client.list_models(hive_id, callback)`
  - Params: `{ hive_id }`

- [x] **ModelGet** - Get model details
  - Builder: ✅ `OperationBuilder::model_get(params)`
  - Convenience: ⏳ `client.get_model(hive_id, id, callback)`
  - Params: `{ hive_id, id }`

- [x] **ModelDelete** - Delete a model
  - Builder: ✅ `OperationBuilder::model_delete(params)`
  - Convenience: ⏳ `client.delete_model(hive_id, id, callback)`
  - Params: `{ hive_id, id }`

---

## Inference Operations (1)

- [x] **Infer** - Run inference (streaming or non-streaming)
  - Builder: ✅ `OperationBuilder::infer(params)`
  - Convenience: ✅ `client.infer(params, callback)`
  - Params: `{ hive_id, model, prompt, max_tokens, temperature, top_p?, top_k?, device?, worker_id?, stream? }`

---

## Implementation Order

### Phase 3A: Operation Builders ✅ COMPLETE
1. ✅ Create `src/operations.rs`
2. ✅ Implement all 17 builders
3. ✅ Export from `src/lib.rs`

### Phase 3B: Convenience Methods ⏳ IN PROGRESS
1. ⏳ Add all convenience methods to `RbeeClient` (1/17 done)
2. ⏳ Handle response parsing
3. ⏳ Add proper documentation

### Phase 3C: Testing 📋 TODO
1. Update `test.html` with all operations
2. Test each operation
3. Verify responses

---

## Progress Tracking

**Builders:** 17/17 (100%) ✅  
**Convenience Methods:** 1/17 (6%) ⏳  
**Overall:** 18/34 (53%)

---

## Summary

**What Works Now:**
- ✅ All 17 operation builders implemented
- ✅ OperationBuilder exported to JavaScript
- ✅ `client.infer()` convenience method
- ✅ Compiles successfully

**What's Left:**
- ⏳ 16 more convenience methods
- 📋 Update test.html
- 📋 Test all operations

**Note:** Builders alone are sufficient! Users can call:
```javascript
await client.submitAndStream(
  OperationBuilder.status(),
  (line) => console.log(line)
);
```

Convenience methods just make it easier:
```javascript
await client.status((line) => console.log(line));
```
