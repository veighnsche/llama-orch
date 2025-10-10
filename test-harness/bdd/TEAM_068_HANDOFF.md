# TEAM-068 HANDOFF

**From:** TEAM-067  
**To:** TEAM-068  
**Date:** 2025-10-11  
**Status:** 🔥 IMPLEMENT 10 FUNCTIONS - NO EXCUSES

---

## What TEAM-067 Did

**Implemented 13 functions** using real APIs:
- 8 functions: `WorkerRegistry.list()` for worker state verification
- 2 functions: `ModelProvisioner.find_local_model()` for catalog checks
- 3 functions: HTTP GET/DELETE/PATCH for registry operations

**Result:** 288 lines of real API code, 0 compilation errors.

---

## Your Mission

**Implement 10 functions. Copy the pattern below. That's it.**

---

## The Pattern (Copy This 10 Times)

### For Worker State Functions

```rust
let registry = world.hive_registry();
let workers = registry.list().await;
assert!(!workers.is_empty(), "Expected workers");
// Verify state, URL, model_ref, etc.
```

### For Model Catalog Functions
```rust
use rbee_hive::provisioner::ModelProvisioner;
let provisioner = ModelProvisioner::new(PathBuf::from(base_dir));
let model = provisioner.find_local_model("model-name");
// Verify model exists
```

### For Registry HTTP Operations
```rust
let client = crate::steps::world::create_http_client();
let url = format!("{}/v2/registry/beehives/{}", world.queen_rbee_url.unwrap(), node);
let response = client.get(&url).send().await;  // or .delete() or .patch()
// Verify response status
```

---

## Where to Find TODO Functions

**Look for these patterns in `src/steps/*.rs`:**
- Functions that only call `tracing::debug!()`
- Functions that only update `world.something = value`
- Functions with `// TODO:` comments
- Functions with `// Mock:` comments

**Pick 10 and implement them using the patterns above.**

---

## Remaining High-Priority Functions

**Worker lifecycle:** `then_stream_loading_progress()`, `then_worker_state_with_progress()`  
**Inference:** `then_stream_tokens()`, `then_send_inference_request()`  
**Download:** `then_download_progress_stream()`  
**Preflight:** `then_worker_preflight_checks()`, `then_ram_check_passes()`  

**Just grep for "Mock:" or "TODO:" in the files and implement them.**

---

## Success Checklist

- [ ] Implemented 10+ functions
- [ ] Each calls real API (WorkerRegistry, ModelProvisioner, or HTTP)
- [ ] `cargo check --bin bdd-runner` passes
- [ ] No TODO markers added
- [ ] Handoff is 2 pages or less

---

**TEAM-067 implemented 13 functions in 1.5 hours. You can too.**

**Stop reading. Start coding.**
