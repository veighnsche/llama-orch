# Responsibility Refactor Summary — engine-provisioner & pool-managerd

**Date:** 2025-09-30  
**Branch:** deleted-engine-provisioner-responsibilities  
**Status:** ✅ Phase 1 Complete (engine-provisioner cleaned up)

---

## Problem Identified

**engine-provisioner was doing TOO MUCH** due to E2E tests demanding:
1. ❌ Spawn process (line 246)
2. ❌ Wait for health check (line 252)
3. ❌ Write handoff file (line 269)
4. ❌ Write PID file (line 247)
5. ❌ Keep process running (tests polled /health and made inference requests!)

**Root cause:** Tests validated that `ensure()` left a **running, healthy server**, forcing implementation to spawn+supervise instead of just prepare.

---

## Phase 1: engine-provisioner Cleanup ✅ DONE

### Changes Made:

1. **DELETED 4 E2E tests** (they tested the wrong thing):
   - ❌ `llamacpp_fixture_cpu_e2e.rs` — tested running server + health
   - ❌ `llamacpp_source_cpu_real_e2e.rs` — tested inference requests!
   - ❌ `llamacpp_smoke.rs` — tested running server
   - ❌ `restart_on_crash.rs` — tested supervision (belongs in pool-managerd)

2. **ADDED PreparedEngine struct**:
   ```rust
   pub struct PreparedEngine {
       pub binary_path: PathBuf,  // /path/to/llama-server
       pub flags: Vec<String>,     // ["--model", "...", "--port", "8080"]
       pub port: u16,
       pub host: String,
       pub model_path: PathBuf,
       pub engine_version: String,
       pub pool_id: String,
       pub replica_id: String,
       pub device_mask: Option<String>,
   }
   ```

3. **CHANGED EngineProvisioner trait**:
   ```rust
   // OLD (WRONG):
   fn ensure(&self, pool: &PoolConfig) -> Result<()>;
   
   // NEW (CORRECT):
   fn ensure(&self, pool: &PoolConfig) -> Result<PreparedEngine>;
   ```

4. **REMOVED from llamacpp provider**:
   - ❌ `cmdline.spawn()` — spawning process
   - ❌ `wait_for_health()` — health monitoring
   - ❌ `write_handoff_file()` — handoff writing
   - ❌ `std::fs::write(&pid_file, ...)` — PID file writing
   - ❌ `child.kill()` — process cleanup

5. **REMOVED stop_pool() function** (belongs in pool-managerd)

### Result:

✅ engine-provisioner now **ONLY prepares** (download, build, stage model)  
✅ Returns `PreparedEngine` metadata  
✅ Does **NOT spawn** or supervise processes  
✅ Compiles successfully  

---

## Phase 2: pool-managerd Implementation (NEXT)

### What Needs to Be Done:

1. **Implement preload.rs**:
   ```rust
   pub fn execute(prepared: PreparedEngine, registry: &mut Registry) -> Result<()> {
       // 1. Spawn process using prepared.binary_path + prepared.flags
       // 2. Write PID file
       // 3. Wait for health check on prepared.port
       // 4. Write handoff file when healthy
       // 5. Update registry.set_health(ready=true)
       // 6. Return Ok or Err
   }
   ```

2. **Implement supervision (backoff.rs)**:
   - Monitor process health
   - Restart with exponential backoff on crash
   - Circuit breaker to prevent restart storms

3. **Implement drain.rs**:
   - Drain: stop accepting new leases, wait for in-flight, kill process
   - Reload: drain → call engine-provisioner.ensure() → spawn new → health check

4. **Wire to orchestratord**:
   - orchestratord calls engine-provisioner.ensure() → gets PreparedEngine
   - orchestratord calls pool-managerd.preload(prepared) → spawns and supervises
   - orchestratord reads handoff files and binds adapters

---

## Correct Separation of Concerns

| Responsibility | Owner | Status |
|----------------|-------|--------|
| Download/build engine | engine-provisioner | ✅ DONE |
| Stage model | engine-provisioner (via model-provisioner) | ✅ DONE |
| Return PreparedEngine | engine-provisioner | ✅ DONE |
| Spawn process | pool-managerd | ⏳ TODO |
| Monitor health | pool-managerd | ⏳ TODO |
| Write handoff | pool-managerd | ⏳ TODO |
| Write PID file | pool-managerd | ⏳ TODO |
| Supervise with backoff | pool-managerd | ⏳ TODO |
| Update registry | pool-managerd | ⏳ TODO |
| Read handoff files | orchestratord | ✅ EXISTS |
| Bind adapters | orchestratord | ✅ EXISTS |
| Placement decisions | orchestratord | ✅ EXISTS |

---

## Files Changed

### engine-provisioner:
- ✅ `src/lib.rs` — added PreparedEngine, changed trait, removed stop_pool()
- ✅ `src/providers/llamacpp/mod.rs` — removed spawn/health/handoff, returns PreparedEngine
- ✅ `tests/` — deleted 4 E2E tests
- ✅ `TEST_ANALYSIS.md` — created (analysis of test scope creep)

### pool-managerd:
- ⏳ `src/preload.rs` — needs implementation (spawn/health/handoff)
- ⏳ `src/backoff.rs` — needs implementation (supervision)
- ⏳ `src/drain.rs` — needs implementation (drain/reload)
- ✅ `RESPONSIBILITY_AUDIT.md` — created (overlap analysis)
- ✅ `STUB_ANALYSIS.md` — created (stub documentation)
- ✅ `REFACTOR_SUMMARY.md` — this file

---

## Next Steps

1. **Implement pool-managerd.preload.rs** (spawn/health/handoff)
2. **Implement pool-managerd.backoff.rs** (supervision)
3. **Wire orchestratord** to call engine-provisioner → pool-managerd
4. **Write correct tests**:
   - engine-provisioner: test PreparedEngine is correct
   - pool-managerd: test spawn/health/handoff/supervision
   - test-harness/e2e: test full flow (provision → spawn → inference)
5. **Delete pool-managerd/bdd** (broken, unused)
6. **Delete src/leases.rs** (redundant)
7. **Update specs** to clarify responsibilities

---

## Verification

```bash
# engine-provisioner compiles
cargo check -p provisioners-engine-provisioner
# ✅ Success (1 warning about unused try_fetch_engine_version)

# Deleted tests are gone
ls libs/provisioners/engine-provisioner/tests/
# (empty directory)

# PreparedEngine struct exists
grep -A 10 "pub struct PreparedEngine" libs/provisioners/engine-provisioner/src/lib.rs
# ✅ Found

# ensure() returns PreparedEngine
grep "fn ensure" libs/provisioners/engine-provisioner/src/lib.rs
# ✅ fn ensure(&self, pool: &PoolConfig) -> Result<PreparedEngine>;
```

---

## Conclusion

✅ **Phase 1 Complete:** engine-provisioner scope creep removed  
⏳ **Phase 2 Next:** Implement pool-managerd spawn/supervise logic  

The architecture is now correct:
- **engine-provisioner** = prepare (build binary)
- **pool-managerd** = manage (spawn, supervise, health)
- **orchestratord** = orchestrate (placement, admission)
