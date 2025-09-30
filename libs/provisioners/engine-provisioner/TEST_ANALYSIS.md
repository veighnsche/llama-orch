# engine-provisioner Test Analysis — Scope Creep Investigation

**Date:** 2025-09-30  
**Finding:** Tests are asking engine-provisioner to do TOO MUCH, causing scope creep

---

## Executive Summary

**YOU WERE RIGHT!** The E2E tests are demanding that engine-provisioner:
1. ✅ Build the engine (CORRECT)
2. ❌ **SPAWN the process** (WRONG — should be pool-managerd)
3. ❌ **WAIT for health check** (WRONG — should be pool-managerd)
4. ❌ **WRITE handoff file** (WRONG — should be pool-managerd)
5. ❌ **WRITE PID file** (WRONG — should be pool-managerd)
6. ❌ **KEEP process running** (WRONG — should be pool-managerd)

**Root cause:** Tests validate that `ensure()` leaves a **running, healthy server**. This forced the implementation to spawn+supervise instead of just prepare.

---

## Test-by-Test Analysis

### 1. `llamacpp_fixture_cpu_e2e.rs` — ASKS TOO MUCH

**What it tests (lines 155-190):**
```rust
// Line 156: Calls ensure()
prov.ensure(&pool)?;

// Line 160-172: Expects handoff file to exist
let handoff_path = PathBuf::from(".runtime").join("engines").join("llamacpp.json");
let handoff = fs::read_to_string(&handoff_path)?;
let v: serde_json::Value = serde_json::from_str(&handoff)?;
assert_eq!(v["engine"], "llamacpp");
let url = v["url"].as_str().unwrap_or("");
let port: u16 = url.strip_prefix("http://127.0.0.1:").unwrap().parse().unwrap();

// Line 184-190: Expects server to be RUNNING and HEALTHY
let deadline = std::time::Instant::now() + Duration::from_secs(10);
let mut ok = false;
while std::time::Instant::now() < deadline {
    if http_health_ok("127.0.0.1", port) { ok = true; break; }
    std::thread::sleep(Duration::from_millis(200));
}
assert!(ok, "fixture server did not respond healthy");

// Line 193-198: Cleanup by killing process via PID file
let pid_path = provisioners_engine_provisioner::util::default_run_dir().join("p-fixture.pid");
if let Ok(s) = fs::read_to_string(&pid_path) {
    let pid = s.trim();
    let _ = Command::new("/bin/kill").arg(pid).status();
}
```

**Problems:**
- ❌ Test expects `ensure()` to **spawn and leave running** a healthy server
- ❌ Test expects handoff file to exist (should be pool-managerd's job)
- ❌ Test expects PID file to exist (should be pool-managerd's job)
- ❌ Test manually kills process (should be pool-managerd's job)

**What it SHOULD test:**
- ✅ `ensure()` returns PreparedEngine with binary_path, flags, port
- ✅ Binary exists and is executable
- ✅ Model is staged
- ❌ NOT: process is running
- ❌ NOT: handoff file exists
- ❌ NOT: health check passes

---

### 2. `llamacpp_source_cpu_real_e2e.rs` — ASKS TOO MUCH

**What it tests (lines 76-96):**
```rust
// Line 77: Calls ensure()
prov.ensure(&pool)?;

// Line 80-86: Expects handoff file with running server URL
let handoff_path = PathBuf::from(".runtime").join("engines").join("llamacpp.json");
let handoff = fs::read_to_string(&handoff_path)?;
let v: serde_json::Value = serde_json::from_str(&handoff)?;
let url = v["url"].as_str().unwrap();
let port: u16 = url.strip_prefix("http://127.0.0.1:").unwrap().parse().unwrap();

// Line 88-96: Expects server to be RUNNING and HEALTHY
let deadline = std::time::Instant::now() + Duration::from_secs(60);
let mut ok = false;
while std::time::Instant::now() < deadline {
    if provisioners_engine_provisioner::util::http_ok("127.0.0.1", port, "/health").unwrap_or(false) {
        ok = true; break;
    }
    std::thread::sleep(Duration::from_millis(500));
}
assert!(ok, "llama-server did not become healthy");

// Line 98-129: Makes INFERENCE REQUEST to running server!
let prompt = format!("Write a three-line haiku...");
let resp = http_post("127.0.0.1", port, "/v1/completions", &body_v1).ok();

// Line 135-140: Cleanup by killing process
let pid_path = provisioners_engine_provisioner::util::default_run_dir().join("p-real.pid");
if let Ok(s) = fs::read_to_string(&pid_path) {
    let pid = s.trim();
    let _ = std::process::Command::new("/bin/kill").arg(pid).status();
}
```

**Problems:**
- ❌ Test expects `ensure()` to spawn and leave running a healthy server
- ❌ Test makes **INFERENCE REQUESTS** to the server (way beyond provisioning!)
- ❌ Test expects handoff file to exist
- ❌ Test expects PID file to exist
- ❌ Test manually kills process

**What it SHOULD test:**
- ✅ `ensure()` returns PreparedEngine
- ✅ Real llama.cpp binary is built
- ✅ Model is staged
- ❌ NOT: process is running
- ❌ NOT: inference works

---

### 3. `llamacpp_smoke.rs` — ASKS TOO MUCH

**What it tests (lines 44-59):**
```rust
// Line 45: Calls ensure()
prov.ensure(pool)?;

// Line 50-59: Expects server to be RUNNING and HEALTHY
let deadline = Instant::now() + Duration::from_secs(120);
let mut healthy = false;
while Instant::now() < deadline {
    if http_health_probe("127.0.0.1", port).unwrap_or(false) {
        healthy = true;
        break;
    }
    thread::sleep(Duration::from_secs(2));
}
assert!(healthy, "llama-server did not become healthy on port {}", port);

// Line 62-67: Cleanup by killing process
let pid_path = util::default_run_dir().join(format!("{}.pid", pool.id));
if let Ok(s) = std::fs::read_to_string(&pid_path) {
    let pid = s.trim();
    let _ = std::process::Command::new("kill").arg(pid).status();
}
```

**Problems:**
- ❌ Test expects `ensure()` to spawn and leave running a healthy server
- ❌ Test expects PID file to exist
- ❌ Test manually kills process

---

### 4. `restart_on_crash.rs` — SUPERVISION TEST (WRONG CRATE!)

**What it tests:**
```rust
// This test is about SUPERVISION and RESTART logic
// This belongs in pool-managerd tests, NOT engine-provisioner tests!
```

**Problems:**
- ❌ This is a **supervision test**, not a provisioning test
- ❌ Should be in `pool-managerd/tests/` or `test-harness/`
- ❌ engine-provisioner should NOT have restart/crash logic

---

## Root Cause Analysis

### The Contract Violation

**Spec says (`.specs/50-engine-provisioner.md` line 9):**
> "returns a running engine or a prepared artifact for `pool-managerd` to supervise"

**This is AMBIGUOUS!** It could mean:
- **Interpretation A:** Returns a running process (what tests demand)
- **Interpretation B:** Returns metadata for pool-managerd to start (what it should be)

### How Scope Creep Happened

1. **Tests were written first** expecting `ensure()` to leave a running server
2. **Implementation followed tests** to make them pass
3. **Now engine-provisioner does:**
   - Spawn process (line 246)
   - Wait for health (line 252)
   - Write handoff (line 269)
   - Write PID file (line 247)
   - Keep process running (detached child)

4. **This made pool-managerd redundant** because engine-provisioner already does everything!

---

## Correct Separation of Concerns

### engine-provisioner SHOULD do:
- ✅ Preflight checks (tools, CUDA)
- ✅ Clone/build engine binary
- ✅ Stage model (via model-provisioner)
- ✅ Return `PreparedEngine { binary_path, flags, port, device_mask, ... }`

### pool-managerd SHOULD do:
- ✅ Receive PreparedEngine from engine-provisioner
- ✅ Spawn the process using PreparedEngine.binary_path + flags
- ✅ Monitor health endpoint
- ✅ Write handoff file when healthy
- ✅ Write PID file for supervision
- ✅ Restart with backoff on crash
- ✅ Update registry (live/ready status)

### orchestratord SHOULD do:
- ✅ Read handoff files
- ✅ Bind adapters
- ✅ Query registry for placement decisions

---

## Recommended Actions

### 1. Delete or Rewrite Tests

**DELETE these tests (they test the wrong thing):**
- ❌ `llamacpp_fixture_cpu_e2e.rs` — tests running server, not provisioning
- ❌ `llamacpp_source_cpu_real_e2e.rs` — tests inference, not provisioning
- ❌ `llamacpp_smoke.rs` — tests running server, not provisioning
- ❌ `restart_on_crash.rs` — tests supervision, belongs in pool-managerd

**OR REWRITE to test only provisioning:**
- ✅ Test that `ensure()` returns PreparedEngine
- ✅ Test that binary exists and is executable
- ✅ Test that model is staged
- ❌ Do NOT test that process is running
- ❌ Do NOT test health checks
- ❌ Do NOT test inference

### 2. Change engine-provisioner API

**Current (WRONG):**
```rust
trait EngineProvisioner {
    fn ensure(&self, pool: &PoolConfig) -> Result<()>;
    // ^ Returns nothing, spawns process as side effect
}
```

**Correct:**
```rust
trait EngineProvisioner {
    fn ensure(&self, pool: &PoolConfig) -> Result<PreparedEngine>;
    // ^ Returns metadata, does NOT spawn
}

struct PreparedEngine {
    binary_path: PathBuf,
    flags: Vec<String>,
    port: u16,
    device_mask: String,
    model_path: PathBuf,
    engine_version: String,
}
```

### 3. Move spawn/health/handoff to pool-managerd

**engine-provisioner removes (lines 246-284):**
- ❌ `cmdline.spawn()` — move to pool-managerd
- ❌ `wait_for_health()` — move to pool-managerd
- ❌ `write_handoff_file()` — move to pool-managerd
- ❌ `std::fs::write(&pid_file, ...)` — move to pool-managerd

**pool-managerd implements:**
- ✅ `preload::execute(prepared: PreparedEngine, registry: &mut Registry)`
- ✅ Spawn process using prepared.binary_path + prepared.flags
- ✅ Wait for health on prepared.port
- ✅ Write handoff when healthy
- ✅ Update registry.set_health(ready=true)
- ✅ Supervise with backoff.rs on crash

### 4. Update spec to remove ambiguity

**`.specs/50-engine-provisioner.md` line 9 should say:**
> "Prepares engine binaries or images for pools based on configuration. Plans and executes steps to fetch/build (or pull) engines, ensures required tools, and **returns PreparedEngine metadata** for `pool-managerd` to spawn and supervise."

**NOT:**
> "returns a running engine or a prepared artifact"

---

## Migration Plan

### Phase 1: Delete Bad Tests (NOW)
- Delete `llamacpp_fixture_cpu_e2e.rs`
- Delete `llamacpp_source_cpu_real_e2e.rs`
- Delete `llamacpp_smoke.rs`
- Delete `restart_on_crash.rs`

### Phase 2: Change API (NOW)
- Add `PreparedEngine` struct
- Change `ensure()` to return `Result<PreparedEngine>`
- Remove spawn/health/handoff from `ensure()` implementation

### Phase 3: Implement pool-managerd (NEXT)
- Implement `preload::execute(PreparedEngine)`
- Spawn process
- Health monitoring
- Handoff writing
- Registry updates

### Phase 4: Write Correct Tests (NEXT)
- engine-provisioner: test PreparedEngine is returned correctly
- pool-managerd: test spawn/health/handoff/supervision
- test-harness/e2e: test full flow (provision → spawn → inference)

---

## Conclusion

**The tests forced engine-provisioner to do pool-managerd's job.**

By deleting these tests and changing the API to return `PreparedEngine` instead of spawning, we restore correct separation of concerns:
- engine-provisioner = **prepare** (build binary)
- pool-managerd = **manage** (spawn, supervise, health)
- orchestratord = **orchestrate** (placement, admission)
