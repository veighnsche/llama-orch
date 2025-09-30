# pool-managerd Daemon Decision

**Date:** 2025-09-30  
**Status:** Decision Needed  
**Question:** Make pool-managerd a standalone daemon NOW or wait for CLOUD_PROFILE?

---

## Current State

### Home Profile (v0.1.0)

- orchestratord **embeds** pool-managerd as library (bin/orchestratord/src/state.rs:6)
- Single binary: `orchestratord`
- Single workstation deployment
- On-demand engine startup

### Cloud Profile (future)

- Multi-tenant or dedicated deployments
- Kubernetes with DaemonSet pattern
- Separate pool-managerd instances per namespace/VPC

---

## Option A: Make Daemon NOW ✅ (RECOMMENDED)

### Pros

1. **🎯 Cleaner Architecture from Day 1**
   - Clear separation: orchestratord = control plane, pool-managerd = data plane
   - Each binary has single responsibility
   - Easier to reason about

2. **🔧 Easier Development & Testing**
   - Can test pool-managerd in isolation
   - Can restart pool-managerd without restarting orchestratord
   - Easier to debug (separate logs, separate processes)

3. **🚀 Cloud-Ready from Start**
   - No migration needed when scaling up
   - Same architecture for home and cloud (just different deployment)
   - Validate the architecture early

4. **💾 Better Resource Management**
   - Pool-managerd crash doesn't affect orchestratord
   - Can restart/upgrade pool-managerd independently
   - Clearer memory/CPU attribution

5. **🔐 Security Boundary**
   - Pool-managerd runs engine processes (potentially untrusted)
   - orchestratord handles auth/admission (trusted)
   - Process isolation = security boundary

6. **📦 Simpler Deployment Story**
   - Home: both binaries on same machine (systemd units)
   - Cloud: orchestratord Deployment + pool-managerd DaemonSet
   - Same binaries, different deployment configs

7. **🧪 Already Implemented!**
   - pool-managerd has `src/main.rs` (stub)
   - preload.rs is ready (spawn/health/handoff)
   - Just needs HTTP API for orchestratord to call

8. **📁 Move to bin/ Makes Sense**
   - It's a daemon, not a library
   - Aligns with orchestratord (also in bin/)
   - Clearer workspace structure

### Cons

1. **⚠️ More Complexity for Home Profile**
   - Two processes instead of one
   - Need IPC (HTTP/gRPC) between orchestratord ↔ pool-managerd
   - Two systemd units instead of one
   - More moving parts

2. **🔌 Need HTTP API**
   - Implement control API in pool-managerd
   - orchestratord calls pool-managerd HTTP endpoints
   - Need to handle connection failures

3. **🐛 More Failure Modes**
   - pool-managerd could be down when orchestratord starts
   - Network issues between processes (even on localhost)
   - Need health checks and retry logic

4. **📝 More Configuration**
   - Need to configure pool-managerd address
   - Need to configure ports
   - Need to handle auth between processes (optional)

5. **🏗️ More Work NOW**
   - Implement HTTP API (routes for health, drain, reload)
   - Update orchestratord to call pool-managerd
   - Write systemd units
   - Test inter-process communication

### Implementation Effort

- **~2-3 hours** to implement HTTP API
- **~1 hour** to update orchestratord
- **~30 min** to write systemd units
- **Total: ~4 hours**

---

## Option B: Wait for CLOUD_PROFILE ⏳

### Pros

1. **⚡ Simpler NOW**
   - Keep orchestratord embedding pool-managerd
   - No IPC overhead
   - Single binary deployment
   - Fewer moving parts

2. **🎯 YAGNI (You Ain't Gonna Need It)**
   - Home profile doesn't need separate processes
   - Don't build for future requirements
   - Implement when actually needed

3. **🚀 Faster to v1.0**
   - No extra work now
   - Focus on core features
   - Defer complexity

4. **🔧 Less to Maintain**
   - One binary to build/test/deploy
   - No HTTP API to maintain
   - No IPC to debug

### Cons

1. **🔄 Migration Pain Later**
   - Need to refactor orchestratord when scaling
   - Need to implement HTTP API under pressure
   - Risk of architectural debt

2. **🧪 Can't Test Separation**
   - Can't validate daemon architecture until cloud
   - Risk of discovering issues late
   - Harder to refactor under production pressure

3. **🏗️ Architectural Confusion**
   - Is pool-managerd a library or daemon?
   - Unclear from workspace structure
   - Mixing concerns in orchestratord

4. **📦 Deployment Inconsistency**
   - Home: embedded
   - Cloud: separate daemon
   - Different architectures = different bugs

5. **🔐 No Security Boundary**
   - Engine processes run in same process as orchestratord
   - Crash in engine could affect orchestratord
   - No process isolation

6. **📁 libs/ Location is Wrong**
   - It's not really a library if it has main.rs
   - Should be in bin/ if it's a daemon
   - Confusing workspace structure

### Implementation Effort

- **~0 hours NOW**
- **~6-8 hours LATER** (under pressure, with production constraints)

---

## Comparison Table

| Aspect | Daemon NOW | Wait for Cloud |
|--------|------------|----------------|
| **Architecture** | ✅ Clean separation | ⚠️ Mixed concerns |
| **Development** | ✅ Easier to test | ⚠️ Harder to isolate |
| **Deployment (Home)** | ⚠️ Two processes | ✅ One binary |
| **Deployment (Cloud)** | ✅ Same architecture | ⚠️ Need migration |
| **Security** | ✅ Process isolation | ❌ No boundary |
| **Failure Isolation** | ✅ Independent crashes | ❌ Coupled |
| **Work NOW** | ⚠️ ~4 hours | ✅ 0 hours |
| **Work LATER** | ✅ 0 hours | ⚠️ ~6-8 hours |
| **Risk** | ✅ Low (validate early) | ⚠️ High (late discovery) |
| **Workspace Structure** | ✅ bin/pool-managerd | ⚠️ libs/pool-managerd |

---

## Recommendation: Make Daemon NOW ✅

### Rationale

1. **Better Architecture:** Clean separation from day 1
2. **Cloud-Ready:** No migration pain later
3. **Already 80% There:** preload.rs is done, just need HTTP API
4. **Low Risk:** Validate architecture early
5. **Correct Location:** Move to bin/ (it's a daemon, not a library)

### Why NOW is Better Than Later

- **No production pressure** — can take time to do it right
- **Can test thoroughly** — validate IPC, failure modes, etc.
- **Easier to change** — no users depending on embedded architecture
- **Learn early** — discover issues before cloud deployment
- **Correct structure** — bin/ location makes sense

### Implementation Plan

1. **Move to bin/** (5 min)

   ```bash
   git mv libs/pool-managerd bin/pool-managerd
   # Update Cargo.toml workspace members
   ```

2. **Implement HTTP API** (2-3 hours)

   ```rust
   // bin/pool-managerd/src/api/
   // - GET /health
   // - POST /pools/{id}/preload (body: PreparedEngine)
   // - POST /pools/{id}/drain
   // - POST /pools/{id}/reload
   // - GET /pools/{id}/status
   ```

3. **Update orchestratord** (1 hour)

   ```rust
   // Replace:
   // state.pool_manager.lock()
   
   // With:
   // http_client.post("http://localhost:9001/pools/{id}/preload")
   ```

4. **Write systemd units** (30 min)

   ```ini
   # /etc/systemd/system/pool-managerd.service
   # /etc/systemd/system/orchestratord.service
   ```

5. **Test** (1 hour)
   - Start pool-managerd
   - Start orchestratord
   - Verify IPC works
   - Test failure modes

**Total: ~4-5 hours**

---

## Alternative: Hybrid Approach (NOT RECOMMENDED)

Keep embedded for home, but structure code for daemon:

```rust
// orchestratord can:
// - Embed pool-managerd (home profile)
// - Call HTTP API (cloud profile)
```

**Why Not:**

- ❌ More complexity (two code paths)
- ❌ Need to test both modes
- ❌ Harder to maintain
- ❌ Doesn't solve workspace structure issue

---

## Decision Criteria

### Choose Daemon NOW if

- ✅ You want clean architecture from start
- ✅ You plan to do cloud profile eventually
- ✅ You want to validate separation early
- ✅ You have ~4 hours to implement
- ✅ You want correct workspace structure (bin/)

### Choose Wait if

- ✅ You need to ship v1.0 ASAP
- ✅ You'll never do cloud profile
- ✅ You want absolute simplicity for home
- ✅ You're okay with migration pain later

---

## User's Preference

> "I am leaning towards making it its own daemon and moving pool-managerd into the bin folder"

**This aligns with Option A (Daemon NOW) ✅**

---

## Next Steps (if Daemon NOW)

1. **Create decision record** (this file) ✅
2. **Move to bin/**

   ```bash
   git mv libs/pool-managerd bin/pool-managerd
   ```

3. **Implement HTTP API**
   - Add axum dependency
   - Create routes (health, preload, drain, status)
   - Wire to preload.rs
4. **Update orchestratord**
   - Add HTTP client
   - Replace registry.lock() with HTTP calls
5. **Write systemd units**
6. **Test end-to-end**
7. **Update documentation**

---

## Conclusion

**Recommendation: Make pool-managerd a standalone daemon NOW**

**Reasons:**

1. ✅ Cleaner architecture
2. ✅ Cloud-ready from start
3. ✅ Better development experience
4. ✅ Correct workspace structure (bin/)
5. ✅ Low risk (validate early)
6. ✅ Only ~4 hours of work
7. ✅ Aligns with user's preference

**Move forward with Option A: Daemon NOW** 🚀
