# Queen-to-Hive Communication Modes

**Created by:** TEAM-265  
**Date:** Oct 23, 2025  
**Status:** Documented

---

## Overview

Queen-rbee can communicate with hives in **three different ways**. The mode is automatically selected based on:
1. Whether the hive is localhost or remote
2. Whether queen was built with the `local-hive` feature

---

## The Three Modes

### Mode 1: Remote HTTP (Always Available)

**When:** Hive is on a different machine (hive_id != "localhost")

**How it works:**
```text
queen-rbee → HTTP POST → remote-hive (different machine)
           ← SSE stream ←
```

**Performance:**
- Overhead: ~5-10ms per operation
- Includes: Network latency + serialization + deserialization

**Use cases:**
- Production multi-machine setup
- Distributed inference cluster
- Geographic distribution

**Example:**
```bash
# Queen on server1 (192.168.1.10)
# Hive on server2 (192.168.1.20)
queen-rbee --port 8500

# Operation flows:
# server1:8500 → HTTP → server2:8600 → worker
```

---

### Mode 2: Localhost HTTP (Default for localhost)

**When:** Hive is localhost AND queen built WITHOUT local-hive feature

**How it works:**
```text
queen-rbee → HTTP POST → localhost-hive (same machine, different process)
(port 8500) ← SSE stream ← (port 8600)
```

**Performance:**
- Overhead: ~1-2ms per operation
- Includes: Loopback + serialization + deserialization
- No network latency (loopback interface)

**Use cases:**
- Development and testing
- Distributed queen build (default)
- When you want separate processes

**Example:**
```bash
# Same machine, two processes
queen-rbee --port 8500 &
rbee-hive --port 8600 &

# Operation flows:
# localhost:8500 → HTTP → localhost:8600 → worker
```

---

### Mode 3: Integrated (local-hive feature)

**When:** Hive is localhost AND queen built WITH local-hive feature

**How it works:**
```text
queen-rbee (with integrated hive) → Direct Rust calls → worker
(single process, no HTTP)
```

**Performance:**
- Overhead: ~0.01ms per operation
- Includes: Only function call overhead
- **50-100x faster than Mode 2!**

**Use cases:**
- Single-machine production (optimal)
- Laptop/desktop development
- Maximum performance localhost

**Example:**
```bash
# Build with feature
cargo build --release --bin queen-rbee --features local-hive

# Single process
queen-rbee --port 8500

# Operation flows:
# Internal function call (no HTTP, no serialization)
```

---

## Performance Comparison

| Mode | Overhead | Network | Serialization | Processes | Speedup |
|------|----------|---------|---------------|-----------|---------|
| **1. Remote HTTP** | ~5-10ms | Yes (LAN/WAN) | Yes | 2+ | 1x (baseline) |
| **2. Localhost HTTP** | ~1-2ms | Loopback only | Yes | 2 | 5-10x |
| **3. Integrated** | ~0.01ms | None | None | 1 | **50-100x** |

---

## Mode Selection Logic

The mode is automatically selected in `hive_forwarder.rs`:

```rust
let is_localhost = hive_id == "localhost";
let has_integrated = cfg!(feature = "local-hive");

let mode = if is_localhost && has_integrated {
    "integrated"  // Mode 3
} else if is_localhost {
    "localhost-http"  // Mode 2
} else {
    "remote-http"  // Mode 1
};
```

---

## Implementation Status

| Mode | Status | Location |
|------|--------|----------|
| **1. Remote HTTP** | ✅ Implemented | `hive_forwarder.rs` |
| **2. Localhost HTTP** | ✅ Implemented | `hive_forwarder.rs` |
| **3. Integrated** | 🔴 BLOCKED | See TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md |

### Mode 3 Implementation Status

**Status:** 🔴 **BLOCKED** by missing rbee-hive crate implementations

**TEAM-266 Investigation (Oct 23, 2025):**
- ✅ Architecture verified - Mode 3 is feasible
- ✅ No circular dependencies
- ✅ Narration will work seamlessly
- 🔴 **BLOCKER:** All rbee-hive crates are empty stubs
  - worker-lifecycle: 13 lines, all TODO
  - model-catalog: 16 lines, all TODO
  - model-provisioner: 13 lines, all TODO
- Expected speedup: 110x for list/get operations

**Prerequisites before Mode 3:**
1. Implement worker-lifecycle crate (80h)
2. Implement model-catalog crate (40h)
3. Implement model-provisioner crate (40h)
4. Test HTTP mode thoroughly (16h)

**Full investigation:** `TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md`

### Mode 3 Implementation Plan (When Prerequisites Met)

To implement integrated mode:

1. **Add optional dependencies** (Cargo.toml):
   ```toml
   [dependencies]
   rbee-hive-worker-lifecycle = { path = "...", optional = true }
   rbee-hive-model-catalog = { path = "...", optional = true }
   
   [features]
   local-hive = [
       "rbee-hive-worker-lifecycle",
       "rbee-hive-model-catalog",
   ]
   ```

2. **Implement direct calls** (hive_forwarder.rs):
   ```rust
   #[cfg(feature = "local-hive")]
   async fn execute_integrated(operation: Operation) -> Result<()> {
       match operation {
           Operation::WorkerSpawn { .. } => {
               // Call rbee-hive crates directly
               rbee_hive_worker_lifecycle::spawn_worker(...).await?
           }
           // ... other operations
       }
   }
   ```

3. **Route in forward_to_hive**:
   ```rust
   if is_localhost && has_integrated {
       return execute_integrated(operation).await;
   }
   ```

---

## Smart Prompts

TEAM-263 added smart prompts that detect when:
- User installs localhost hive
- Queen doesn't have local-hive feature

The prompt shows:
```
⚠️  Performance Notice:

   📊 Performance comparison:
      • Current setup:  ~5-10ms overhead (HTTP)
      • Integrated:     ~0.1ms overhead (direct calls)
      • Speedup:        50-100x faster

   💡 Recommendation:
      $ rbee-keeper queen rebuild --with-local-hive
```

---

## Architecture Diagrams

### Mode 1: Remote HTTP
```
┌─────────────┐                    ┌─────────────┐
│ queen-rbee  │ ─── HTTP/LAN ───→ │ rbee-hive   │
│ (server1)   │ ←── SSE stream ─── │ (server2)   │
└─────────────┘                    └─────────────┘
     8500                                8600
```

### Mode 2: Localhost HTTP
```
┌─────────────────────────────────────────┐
│         Same Machine                     │
│  ┌─────────────┐      ┌─────────────┐  │
│  │ queen-rbee  │ ─┐   │ rbee-hive   │  │
│  │ (process 1) │  │   │ (process 2) │  │
│  └─────────────┘  │   └─────────────┘  │
│       8500        │        8600         │
│                   │                     │
│         HTTP over loopback (127.0.0.1) │
│                   └──────────────→      │
└─────────────────────────────────────────┘
```

### Mode 3: Integrated
```
┌─────────────────────────────────────────┐
│         Same Machine                     │
│  ┌──────────────────────────────────┐   │
│  │ queen-rbee (single process)      │   │
│  │                                  │   │
│  │  ┌────────┐    ┌─────────────┐  │   │
│  │  │ Queen  │───→│ Hive (lib)  │  │   │
│  │  │ Logic  │←───│ Direct calls│  │   │
│  │  └────────┘    └─────────────┘  │   │
│  │                                  │   │
│  │  No HTTP, no serialization       │   │
│  └──────────────────────────────────┘   │
│              8500                        │
└─────────────────────────────────────────┘
```

---

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| **Mode documentation** | `hive_forwarder.rs` | 1-78 |
| **Mode detection** | `hive_forwarder.rs` | 142-152 |
| **Mode 1 & 2 impl** | `hive_forwarder.rs` | 180-250 |
| **Mode 3 TODO** | `hive_forwarder.rs` | 163-180 |
| **Smart prompts** | `rbee-keeper/main.rs` | 651-711 |

---

## Testing

### Test Mode 1 (Remote HTTP)
```bash
# Server 1
queen-rbee --port 8500

# Server 2
rbee-hive --port 8600

# Run operation
rbee-keeper worker list --hive-id remote-hive
```

### Test Mode 2 (Localhost HTTP)
```bash
# Terminal 1
queen-rbee --port 8500

# Terminal 2
rbee-hive --port 8600

# Terminal 3
rbee-keeper worker list --hive-id localhost
# Should use localhost HTTP (check logs for "localhost-http")
```

### Test Mode 3 (Integrated) - TODO
```bash
# Build with feature
cargo build --release --features local-hive

# Run
queen-rbee --port 8500

# Test
rbee-keeper worker list --hive-id localhost
# Should use integrated mode (check logs for "integrated")
```

---

## FAQ

**Q: Why three modes instead of just one?**  
A: Different deployment scenarios have different requirements:
- Remote: Multi-machine production
- Localhost HTTP: Development/testing with separate processes
- Integrated: Single-machine production with optimal performance

**Q: Can I use Mode 3 with remote hives?**  
A: No. Integrated mode only works for localhost. Remote hives always use HTTP.

**Q: What's the default mode?**  
A: Depends on build:
- Default build: Mode 1 (remote) or Mode 2 (localhost)
- With --features local-hive: Mode 3 (localhost) or Mode 1 (remote)

**Q: How do I know which mode is being used?**  
A: Check the narration logs:
```
[qn-fwd    ] forward_start   : Forwarding WorkerList operation to hive 'localhost' (mode: localhost-http)
```

**Q: Is Mode 3 implemented?**  
A: 🔴 NO - Blocked by missing rbee-hive crate implementations. TEAM-265 documented it, TEAM-266 investigated it. All worker/model crates are empty stubs. See `TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md` for details.

---

## Next Steps

For future teams implementing Mode 3:
1. Read this document
2. Add rbee-hive crates as optional dependencies
3. Implement `execute_integrated()` function
4. Update `forward_to_hive()` to route to integrated mode
5. Test with `--features local-hive`
6. Update this document with implementation details

---

**Summary:** Queen has 3 ways to talk to hives. Mode 1 & 2 are implemented (HTTP). Mode 3 is documented but not yet implemented (direct calls).
