# Discovery Problem Analysis

**Date:** Oct 30, 2025  
**Status:** ✅ SOLVED  
**Purpose:** Document the discovery problem and final solution

**Canonical Solution:** See [`HEARTBEAT_ARCHITECTURE.md`](./HEARTBEAT_ARCHITECTURE.md)

---

## 🎯 Final Architecture (Post-Refinement)

### **Key Constraints:**

1. **Workers do not send heartbeats** - Workers never communicate directly with Queen
2. **Hive performs system-level monitoring** - Via cgroup v2 tree (`rbee.slice/<service>/<instance>`)
3. **Registry provides discovery only** - No metric fields, only existence/liveness
4. **Telemetry flows Hive → Queen** - Single authoritative path for live stats

### **Discovery Solution:**

**Hives:** Static port (7835), discovered via SSH config  
**Workers:** Monitored by Hive via cgroup v2, reported in Hive telemetry

---

## 🚨 The Original Problem

### **Core Issue: Variable Ports Break Discovery**

**Hives:**
- Static port: `7835`
- Static hostname in SSH config
- Queen can discover via SSH config → GET /capabilities
- ✅ **Discovery works**

**Workers:**
- Variable ports: `8000` (vllm), `8080` (llm), `8188` (comfy)
- Dynamic spawning
- No static hostname/port mapping
- ❌ **Discovery was broken** (now solved via Hive telemetry)

---

## 🔍 Discovery Scenarios

### **Scenario 1: Hive Monitors Workers (FINAL SOLUTION)**

```
Worker starts on Hive (any port: 8080, 8081, etc.)
    ↓
Hive spawns worker in cgroup: rbee.slice/llm/8080
    ↓
Hive Monitor polls cgroup tree every ~1s
    ↓
Hive collects: CPU%, RSS, VRAM, I/O, PIDs, state
    ↓
Hive sends telemetry to Queen (includes all workers)
    ↓
Queen discovers worker via Hive telemetry
```

**Why this works:**
- ✅ Workers never send heartbeats (no cooperation needed)
- ✅ Hive monitors via OS (cgroups) - workers can't lie
- ✅ Single telemetry path: Hive → Queen
- ✅ Works for any port (dynamic ports solved)
- ✅ Works Queenless (Hive still monitors, just doesn't send telemetry)

---

### **Scenario 2: Worker Dynamic Ports (SOLVED)**

```
Worker 1: llm-worker on port 8080 → cgroup: rbee.slice/llm/8080
Worker 2: llm-worker on port 8081 → cgroup: rbee.slice/llm/8081
Worker 3: llm-worker on port 8082 → cgroup: rbee.slice/llm/8082
```

**Solution:** Hive enumerates cgroup tree, discovers all workers regardless of port.

---

### **Scenario 3: Queen Starts After Workers (SOLVED)**

```
Workers running (8000, 8080, 8188) - monitored by Hive
    ↓
Queen starts
    ↓
Queen sends GET /capabilities?queen_url=... to Hive
    ↓
Hive starts telemetry to Queen
    ↓
First telemetry includes all existing workers
    ↓
Queen discovers all workers immediately
```

**Solution:** Hive telemetry includes snapshot of all running workers.

---

## 🎯 Requirements

### **Must Have:**

1. ✅ Queen discovers all hives (SOLVED: SSH config)
2. ✅ Queen discovers all workers (SOLVED: Hive telemetry)
3. ✅ Works if Queen starts first
4. ✅ Works if Workers start first (Queenless scenario)
5. ✅ Works with dynamic worker ports
6. ✅ No Hive worker registry
7. ✅ No Hive receiving worker heartbeats
8. ✅ Systematic (not ad-hoc)
9. ✅ No persistent data for lifecycles (in-memory only)

### **Nice to Have:**

- Works across multiple machines
- Works on local network without Queen
- Minimal configuration
- Self-healing (auto-discover new workers)

---

## 🤔 Key Questions

### **Q1: Who spawns workers?**

**Answer:**
- **GUI (rbee-keeper):** Spawns workers for Queenless scenarios
- **Hive:** Spawns workers when Queen requests (via job)

**Implication:** The spawner knows the worker's port!

---

### **Q2: Can we leverage the spawner?**

**Answer:** YES! The component that spawns a worker knows:
- Worker type (llm, vllm, comfy)
- Worker port (assigned at spawn)
- Worker PID

**Implication:** Spawner can inform Queen about new worker.

---

### **Q3: What if Queen doesn't exist yet?**

**Answer:** Need a mechanism for:
- Queenless discovery (GUI queries workers directly)
- Delayed registration (worker remembers to register when Queen appears)
- OR: Central registry that both Queen and GUI use

---

### **Q4: What about cross-machine workers?**

**Answer:** If worker runs on different machine than Queen:
- Need hostname + port
- Need network reachability
- SSH config might not include worker machines

---

## 📊 Architecture Constraints

### **Current Architecture:**

```
┌─────────────┐
│ rbee-keeper │ (GUI - Tauri app)
│  (7811)     │
└──────┬──────┘
       │
       │ Can spawn workers locally
       │
       ├──────────────┐
       ▼              ▼
  ┌─────────┐   ┌──────────┐
  │  Hive   │   │ Workers  │
  │ (7835)  │   │ (8000+)  │
  └────┬────┘   └──────────┘
       │
       │ SSH config
       │
       ▼
  ┌─────────┐
  │  Queen  │
  │ (7833)  │
  └─────────┘
```

### **Constraints:**

1. **Hive:** Static port (7835), SSH config
2. **Worker:** Dynamic port (8000+), no SSH config
3. **Queen:** Needs to discover both
4. **GUI:** Can spawn workers without Queen
5. **No Hive worker registry**
6. **No Hive worker heartbeats**

---

## 🔥 Edge Cases (All Solved)

### **Edge Case 1: Port Collision** ✅

```
Worker 1 tries to bind 8080 → Success → cgroup: rbee.slice/llm/8080
Worker 2 tries to bind 8080 → Fail, retry 8081 → cgroup: rbee.slice/llm/8081
Worker 3 tries to bind 8080 → Fail, retry 8081 → Fail, retry 8082 → cgroup: rbee.slice/llm/8082
```

**Solution:** Hive enumerates cgroup tree, discovers all workers regardless of port.

---

### **Edge Case 2: Multiple Hives, Multiple Workers** ✅

```
Hive 1 (machine-a:7835) monitors:
  - Worker 1 (machine-a:8080) → Hive 1 telemetry
  - Worker 2 (machine-a:8081) → Hive 1 telemetry

Hive 2 (machine-b:7835) monitors:
  - Worker 3 (machine-b:8080) → Hive 2 telemetry
  - Worker 4 (machine-b:8081) → Hive 2 telemetry

GUI (machine-c) spawns:
  - Worker 5 (machine-c:8080) → No telemetry (Queenless)
```

**Solution:** Each Hive sends telemetry for its own workers. Queen aggregates.

---

### **Edge Case 3: Worker Crash and Restart** ✅

```
Worker 1 (8080) crashes → disappears from cgroup
Hive telemetry: workers[] no longer includes Worker 1
Queen marks Worker 1 as down

Worker 1 restarts on 8080 → new cgroup entry
Hive telemetry: workers[] includes new Worker 1
Queen discovers new worker instance
```

**Solution:** Hive telemetry reflects current cgroup state. Workers auto-discovered/removed.

---

### **Edge Case 4: Queenless Operation** ✅

```
GUI spawns Worker 1 (8080)
Hive monitors Worker 1 via cgroup (no telemetry sent - no Queen)
User does inference via GUI → Worker 1
Queen starts later
Queen sends GET /capabilities?queen_url=... to Hive
Hive starts telemetry (includes Worker 1)
Queen discovers Worker 1 immediately
```

**Solution:** Hive always monitors workers. Telemetry starts when Queen discovered.

---

## ✅ Final Solution Summary

**Implemented:** Hive monitors workers via cgroup v2, reports in telemetry to Queen.

**Key Benefits:**
1. ✅ **Systematic** - Clear cgroup enumeration pattern
2. ✅ **Scalable** - Works with 1 worker or 100 workers
3. ✅ **Resilient** - Handles all startup order variations
4. ✅ **Simple** - OS-level monitoring, no worker cooperation
5. ✅ **No persistent data** - In-memory cgroup polling
6. ✅ **No Hive worker registry** - Just telemetry reporting
7. ✅ **Works Queenless** - Hive monitors regardless of Queen

**See:** [`HEARTBEAT_ARCHITECTURE.md`](./HEARTBEAT_ARCHITECTURE.md) for full implementation details.

---

## 🗑️ Deprecated Solution Documents (Rule Zero)

The following solution documents are **DEPRECATED** and should be deleted:

- ❌ `SOLUTION_1_REGISTRY_SERVICE.md` - Registry not needed for worker discovery
- ❌ `SOLUTION_2_PARENT_CHILD_REGISTRATION.md` - Workers don't register
- ❌ `SOLUTION_3_HYBRID.md` - Hybrid approach not needed
- ❌ `SOLUTION_COMPARISON.md` - No alternatives to compare
- ❌ `HEARTBEAT_CONSOLIDATION_ANALYSIS.md` - Superseded by HEARTBEAT_ARCHITECTURE.md
- ❌ `HEARTBEAT_IMPLEMENTATION_SUMMARY.md` - Superseded by HEARTBEAT_ARCHITECTURE.md

**Reason:** Problem solved via Hive telemetry monitoring. No alternative solutions needed.

---

## 🗂️ Historical Context (Deprecated Solution Categories)

### **Category A: Central Registry (New Daemon)**

Create a separate daemon responsible for tracking all components.

**Pros:**
- Single source of truth
- Works Queenless
- Systematic

**Cons:**
- New daemon to maintain
- New failure point
- More complexity

---

### **Category B: Parent-Child Registration**

Spawner informs Queen about new worker.

**Pros:**
- Leverages existing knowledge (spawner knows port)
- No new daemons
- Simple

**Cons:**
- Requires Queen to exist
- Doesn't work Queenless

---

### **Category C: Network Discovery (Multicast/Broadcast)**

Components announce themselves on network.

**Pros:**
- No central authority
- Self-healing
- Works Queenless

**Cons:**
- Complex networking
- May not work across subnets
- Security concerns

---

### **Category D: Port Scanning**

Queen actively scans port ranges.

**Pros:**
- No worker cooperation needed
- Simple implementation

**Cons:**
- Slow
- Wasteful (scan 200 ports?)
- May trigger firewalls

---

### **Category E: File-based Discovery**

Components write PID files to shared location.

**Pros:**
- Simple
- Works Queenless
- Cross-platform

**Cons:**
- File system dependency
- Stale files (crash cleanup)
- Doesn't work across machines

---

## 🏆 Recommended Solutions (Detailed in Separate Docs)

1. **Solution 1: Registry Service (rbee-registry)** → See `SOLUTION_1_REGISTRY_SERVICE.md`
2. **Solution 2: Parent-Child Registration** → See `SOLUTION_2_PARENT_CHILD_REGISTRATION.md`
3. **Solution 3: Hybrid (Registry + Parent-Child)** → See `SOLUTION_3_HYBRID.md`
4. **Solution 4: Network Discovery (mDNS/Avahi)** → See `SOLUTION_4_NETWORK_DISCOVERY.md`
5. **Solution 5: File-based Discovery** → See `SOLUTION_5_FILE_BASED_DISCOVERY.md`
6. **Solution 6: Port Scanning with Caching** → See `SOLUTION_6_PORT_SCANNING.md`

---

## 📝 Summary

**The Problem:** Workers have dynamic ports, Queen can't discover them systematically.

**Root Cause:** Heartbeat architecture assumes Queen knows about workers before they heartbeat.

**User Constraints:**
- No Hive worker registry
- No Hive worker heartbeats
- Must work Queenless (GUI-spawned workers)

**Next Steps:** Evaluate each solution in detail (see solution docs).

---

**This document defines the problem. See individual solution docs for proposals.**
