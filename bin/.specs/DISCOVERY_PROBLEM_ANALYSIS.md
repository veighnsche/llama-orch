# Discovery Problem Analysis

**Date:** Oct 30, 2025  
**Status:** CRITICAL ARCHITECTURAL GAP  
**Purpose:** Analyze the worker/hive discovery problem and evaluate solutions

---

## 🚨 The Problem

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
- ❌ **Discovery broken**

---

## 🔍 Discovery Scenarios

### **Scenario 1: Queen → Hive → Worker (REJECTED)**

```
Queen starts
    ↓
Queen discovers Hive via SSH config
    ↓
Worker starts on Hive
    ↓
Worker sends heartbeat to Hive ❌ (User doesn't want this)
    ↓
Hive has worker registry ❌ (User doesn't want this)
    ↓
Queen fetches capabilities from Hive (includes workers)
```

**Why rejected:** User explicitly does NOT want Hive to have worker registry or receive worker heartbeats.

---

### **Scenario 2: Queenless Workers (GUI-spawned)**

```
GUI (rbee-keeper) starts
    ↓
User spawns worker via GUI
    ↓
Worker starts on port 8080
    ↓
Worker tries to send heartbeat to Queen... but Queen doesn't exist yet
    ↓
Worker can't blindly send heartbeats to nowhere ❌
```

**Problem:** How does Queen discover workers that started before Queen?

---

### **Scenario 3: Queen Starts After Workers**

```
Workers running (8000, 8080, 8188)
    ↓
Queen starts
    ↓
Queen needs to discover all existing workers
    ↓
How??? ❌
```

**Problem:** Queen doesn't know what ports to check.

---

### **Scenario 4: Worker Dynamic Ports**

```
Worker 1: llm-worker on port 8080
Worker 2: llm-worker on port 8081 (port 8080 was taken)
Worker 3: llm-worker on port 8082 (port 8081 was taken)
```

**Problem:** Ports are not fixed. Queen can't assume "llm-worker is always on 8080".

---

## 🎯 Requirements

### **Must Have:**

1. ✅ Queen discovers all hives (SOLVED: SSH config)
2. ✅ Queen discovers all workers (UNSOLVED)
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

## 🔥 Edge Cases

### **Edge Case 1: Port Collision**

```
Worker 1 tries to bind 8080 → Success
Worker 2 tries to bind 8080 → Fail, retry 8081
Worker 3 tries to bind 8080 → Fail, retry 8081 → Fail, retry 8082
```

**Problem:** Can't assume fixed ports.

---

### **Edge Case 2: Multiple Hives, Multiple Workers**

```
Hive 1 (machine-a:7835) spawns:
  - Worker 1 (machine-a:8080)
  - Worker 2 (machine-a:8081)

Hive 2 (machine-b:7835) spawns:
  - Worker 3 (machine-b:8080)
  - Worker 4 (machine-b:8081)

GUI (machine-c) spawns:
  - Worker 5 (machine-c:8080)
```

**Problem:** Queen needs to discover workers across multiple machines.

---

### **Edge Case 3: Worker Crash and Restart**

```
Worker 1 (8080) crashes
Worker 1 restarts on 8080
```

**Problem:** Queen needs to know it's the same worker (or a new one).

---

### **Edge Case 4: Queenless Operation**

```
GUI spawns Worker 1 (8080)
User does inference via GUI → Worker 1
Queen starts later
Queen needs to discover Worker 1
```

**Problem:** Worker didn't know Queen URL at startup.

---

## 💡 Solution Criteria

### **Good Solution Must:**

1. **Systematic** - Not ad-hoc, follows clear pattern
2. **Scalable** - Works with 1 worker or 100 workers
3. **Resilient** - Handles startup order variations
4. **Simple** - Easy to understand and maintain
5. **No persistent data** - In-memory only for lifecycles
6. **No Hive worker registry** - Respects user constraint
7. **Works Queenless** - GUI can operate without Queen

---

## 🎯 Solution Categories

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
