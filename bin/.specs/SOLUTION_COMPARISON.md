# Discovery Solutions Comparison

**Date:** Oct 30, 2025  
**Purpose:** Compare all proposed solutions for worker/hive discovery  
**Problem:** See `DISCOVERY_PROBLEM_ANALYSIS.md`

---

## 📊 Solutions Overview

| Solution | Approach | Complexity | Rating |
|----------|----------|------------|--------|
| **Solution 1** | Registry Service (rbee-registry) | HIGH | ⭐⭐⭐⭐⭐ |
| **Solution 2** | Parent-Child Registration | MEDIUM | ⭐⭐⭐⭐ |
| **Solution 3** | Hybrid (Registry + Parent-Child) | MEDIUM-HIGH | ⭐⭐⭐⭐⭐ |
| Solution 4 | Network Discovery (mDNS) | VERY HIGH | ⭐⭐ |
| Solution 5 | File-based Discovery | LOW | ⭐⭐⭐ |
| Solution 6 | Port Scanning | LOW | ⭐⭐ |

---

## 🎯 Requirements Matrix

| Requirement | Sol 1 | Sol 2 | Sol 3 | Sol 4 | Sol 5 | Sol 6 |
|-------------|-------|-------|-------|-------|-------|-------|
| Discover hives | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Discover workers | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Queen starts first | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Worker starts first | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Queenless operation | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Dynamic ports | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| No Hive worker registry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cross-machine | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚠️ |
| Systematic | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ❌ |
| No persistent data | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

**Legend:** ✅ Full support | ⚠️ Partial support | ❌ No support

---

## 💡 Solution 1: Registry Service

**Approach:** New daemon `rbee-registry` on port 7830

**Pros:**
- ✅ Solves ALL requirements
- ✅ Systematic (single source of truth)
- ✅ Works Queenless
- ✅ In-memory only (no persistent data)
- ✅ Self-healing
- ✅ Cross-machine

**Cons:**
- ❌ New daemon to maintain
- ❌ Single point of failure
- ❌ Need to start first

**Verdict:** ⭐⭐⭐⭐⭐ **HIGHLY RECOMMENDED**

**See:** `SOLUTION_1_REGISTRY_SERVICE.md`

---

## 💡 Solution 2: Parent-Child Registration

**Approach:** Spawner tells Queen about new workers

**Pros:**
- ✅ Simple (no new daemon)
- ✅ Leverages existing knowledge
- ✅ Fast (immediate notification)

**Cons:**
- ❌ Requires Queen to exist
- ❌ Can't discover pre-existing workers
- ❌ Breaks Queenless requirement

**Verdict:** ⭐⭐⭐⭐ **Good as part of hybrid, insufficient alone**

**See:** `SOLUTION_2_PARENT_CHILD_REGISTRATION.md`

---

## 💡 Solution 3: Hybrid (Registry + Parent-Child)

**Approach:** Registry for discovery + Spawner notification for optimization

**Pros:**
- ✅ Solves ALL requirements
- ✅ Fast path optimization
- ✅ Resilient (dual mechanisms)
- ✅ Works Queenless

**Cons:**
- ❌ More complexity (two mechanisms)
- ❌ Still need registry daemon
- ❌ More coordination

**Verdict:** ⭐⭐⭐⭐⭐ **HIGHLY RECOMMENDED (best of both worlds)**

**See:** `SOLUTION_3_HYBRID.md`

---

## 💡 Solution 4: Network Discovery (mDNS/Avahi)

**Approach:** Components broadcast themselves via multicast DNS

**How it works:**
```rust
// Each component broadcasts
service_name = "_rbee._tcp.local"
txt_records = {
    "component_type": "worker",
    "port": "8080",
    "worker_type": "llm"
}

// Queen discovers via mDNS browser
let services = mdns_browser.browse("_rbee._tcp.local")?;
for service in services {
    // Resolve service to get hostname + port
}
```

**Pros:**
- ✅ No central authority
- ✅ Self-healing
- ✅ Works Queenless
- ✅ Standard protocol

**Cons:**
- ❌ Complex networking (multicast, DNS-SD)
- ❌ May not work across subnets
- ❌ Platform-specific (needs Avahi/Bonjour)
- ❌ Security concerns (broadcast to network)
- ❌ Overkill for local discovery

**Verdict:** ⭐⭐ **Too complex, not worth it**

---

## 💡 Solution 5: File-based Discovery

**Approach:** Components write PID files to shared directory

**How it works:**
```
~/.cache/rbee/registry/
├── workers/
│   ├── worker-llm-8080.json
│   ├── worker-vllm-8000.json
│   └── worker-comfy-8188.json
├── hives/
│   └── hive-localhost.json
└── queen.json

# worker-llm-8080.json
{
  "worker_id": "worker-llm-8080",
  "port": 8080,
  "worker_type": "llm",
  "pid": 12345,
  "started_at": "2025-10-30T15:00:00Z"
}
```

**Pros:**
- ✅ Simple implementation
- ✅ Works Queenless
- ✅ Cross-platform
- ✅ No network dependency

**Cons:**
- ❌ Persistent data (violates requirement)
- ❌ Stale files (if process crashes)
- ❌ File system cleanup needed
- ❌ Doesn't work cross-machine
- ❌ Race conditions (file locking)

**Verdict:** ⭐⭐⭐ **Simple but violates "no persistent data" requirement**

---

## 💡 Solution 6: Port Scanning

**Approach:** Queen actively scans port ranges to find workers

**How it works:**
```rust
// Queen on startup
for port in 8000..8200 {
    if let Ok(response) = try_connect(port).await {
        if response.is_rbee_component() {
            // Found a worker!
            register_worker(port);
        }
    }
}
```

**Pros:**
- ✅ No worker cooperation needed
- ✅ Simple to implement

**Cons:**
- ❌ Slow (scan 200 ports?)
- ❌ Wasteful (most ports empty)
- ❌ May trigger firewalls
- ❌ Doesn't work Queenless
- ❌ Not systematic
- ❌ Can't distinguish component types

**Verdict:** ⭐⭐ **Inefficient and unreliable**

---

## 🏆 Recommendations

### **Recommended: Solution 3 (Hybrid)**

**Why:**
- Solves ALL requirements
- Fast path for common case
- Resilient (dual mechanisms)
- Systematic (registry is source of truth)
- Works Queenless

**Implementation:**
1. Create `rbee-registry` daemon (port 7830)
2. All components register with registry
3. Spawners also notify Queen directly (optimization)
4. Queen queries registry on startup for existing components

**Trade-off:** More complexity, but most robust solution.

---

### **Alternative: Solution 1 (Registry Service)**

**Why:**
- Simpler than hybrid (single mechanism)
- Still solves all requirements
- Easier to reason about

**Trade-off:** Slightly slower than hybrid (no fast path).

---

### **NOT Recommended:**

- ❌ **Solution 2 alone** - Breaks Queenless requirement
- ❌ **Solution 4** - Too complex, overkill
- ❌ **Solution 5** - Violates no-persistent-data requirement
- ❌ **Solution 6** - Inefficient, unreliable

---

## 📋 Implementation Plan (Solution 3)

### **Phase 1: Registry Service (4 weeks)**

**Week 1-2: Core Registry**
- [ ] Create `bin/05_rbee_registry` binary
- [ ] Implement registration API
- [ ] Implement query API
- [ ] Implement heartbeat tracking
- [ ] Add stale component cleanup

**Week 3-4: Registry Client**
- [ ] Create `bin/99_shared_crates/registry-client` crate
- [ ] Implement registration logic
- [ ] Implement heartbeat logic
- [ ] Implement query logic
- [ ] Add tests

---

### **Phase 2: Update All Components (3 weeks)**

**Week 1: Contracts**
- [ ] Update component contracts to include registry fields
- [ ] Add registration payloads
- [ ] Add query response types

**Week 2: Component Updates**
- [ ] Update Queen to query registry on startup
- [ ] Update Hive to register with registry
- [ ] Update Worker to register with registry
- [ ] Update GUI to use registry

**Week 3: Testing**
- [ ] Test all startup scenarios
- [ ] Test Queenless operation
- [ ] Test cross-machine discovery

---

### **Phase 3: Fast Path Optimization (2 weeks)**

**Week 1: Queen API**
- [ ] Add `/v1/queen/register-worker` endpoint
- [ ] Add `/v1/queen/register-hive` endpoint
- [ ] Update WorkerRegistry/HiveRegistry

**Week 2: Spawner Updates**
- [ ] Update GUI spawner to notify Queen
- [ ] Update Hive spawner to notify Queen
- [ ] Add fallback logic (registry if Queen unreachable)
- [ ] Test optimization

---

### **Phase 4: Documentation & Rollout (1 week)**

- [ ] Update all documentation
- [ ] Create migration guide
- [ ] Update deployment instructions
- [ ] Add troubleshooting guide

**Total: 10 weeks**

---

## 🎯 Decision Matrix

### **Choose Solution 1 if:**
- You want simplicity
- You don't mind slight delay in discovery
- You want single mechanism

### **Choose Solution 3 if:**
- You want best performance
- You want resilience (dual mechanisms)
- You don't mind extra complexity

### **Choose Something Else if:**
- You have specific constraints not covered above
- (But probably reconsider - Solutions 1/3 cover 99% of cases)

---

## Summary

**Problem:** Workers have dynamic ports, Queen can't discover them systematically.

**Best Solution:** Solution 3 (Hybrid) - Registry Service + Parent-Child Registration

**Why:** Solves ALL requirements, optimizes common case, resilient, systematic.

**Recommendation:** Implement Solution 3 (or Solution 1 if you want simpler).

---

**See individual solution documents for detailed designs.**
