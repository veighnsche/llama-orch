# TEAM-261 Pivot Decision Matrix: Hive as CLI vs Daemon

**Date:** Oct 23, 2025  
**Decision:** Should rbee-hive be a CLI tool or daemon?  
**Status:** 🎯 DECISION FRAMEWORK

---

## Quick Comparison

| Aspect | Daemon (Current) | CLI (Proposed) | Winner |
|--------|------------------|----------------|--------|
| **Performance** | 1-5ms per op | 80-350ms per op | 🟢 Daemon |
| **Real-time UX** | SSE streaming | No streaming | 🟢 Daemon |
| **Resource Usage** | 50-100 MB always | 0 MB idle | 🟡 CLI |
| **Complexity** | HTTP + job-server | Just CLI | 🟡 CLI |
| **Security** | HTTP auth | SSH + injection risk | 🟢 Daemon |
| **Reliability** | HTTP + SSH | SSH only | 🟢 Daemon |
| **Consistency** | Matches queen/worker | Different pattern | 🟢 Daemon |
| **Debugging** | Live inspection | No inspection | 🟢 Daemon |
| **Deployment** | Service management | Just binary | 🟡 CLI |
| **Testing** | HTTP mocks | SSH mocks | 🟢 Daemon |

**Score:** Daemon 7, CLI 3

---

## Critical Decision Factors

### 1. Performance (CRITICAL)

**Question:** Is 10-100x slower acceptable?

**Current (Daemon):**
- HTTP request: 1-5ms
- Batch 10 operations: ~1-2 seconds (parallel)

**Proposed (CLI):**
- SSH exec: 80-350ms
- Batch 10 operations: ~8-35 seconds (sequential)

**Verdict:** 🔴 **BLOCKER** - 10-100x slower is NOT acceptable

---

### 2. User Experience (CRITICAL)

**Question:** Can we live without real-time progress?

**Current (Daemon):**
```
🔍 Detecting GPUs...
✅ Found 2 GPUs
🔧 Spawning worker...
✅ Worker started
```

**Proposed (CLI):**
```
... (waiting 5-10 seconds) ...
Worker started
```

**Verdict:** 🔴 **BLOCKER** - Real-time feedback is essential for UX

---

### 3. Security (CRITICAL)

**Question:** How do we prevent command injection?

**Current (Daemon):**
```rust
// Structured JSON - safe
let operation = Operation::WorkerSpawn { model, device };
```

**Proposed (CLI):**
```rust
// String interpolation - DANGEROUS
let cmd = format!("rbee-hive worker spawn --model {}", model);
// What if model = "test; rm -rf /"?
```

**Verdict:** 🔴 **BLOCKER** - Command injection is a serious security risk

---

### 4. Architecture Consistency (HIGH)

**Question:** Is inconsistency worth the simplicity?

**Current (Daemon):**
- keeper → queen: HTTP + job-server
- queen → hive: HTTP + job-server
- queen → worker: HTTP (simple)
- **Consistent pattern**

**Proposed (CLI):**
- keeper → queen: HTTP + job-server
- queen → hive: SSH + CLI
- queen → worker: HTTP (simple)
- **Inconsistent pattern**

**Verdict:** 🟡 **CONCERN** - Inconsistency adds complexity

---

### 5. Resource Efficiency (MEDIUM)

**Question:** Is 50-100 MB per machine worth saving?

**Current (Daemon):**
- 50-100 MB RAM always
- Background tasks

**Proposed (CLI):**
- 0 MB when idle
- No background tasks

**Verdict:** 🟢 **BENEFIT** - But not worth the tradeoffs

---

## Hybrid Approach Analysis

### Option 3: Daemon with On-Demand Start

**Concept:** Hive daemon that auto-starts on first request and auto-stops after idle period

**Benefits:**
- ✅ Fast when running (HTTP)
- ✅ No resources when idle
- ✅ Real-time streaming
- ✅ Consistent architecture

**Drawbacks:**
- ⚠️ More complexity (auto-start/stop logic)
- ⚠️ Startup delay on first request
- ⚠️ Still need daemon management

**Verdict:** 🟡 **INTERESTING** - Best of both worlds?

---

## Usage Pattern Analysis

### How Often Are Hive Operations Called?

**Frequency:**
- Worker spawn: Once per model (rare)
- Worker stop: Rarely (cleanup)
- Worker list: Occasionally (debugging)
- Model download: Once per model (rare)

**Reality:** Operations are INFREQUENT

**But:**
- When they happen, users want FAST response
- When they happen, users want REAL-TIME feedback
- Frequency doesn't matter if UX is bad

**Verdict:** Low frequency doesn't justify slow operations

---

## Alternative: Simplify Daemon

### What if we keep daemon but simplify it?

**Current Complexity:**
- HTTP server
- Job-server pattern
- Hive heartbeat
- Worker heartbeat aggregation
- State management

**Simplified Daemon:**
- HTTP server (keep)
- Simple request/response (no job-server)
- No hive heartbeat (remove)
- Workers heartbeat directly to queen (change)
- Minimal state (simplify)

**Benefits:**
- ✅ Keep fast HTTP
- ✅ Keep real-time streaming
- ✅ Remove hive heartbeat
- ✅ Simplify state management
- ✅ Still consistent architecture

**Verdict:** 🟢 **BETTER OPTION** - Simplify, don't eliminate

---

## Recommendation

### Keep Daemon, But Simplify

**Changes to Make:**

1. **Remove Hive Heartbeat** ✅
   - Workers heartbeat directly to queen
   - Queen is source of truth
   - No aggregation needed

2. **Simplify State Management** ✅
   - No in-memory worker registry in hive
   - Query processes on-demand
   - Less state to manage

3. **Keep HTTP + job-server** ✅
   - Fast operations (1-5ms)
   - Real-time streaming
   - Consistent architecture

4. **Optional: Auto-start/stop** 🤔
   - Start on first request
   - Stop after idle period
   - Best of both worlds

**Result:**
- ✅ Fast operations
- ✅ Real-time UX
- ✅ Simpler than current
- ✅ Consistent architecture
- ✅ No hive heartbeat
- ✅ Workers → queen direct

---

## Decision Tree

```
Should hive be a CLI?
│
├─ Is 10-100x slower acceptable?
│  ├─ YES → Continue
│  └─ NO → ❌ Keep daemon
│
├─ Can we live without real-time streaming?
│  ├─ YES → Continue
│  └─ NO → ❌ Keep daemon
│
├─ Can we prevent command injection?
│  ├─ YES → Continue
│  └─ NO → ❌ Keep daemon
│
└─ Is inconsistency worth it?
   ├─ YES → ✅ Make CLI
   └─ NO → ❌ Keep daemon
```

**Our Answer:** NO at step 1 (performance blocker)

---

## Final Verdict

### 🔴 DO NOT PIVOT TO CLI

**Reasons:**
1. ❌ **Performance:** 10-100x slower is unacceptable
2. ❌ **UX:** No real-time streaming hurts user experience
3. ❌ **Security:** Command injection risk is serious
4. ❌ **Consistency:** Breaks architectural patterns

**Instead:**
1. ✅ **Keep daemon** for fast operations
2. ✅ **Remove hive heartbeat** (workers → queen direct)
3. ✅ **Simplify state** (less in-memory state)
4. ✅ **Consider auto-start/stop** (optional optimization)

---

## Implementation Plan (Simplified Daemon)

### Phase 1: Remove Hive Heartbeat
- Remove hive heartbeat task from main.rs
- Remove hive heartbeat endpoint from queen
- Update workers to send heartbeats to queen directly
- Update queen to track workers directly

### Phase 2: Simplify State
- Remove in-memory worker registry from hive
- Query processes on-demand for worker list
- Less state synchronization

### Phase 3: Optional Auto-Start/Stop
- Add idle timeout to hive daemon
- Auto-stop after 5 minutes idle
- Queen auto-starts hive on first request
- Best of both worlds

---

## Metrics to Track

If we implement simplified daemon:

**Before (Current):**
- Hive memory: 50-100 MB
- Hive CPU: ~1-2%
- Operation latency: 1-5ms
- Hive heartbeat: Every 5s

**After (Simplified):**
- Hive memory: 30-50 MB (less state)
- Hive CPU: ~0.5-1% (no heartbeat)
- Operation latency: 1-5ms (same)
- Hive heartbeat: None

**After (Auto-Start/Stop):**
- Hive memory: 0 MB when idle
- Hive CPU: 0% when idle
- Operation latency: 1-5ms (+ startup delay first time)
- Hive heartbeat: None

---

## Conclusion

**The pivot to CLI is NOT recommended.**

**Why?**
- Performance degradation is unacceptable
- Real-time UX is essential
- Security risks are serious
- Architectural inconsistency adds complexity

**Instead:**
- Keep daemon for performance
- Remove hive heartbeat for simplicity
- Simplify state management
- Consider auto-start/stop for resource efficiency

**Best of both worlds:**
- Fast operations (daemon)
- No wasted resources (auto-stop)
- Real-time streaming (HTTP)
- Simpler architecture (no hive heartbeat)

---

**TEAM-261 Pivot Decision**  
**Date:** Oct 23, 2025  
**Verdict:** 🔴 DO NOT PIVOT - Keep simplified daemon  
**Next Steps:** Remove hive heartbeat, simplify state
