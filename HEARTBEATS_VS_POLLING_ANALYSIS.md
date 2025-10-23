# Heartbeats vs Health Polling: Architectural Analysis

**Date:** Oct 23, 2025  
**Context:** Monitoring pattern for worker → queen communication

## Executive Summary

**Current System**: Workers send heartbeats to queen (PUSH model)  
**Proposed in improvements**: Queen polls workers (PULL model)  
**Recommendation**: **Keep heartbeats for workers, use polling for daemons**

## Pattern Definitions

### Heartbeat (Push Model)
**Child reports to parent periodically**
```
Worker ──[heartbeat]──> Queen
Worker ──[heartbeat]──> Queen
Worker ──[heartbeat]──> Queen
```

### Health Polling (Pull Model)
**Parent checks child periodically**
```
Queen ──[health check]──> Worker
Queen <──[response]────── Worker
```

## Detailed Comparison

### ⚡ Heartbeat (Push) - Current System

#### ✅ Pros

1. **Scalability** ⭐⭐⭐
   - Each worker independently sends heartbeats
   - Queen is passive receiver (lower CPU)
   - Easy to scale to 100s of workers
   - No "thundering herd" on queen

2. **Real-time Reporting** ⭐⭐⭐
   - Worker can report immediately when state changes
   - Can include rich metadata (GPU usage, queue depth, etc.)
   - Can report errors immediately
   - No delay waiting for next poll

3. **Network Efficiency** ⭐⭐
   - Workers know their own state
   - Only send updates when needed
   - Can batch multiple metrics in one heartbeat
   - Can adjust frequency based on activity

4. **Decentralized Control** ⭐⭐⭐
   - Workers are autonomous
   - Queen doesn't need to know all worker endpoints
   - Workers can change ports without queen knowing
   - Easy to add new workers (just start sending heartbeats)

5. **Failure Detection** ⭐⭐
   - Missing heartbeat = worker likely dead
   - Simple timeout logic: `last_seen > threshold`
   - Queen can mark worker as "stale" after N missed heartbeats

6. **Resource Reporting** ⭐⭐⭐
   - Workers can report detailed metrics
   - GPU utilization, memory, queue depth
   - Current model loaded, inference count
   - All in a single heartbeat payload

#### ❌ Cons

1. **Missed Heartbeats Ambiguity** ⚠️⚠️
   - Network issue? Worker dead? Worker busy?
   - Can't distinguish between failure modes
   - False positives if network is congested

2. **Delayed Failure Detection** ⚠️
   - Queen only knows worker is dead after timeout
   - If heartbeat interval = 30s, detection delay = 30-60s
   - Can't detect issues faster than heartbeat interval

3. **Clock Skew Issues** ⚠️
   - Worker and queen clocks must be roughly synchronized
   - Incorrect timestamps can cause false alarms
   - Need NTP or similar

4. **Zombie Workers** ⚠️⚠️
   - Worker process may be alive but stuck/unresponsive
   - Still sends heartbeats but can't do work
   - Hard to detect without actual health checks

5. **No Verification** ⚠️⚠️
   - Queen trusts whatever worker reports
   - Can't verify worker is actually healthy
   - Worker might report "healthy" but be serving 500s

6. **State Drift** ⚠️
   - Queen's view of worker is "last heartbeat"
   - Not real-time, always slightly stale
   - Race conditions if state changes between heartbeats

#### 📊 Best For
- Many workers (10s to 100s)
- Workers with varying lifetimes
- Rich state reporting needed
- Autonomous worker processes
- **Current use case: Worker → Queen ✅**

---

### 🔄 Health Polling (Pull Model)

#### ✅ Pros

1. **Active Verification** ⭐⭐⭐
   - Queen confirms worker is responsive
   - Can verify worker is actually serving requests
   - No ambiguity - either responds or doesn't
   - Detects zombie processes

2. **Immediate Failure Detection** ⭐⭐
   - Know instantly if worker is down (within poll interval)
   - Can retry immediately on failure
   - No waiting for missed heartbeat timeout

3. **No Clock Sync Required** ⭐
   - Only need to track "is up/down now"
   - No timestamp comparison issues
   - Simpler logic

4. **Centralized Control** ⭐⭐
   - Queen controls monitoring frequency
   - Can adjust polling based on load
   - Can prioritize critical workers

5. **Easy Debugging** ⭐⭐
   - Can test worker health manually (curl)
   - Clear request/response pattern
   - Easy to add health check details

6. **Startup Detection** ⭐⭐⭐
   - Can poll until worker is ready
   - Useful for daemon startup
   - Know exactly when worker is accepting requests

#### ❌ Cons

1. **Scalability Issues** ⚠️⚠️⚠️
   - Queen must poll every worker
   - O(n) network calls per poll interval
   - 100 workers × 10s interval = 10 req/sec just for health
   - CPU/network load on queen increases linearly

2. **Thundering Herd** ⚠️⚠️
   - All polls happen at same time
   - Network spike every poll interval
   - Can overwhelm queen or network

3. **Worker Overhead** ⚠️⚠️
   - Every worker must implement health endpoint
   - Must respond to health checks while processing
   - Can interfere with actual work
   - Extra HTTP request handling

4. **Limited Information** ⚠️
   - Health check only returns "up/down"
   - To get metrics, need separate calls
   - Can't report state changes between polls

5. **Delayed State Updates** ⚠️⚠️
   - State only known at poll time
   - Worker might fail immediately after poll
   - Queen's view is always stale (by up to poll interval)

6. **Network Chattiness** ⚠️⚠️
   - Constant network traffic even when idle
   - Wastes bandwidth
   - Can't reduce frequency without delaying detection

7. **Configuration Complexity** ⚠️
   - Queen needs list of all worker endpoints
   - Must update when workers start/stop
   - Port management becomes complex

#### 📊 Best For
- Few daemons (1-10)
- Fixed/known endpoints
- Startup synchronization
- Simple up/down status
- **Use case: Queen startup, Hive startup ✅**

---

## Comparison Matrix

| Aspect | Heartbeat (Push) | Health Polling (Pull) | Winner |
|--------|------------------|----------------------|--------|
| **Scalability** | Excellent (100s workers) | Poor (10s workers) | 💚 Heartbeat |
| **Failure Detection** | Delayed (30-60s) | Immediate (poll interval) | 💙 Polling |
| **Zombie Detection** | Poor (trusts worker) | Good (verifies response) | 💙 Polling |
| **Network Efficiency** | Good (event-driven) | Poor (constant polling) | 💚 Heartbeat |
| **Queen CPU Usage** | Low (passive) | High (active polling) | 💚 Heartbeat |
| **State Richness** | Excellent (any data) | Limited (up/down) | 💚 Heartbeat |
| **Startup Detection** | Poor (wait for first HB) | Excellent (poll until ready) | 💙 Polling |
| **Configuration** | Simple (workers self-register) | Complex (track all endpoints) | 💚 Heartbeat |
| **Debugging** | Hard (passive) | Easy (active test) | 💙 Polling |
| **False Positives** | Higher (network issues) | Lower (verify response) | 💙 Polling |

---

## Hybrid Approach: Best of Both Worlds

### Recommendation: **Context-Specific Pattern**

#### Use Heartbeats For: Workers (Current ✅)
```rust
// Workers send heartbeats to queen
POST /v1/worker/heartbeat
{
    "worker_id": "worker-123",
    "status": "idle",
    "gpu_utilization": 0.45,
    "model_loaded": "meta-llama/Llama-3-8b",
    "queue_depth": 0,
    "last_inference_ms": 1234,
    "timestamp": "2025-10-23T20:00:00Z"
}
```

**Why:**
- ✅ Many workers (10-100+)
- ✅ Rich state reporting needed
- ✅ Workers are autonomous
- ✅ Scales well
- ✅ Network efficient

#### Use Health Polling For: Daemon Startup
```rust
// Queen polls until hive is ready (startup only)
async fn wait_for_daemon_ready(url: &str) -> Result<()> {
    poll_until_healthy(HealthPollConfig {
        url,
        max_attempts: 10,
        initial_delay_ms: 200,
        backoff_multiplier: 1.5,
    }).await
}
```

**Why:**
- ✅ One-time operation (startup)
- ✅ Need to know when daemon is accepting requests
- ✅ No scalability concerns (1-2 daemons)
- ✅ Critical for orchestration

---

## Current Architecture Analysis

### What You Have Now (Correct! ✅)

```
Worker-123 ──[heartbeat every 30s]──> Queen
Worker-456 ──[heartbeat every 30s]──> Queen
Worker-789 ──[heartbeat every 30s]──> Queen

Queen maintains registry:
{
  "worker-123": { last_seen: 10s ago, status: "idle", gpu: 0.0 },
  "worker-456": { last_seen: 5s ago, status: "busy", gpu: 0.9 },
  "worker-789": { last_seen: 45s ago, status: "stale" }  // Mark as stale!
}
```

**This is the RIGHT pattern for worker monitoring!**

### What I Was Suggesting (Wrong for workers! ❌)

```
Queen ──[GET /health]──> Worker-123
Queen ──[GET /health]──> Worker-456
Queen ──[GET /health]──> Worker-789
(repeat every 10s)
```

**This would NOT scale well for 100 workers!**

### What Health Polling IS Good For (Correct! ✅)

```
// Hive startup in hive-lifecycle
async fn execute_hive_start() {
    spawn_hive_daemon().await?;
    
    // Poll until ready (one-time)
    poll_until_healthy(HealthPollConfig {
        url: hive_endpoint,
        max_attempts: 10,
        ...
    }).await?;
    
    // Now hive is ready to receive requests
}
```

**This IS appropriate - single daemon, startup synchronization!**

---

## Recommendations

### 1. Keep Current Heartbeat System ✅

**For Workers → Queen:**
- ✅ Continue using heartbeat pattern
- ✅ Workers POST to `/v1/worker/heartbeat`
- ✅ Include rich metrics (GPU, status, model, queue)
- ✅ Queen marks stale after timeout (e.g., 90s)

### 2. Add Health Polling for Daemon Startup ✅

**For Queen startup, Hive startup:**
```rust
// In daemon-lifecycle crate
pub async fn poll_until_healthy(config: HealthPollConfig) -> Result<()>
```

**Use ONLY for:**
- Queen startup (wait until accepting requests)
- Hive startup (wait until accepting requests)
- One-time daemon initialization

**Do NOT use for:**
- Ongoing worker monitoring (use heartbeats!)
- Periodic health checks (use heartbeats!)

### 3. Consider Hybrid for Worker Verification (Optional)

**On-demand health checks:**
```rust
// Queen can manually verify a specific worker if suspicious
async fn verify_worker_health(worker_id: &str) -> Result<()> {
    let endpoint = registry.get_worker_endpoint(worker_id)?;
    is_daemon_healthy(&endpoint, None, Some(Duration::from_secs(5))).await
}
```

**Use cases:**
- Before routing inference request
- After multiple failed heartbeats
- Manual debugging/testing

---

## Implementation Guidance

### ✅ Keep in daemon-lifecycle

```rust
// For daemon startup only
pub async fn poll_until_healthy(config: HealthPollConfig) -> Result<()> {
    // Exponential backoff polling
    // Use for queen/hive startup
}
```

### ✅ Keep in worker-lifecycle

```rust
// Workers send heartbeats (don't poll!)
pub async fn send_heartbeat(queen_url: &str, heartbeat: WorkerHeartbeat) -> Result<()>
```

### ✅ Keep in queen

```rust
// Queen maintains worker registry from heartbeats
pub struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerState>,
}

impl WorkerRegistry {
    pub fn update_from_heartbeat(&mut self, hb: WorkerHeartbeat) {
        self.workers.insert(hb.worker_id, WorkerState {
            last_seen: Instant::now(),
            status: hb.status,
            metrics: hb.metrics,
        });
    }
    
    pub fn get_stale_workers(&self, timeout: Duration) -> Vec<WorkerId> {
        self.workers.iter()
            .filter(|(_, state)| state.last_seen.elapsed() > timeout)
            .map(|(id, _)| id.clone())
            .collect()
    }
}
```

---

## Conclusion

### The Confusion in My Proposals

I was suggesting **health polling** for daemon startup (which IS correct), but the name "health polling" implied it should be used for ongoing monitoring (which is WRONG for workers).

### Correct Pattern by Use Case

| Use Case | Pattern | Reasoning |
|----------|---------|-----------|
| **Worker monitoring** | 💚 Heartbeat | Scalable, rich data, many workers |
| **Daemon startup** | 💙 Poll once | Synchronization, few daemons, one-time |
| **Manual verification** | 💙 Poll on-demand | Debugging, suspicious worker, rare |

### Your Current Architecture: ✅ CORRECT

**Workers sending heartbeats to queen is the RIGHT pattern!**

My daemon-lifecycle improvements should be used for:
- ✅ Queen startup (poll until ready)
- ✅ Hive startup (poll until ready)
- ✅ Manual worker health verification (on-demand)

But NOT for:
- ❌ Ongoing worker monitoring (keep heartbeats!)
- ❌ Periodic health checks (keep heartbeats!)

### Final Recommendation

**Keep your heartbeat system, add polling ONLY for daemon startup:**

```rust
// daemon-lifecycle: For daemon startup only
pub async fn poll_until_healthy(...) // ← Add this

// worker-lifecycle: Keep heartbeats!
pub async fn send_heartbeat(...) // ← Keep this

// queen: Keep registry + heartbeat processing
WorkerRegistry::update_from_heartbeat(...) // ← Keep this
```

This gives you the best of both worlds!
