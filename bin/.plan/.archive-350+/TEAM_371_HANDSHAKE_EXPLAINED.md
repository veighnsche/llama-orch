# TEAM-371: Why Handshake is ESSENTIAL (Visual Guide)

**Date:** Oct 31, 2025  
**Author:** TEAM-371  

---

## The Confusion

**Original statement (WRONG):** "Remove the handshake and discovery"  
**Correct statement:** "Keep handshake, change telemetry delivery method"

---

## Visual Comparison

### Current Implementation (Mixed)

```
Discovery Phase (Exponential Backoff):
┌──────┐                              ┌───────┐
│ Hive │──POST /v1/hive-heartbeat────▶│ Queen │
└──────┘  (with capabilities)         └───────┘
  │
  │ Attempt 1: 0s → 404 (Queen not ready)
  │ Attempt 2: 2s → 404
  │ Attempt 3: 4s → 200 OK! ✅
  │
  └─ Discovery complete

Continuous Telemetry Phase:
┌──────┐                              ┌───────┐
│ Hive │──POST /v1/hive-heartbeat────▶│ Queen │
└──────┘  (with workers, every 1s)    └───────┘
  ▲                                      │
  │                                      │
  └──────────── Forever ─────────────────┘

PROBLEM: Same endpoint for discovery AND telemetry
```

### Proposed Implementation (Separated)

```
Discovery Phase (Exponential Backoff):
┌──────┐                              ┌───────┐
│ Hive │──POST /v1/hive/ready────────▶│ Queen │
└──────┘  (hive_id + hive_url)        └───────┘
  │                                      │
  │ Attempt 1: 0s → 404                 │
  │ Attempt 2: 2s → 404                 │
  │ Attempt 3: 4s → 200 OK! ✅          │
  │                                      │
  └─ Discovery complete                 │
                                        │
                                        ▼
                                  Subscribe to SSE

Continuous Telemetry Phase:
┌──────┐                              ┌───────┐
│ Hive │──SSE /v1/heartbeats/stream──▶│ Queen │
└──────┘  (workers, every 1s)         └───────┘
  ▲                                      
  │  SSE connection (Queen subscribes)   
  └──────────── Forever ─────────────────

SOLUTION: Discovery callback separate from telemetry stream
```

---

## Why Discovery is REQUIRED

### Problem: Queen Doesn't Know What Hives Exist

**Scenario:** You have 3 machines:
- `hive-gpu-0` at 192.168.1.100
- `hive-gpu-1` at 192.168.1.101  
- `hive-cpu-0` at 192.168.1.102

**Question:** How does Queen know these exist?

### Option 1: Configuration File (Static)

```yaml
# queen-config.yaml
hives:
  - name: hive-gpu-0
    url: http://192.168.1.100:7835
  - name: hive-gpu-1
    url: http://192.168.1.101:7835
  - name: hive-cpu-0
    url: http://192.168.1.102:7835
```

**Problems:**
- ❌ Manual configuration required
- ❌ Must update config when adding/removing hives
- ❌ No dynamic discovery
- ❌ Doesn't handle bidirectional startup

### Option 2: SSH Config Reading (Dynamic)

```bash
# ~/.ssh/config
Host hive-gpu-0
  HostName 192.168.1.100
  
Host hive-gpu-1
  HostName 192.168.1.101
  
Host hive-cpu-0
  HostName 192.168.1.102
```

**Queen reads this and knows WHERE to look, but:**
- ❓ Is hive running on that machine?
- ❓ Which port is it on?
- ❓ Is rbee-hive installed?

**Queen needs to DISCOVER, not just guess.**

### Option 3: Handshake (Current/Proposed)

**Bidirectional discovery:**

**If Queen starts first:**
```
1. Queen reads SSH config
2. Queen tries GET /capabilities on each host
3. Hive responds "I'm here!" + capabilities
4. Hive sends POST /v1/hive/ready (callback)
5. Queen subscribes to SSE
```

**If Hive starts first:**
```
1. Hive has --queen-url configured
2. Hive sends POST /v1/hive/ready (exponential backoff)
3. Queen receives "I'm here!" callback
4. Queen subscribes to SSE
```

**Benefits:**
- ✅ Works regardless of startup order
- ✅ Auto-discovery (no manual config)
- ✅ Resilient to restarts
- ✅ Hives announce themselves

---

## The Callback is NOT Telemetry

### Callback Purpose: "I exist, connect to me"

```json
POST /v1/hive/ready
{
  "hive_id": "hive-gpu-0",
  "hive_url": "http://192.168.1.100:7835"
}
```

**Frequency:** ONE-TIME (or retry with exponential backoff until success)

**Payload:** Minimal - just identity and location

**Purpose:** Discovery/registration

### SSE Purpose: Continuous telemetry

```
GET /v1/heartbeats/stream

event: heartbeat
data: {"type":"telemetry","hive_id":"hive-gpu-0","workers":[...]}

event: heartbeat
data: {"type":"telemetry","hive_id":"hive-gpu-0","workers":[...]}

event: heartbeat
data: {"type":"telemetry","hive_id":"hive-gpu-0","workers":[...]}
```

**Frequency:** CONTINUOUS (every 1 second)

**Payload:** Full telemetry - workers, GPU stats, etc.

**Purpose:** Real-time monitoring

---

## Sequence Diagrams

### Scenario 1: Queen Starts First

```
Time │ Hive                  │ Queen
─────┼───────────────────────┼─────────────────────────
0s   │ (not started)         │ Queen starts
     │                       │ Reads SSH config
     │                       │ Finds: hive-gpu-0, hive-gpu-1
─────┼───────────────────────┼─────────────────────────
5s   │                       │ GET /capabilities?queen_url=X
     │                       │   → hive-gpu-0 (timeout)
     │                       │   → hive-gpu-1 (timeout)
─────┼───────────────────────┼─────────────────────────
10s  │ Hive starts           │
     │ Args: --queen-url ... │
     │                       │
─────┼───────────────────────┼─────────────────────────
11s  │ POST /v1/hive/ready ──┼──▶ Receives callback
     │   (attempt 1: 0s)     │    "hive-gpu-0 is ready"
     │                       │    Subscribe to SSE:
     │                       │    GET /v1/heartbeats/stream
     │ ◀────────────────────┼─── 200 OK
─────┼───────────────────────┼─────────────────────────
12s  │ SSE: telemetry ───────┼──▶ Receives workers
13s  │ SSE: telemetry ───────┼──▶ Receives workers
14s  │ SSE: telemetry ───────┼──▶ Receives workers
     │        ...            │         ...
```

### Scenario 2: Hive Starts First

```
Time │ Hive                  │ Queen
─────┼───────────────────────┼─────────────────────────
0s   │ Hive starts           │ (not started)
     │ Args: --queen-url ... │
─────┼───────────────────────┼─────────────────────────
1s   │ POST /v1/hive/ready ──┼──▶ (connection refused)
     │   (attempt 1: 0s)     │
─────┼───────────────────────┼─────────────────────────
3s   │ POST /v1/hive/ready ──┼──▶ (connection refused)
     │   (attempt 2: 2s)     │
─────┼───────────────────────┼─────────────────────────
5s   │                       │ Queen starts
─────┼───────────────────────┼─────────────────────────
7s   │ POST /v1/hive/ready ──┼──▶ Receives callback
     │   (attempt 3: 4s)     │    "hive-gpu-0 is ready"
     │                       │    Subscribe to SSE:
     │                       │    GET /v1/heartbeats/stream
     │ ◀────────────────────┼─── 200 OK
─────┼───────────────────────┼─────────────────────────
8s   │ SSE: telemetry ───────┼──▶ Receives workers
9s   │ SSE: telemetry ───────┼──▶ Receives workers
10s  │ SSE: telemetry ───────┼──▶ Receives workers
     │        ...            │         ...
```

---

## What Happens on Restart

### Queen Restarts

```
Before:
  Hive ──SSE──▶ Queen ──SSE──▶ UI
               (connected)

Queen crashes:
  Hive ──SSE──✗ Queen (dead)
               (connection lost)

Hive detects:
  POST /v1/hive/ready → 404/refused
  "Queen is down, start exponential backoff"

Queen recovers:
  Hive ──POST /v1/hive/ready──▶ Queen
     ◀────────200 OK───────────

Queen subscribes:
  Hive ──SSE──▶ Queen (reconnected)
```

### Hive Restarts

```
Before:
  Hive ──SSE──▶ Queen ──SSE──▶ UI
               (connected)

Hive crashes:
  Hive (dead) ✗ Queen
               (SSE connection closes)

Queen detects:
  "SSE connection lost for hive-gpu-0"
  "Waiting for new callback..."

Hive recovers:
  Hive ──POST /v1/hive/ready──▶ Queen
     ◀────────200 OK───────────

Queen subscribes:
  Hive ──SSE──▶ Queen (reconnected)
```

**Key insight:** SSE connection failure is automatically detected. Callback re-establishes the connection.

---

## Why NOT Use Queen-Initiated SSE?

**Question:** Why doesn't Queen just try to connect to SSE without the callback?

**Answer:** Because Queen doesn't know:
1. Which hives exist
2. Which ports they're on
3. If they're running

**Example failure scenario:**

```bash
# Queen tries to blindly connect
curl -N http://192.168.1.100:7835/v1/heartbeats/stream
# Connection refused (hive not running)

curl -N http://192.168.1.101:7835/v1/heartbeats/stream
# Timeout (machine offline)

curl -N http://192.168.1.102:7835/v1/heartbeats/stream
# 404 (wrong port, hive is on 7836)
```

**The callback tells Queen:** "I'm at this exact URL, ready to stream."

---

## Summary: Two Separate Mechanisms

### 1. Discovery (Handshake)

**Purpose:** "I exist, here's where to find me"  
**Method:** POST callback  
**Frequency:** ONE-TIME (with exponential backoff retry)  
**Payload:** Identity + location  
**Required for:** Finding hives dynamically

### 2. Telemetry (Streaming)

**Purpose:** "Here's what's happening right now"  
**Method:** SSE stream  
**Frequency:** CONTINUOUS (every 1s)  
**Payload:** Workers, GPU stats, etc.  
**Required for:** Real-time monitoring

**They are NOT interchangeable. Both are essential.**

---

## Final Answer to User's Question

> "Why do we need to get rid of the handshake? What is the replacement?"

**WE DON'T GET RID OF THE HANDSHAKE.**

**What we change:**
- Discovery callback: Still exists, still required
- Continuous telemetry: Changes from POST to SSE

**The handshake IS the discovery. SSE IS the telemetry. They are different things.**

**Before:** Used POST for BOTH discovery AND telemetry (mixed)  
**After:** Use POST for discovery, SSE for telemetry (separated)

**The handshake is preserved because discovery is essential.**

---

**TEAM-371: Handshake explanation complete.**
