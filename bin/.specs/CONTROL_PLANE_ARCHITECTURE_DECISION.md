# Control Plane Architecture Decision: SSH vs HTTP

**Status**: Decision Required  
**Date**: 2025-10-09  
**Context**: How should orchestrator control pool managers?

---

## The Question

**Current confusion:**
- Specs define HTTP APIs for orchestrator ↔ pool-managerd (SYS-5.2.x)
- But we have `llama-orch-sdk` as the HTTP client
- CLI should be for control, not inference
- Do we need HTTP for control plane, or just SSH?

**Options:**
1. **HTTP for everything** - Orchestrator → Pool Manager via HTTP
2. **SSH for control, HTTP for inference** - Control via SSH, inference via HTTP
3. **SSH only** - No HTTP between orchestrator and pool managers

---

## Current Spec (SYS-5.2.x)

From `bin/.specs/00_llama-orch.md`:

```
[Orchestrator] (lab-controller:8080)
      ↓ (HTTP)
[Pool Manager] (gpu-node-1:9200)
      ├─→ [Worker-1] GPU 0
      └─→ [Worker-2] GPU 1
```

**Requirements:**
- SYS-3.2.1: Authentication MUST be enabled for orchestrator ↔ pool-managerd
- SYS-3.2.1: Pool-managerd MUST authenticate using bearer tokens or mTLS
- SYS-5.2.x: HTTP API for orchestrator → pool manager commands

**But also:**
- SYS-3.1.1: Home Mode uses localhost (no auth needed)
- Your homelab is trusted network (mac, workstation, blep)

---

## Analysis

### What llama-orch-sdk Is

**Purpose:** Client library for INFERENCE API
```rust
// llama-orch-sdk usage (from consumers/)
let client = Client::new("http://localhost:8080");
let response = client.enqueue(EnqueueRequest {
    prompt: "Hello",
    model: "llama3",
    max_tokens: 100,
}).await?;

// Stream inference results
let mut stream = client.enqueue_stream(request).await?;
while let Some(event) = stream.next().await {
    println!("{}", event.text);
}
```

**Used by:**
- Web applications
- Node.js apps
- Rust applications
- Browsers (WASM)

**NOT used by:**
- Orchestrator (doesn't call itself)
- Pool managers (don't submit inference jobs)
- CLI tools (different purpose)

### What CLI Tools Are

**Purpose:** CONTROL PLANE operations
```bash
# rbees-ctl (orchestrator control)
llorch pool models download tinyllama --host mac
llorch pool worker spawn metal --host mac
llorch pool status --host workstation

# rbees-pool (pool manager control)
rbees-pool models download tinyllama
rbees-pool worker spawn metal --model tinyllama
rbees-pool git pull
```

**Used by:**
- Operators (humans)
- CI/CD scripts
- Automation tools

**NOT used by:**
- End users (they use web UI)
- Applications (they use llama-orch-sdk)

---

## Option 1: HTTP for Everything (Current Spec)

### Architecture

```
rbees-ctl (CLI)
    ↓ HTTP
rbees-orcd (daemon :8080)
    ↓ HTTP
pool-managerd (daemon :9200)
    ↓ spawn
rbees-workerd (worker)
```

### Pros

**✅ Spec Compliance**
- Follows SYS-5.2.x exactly
- HTTP APIs defined in OpenAPI
- Clear contracts

**✅ Uniform Interface**
- Everything is HTTP
- Same authentication mechanism
- Same error handling

**✅ Language Agnostic**
- Any language can control pools
- Easy to build alternative tools
- OpenAPI spec is contract

### Cons

**❌ Complexity**
- Need HTTP server on pool managers
- Need authentication (bearer tokens or mTLS)
- Need TLS for security
- More moving parts

**❌ Security Surface**
- HTTP port open on pool managers (:9200)
- Need to secure with auth tokens
- Need to rotate tokens
- Attack surface for pool managers

**❌ Overkill for Homelab**
- Your homelab is trusted network
- SSH already works
- SSH keys already configured
- Adding HTTP is extra complexity

**❌ Daemon Required**
- Pool manager must run as daemon
- Can't do simple operations without daemon
- More processes to manage

---

## Option 2: SSH for Control, HTTP for Inference (Hybrid)

### Architecture

```
rbees-ctl (CLI)
    ↓ SSH
rbees-pool (on pool host)
    ↓ direct execution
hf download, git pull, rbees-workerd spawn

rbees-orcd (daemon, M2+)
    ↓ HTTP (inference only)
rbees-workerd (worker)
    ↑ HTTP (inference only)
llama-orch-sdk (client)
```

### Pros

**✅ Separation of Concerns**
- Control plane: SSH (secure, proven)
- Data plane: HTTP (streaming, efficient)
- Clear boundaries

**✅ Security**
- SSH public/private keys (already configured)
- No HTTP port on pool managers for control
- Smaller attack surface
- SSH is battle-tested

**✅ Simplicity**
- No pool-managerd daemon needed for M0
- Direct execution on pool hosts
- No authentication to configure
- No TLS to setup

**✅ Homelab-Friendly**
- SSH already works (mac.home.arpa, workstation.home.arpa)
- No new ports to open
- No new services to manage
- Trusted network assumption

**✅ Progressive Enhancement**
- M0: SSH only (simple)
- M1: Add pool-managerd daemon (optional)
- M2: Add rbees-orcd daemon (optional)
- Can mix modes (some pools SSH, some HTTP)

### Cons

**❌ Spec Deviation**
- SYS-5.2.x defines HTTP API
- Would need to update specs
- Two communication protocols

**❌ SSH Dependency**
- Requires SSH access
- Requires SSH keys
- SSH can be slow for high-frequency calls

**❌ Limited Programmatic Access**
- Can't easily call from other languages
- SSH is harder to automate than HTTP
- No OpenAPI spec for control plane

---

## Option 3: SSH Only (No HTTP for Control)

### Architecture

```
rbees-ctl (CLI)
    ↓ SSH
rbees-pool (on pool host)
    ↓ direct execution
Everything (models, git, workers)

NO rbees-orcd daemon
NO pool-managerd daemon
Just CLIs + SSH + rbees-workerd workers
```

### Pros

**✅ Maximum Simplicity**
- No daemons needed (except workers)
- No HTTP servers for control
- No authentication to configure
- Just SSH keys

**✅ Security**
- SSH only (proven, secure)
- No HTTP ports exposed
- Minimal attack surface

**✅ Homelab Perfect**
- Exactly what you have now (bash scripts)
- SSH already configured
- Trusted network

### Cons

**❌ Major Spec Deviation**
- Specs define rbees-orcd/pool-managerd as daemons
- Would need major spec rewrite
- Loses future scalability

**❌ No State Persistence**
- No job queue persistence
- No worker registry
- No metrics aggregation

**❌ No Scalability**
- Can't scale to multi-tenant
- Can't add web UI easily
- Can't build marketplace (M5)

**❌ SSH Limitations**
- High latency for frequent calls
- No streaming (for heartbeats)
- No pub/sub patterns

---

## Recommendation: Option 2 (SSH for Control, HTTP for Inference)

### Rationale

**For M0 (Homelab):**
- Use SSH for control plane (rbees-ctl → rbees-pool)
- No pool-managerd daemon needed yet
- Simple, secure, works with existing SSH setup
- Direct execution on pool hosts

**For M2+ (Production):**
- Add HTTP APIs for control plane
- Add rbees-orcd daemon (HTTP server)
- Add pool-managerd daemon (HTTP server)
- Keep SSH as fallback/alternative

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ CONTROL PLANE (Operator → System)                               │
├─────────────────────────────────────────────────────────────────┤
│ M0: SSH-based                                                    │
│   rbees-ctl → SSH → rbees-pool → direct execution                │
│                                                                   │
│ M2+: HTTP-based (optional)                                       │
│   rbees-ctl → HTTP → rbees-orcd → HTTP → pool-managerd      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DATA PLANE (Client → Inference)                                 │
├─────────────────────────────────────────────────────────────────┤
│ Always HTTP-based                                                │
│   llama-orch-sdk → HTTP → rbees-orcd → HTTP → worker         │
│   (SSE streaming for tokens)                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

**M0: SSH Control Plane**
```bash
# Orchestrator CLI (on blep)
llorch pool models download tinyllama --host mac
  → ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool models download tinyllama"
  → hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...

llorch pool worker spawn metal --host mac --model tinyllama
  → ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool worker spawn metal --model tinyllama"
  → rbees-workerd --backend metal --model .test-models/tinyllama/... &
```

**M2+: HTTP Control Plane (Optional)**
```bash
# Start daemons
rbees-orcd --config orchestrator.toml  # HTTP server :8080
pool-managerd --config pool.toml          # HTTP server :9200

# CLI calls HTTP APIs
llorch pool worker spawn metal --host mac --model tinyllama
  → HTTP: POST http://localhost:8080/pools/mac/workers/spawn
  → rbees-orcd: POST http://mac.home.arpa:9200/workers/spawn
  → pool-managerd spawns worker
```

**Inference (Always HTTP):**
```rust
// Applications use llama-orch-sdk
let client = Client::new("http://localhost:8080");
let response = client.enqueue(request).await?;
```

---

## Spec Updates Required

### Update SYS-5.2.x

**Current:**
> Orchestrator ↔ Pool Manager communication MUST use HTTP

**Proposed:**
> Orchestrator ↔ Pool Manager communication:
> - M0 (Home Mode): MAY use SSH for control operations
> - M2+ (Lab/Platform Mode): MUST use HTTP with authentication
> - Inference requests: MUST always use HTTP (via llama-orch-sdk)

### Add SYS-5.8.x: SSH Control Protocol

```
[SYS-5.8.1] SSH Control Protocol (M0)

In Home Mode, orchestrator MAY control pool managers via SSH:

**Requirements:**
- SSH MUST use public/private key authentication
- SSH commands MUST execute rbees-pool binary on remote host
- SSH MUST change to repo directory before execution
- SSH MUST stream stdout/stderr back to orchestrator
- SSH MUST preserve exit codes

**Example:**
ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool models download tinyllama"

**Limitations:**
- SSH is for control operations only (not inference)
- SSH is for trusted networks only (homelab)
- SSH does not support streaming inference results
- SSH does not support pub/sub patterns (heartbeats)
```

---

## Benefits of SSH Control Plane

### Security

**SSH advantages:**
- ✅ Public/private key authentication (already configured)
- ✅ No passwords (key-based only)
- ✅ No bearer tokens to manage
- ✅ No TLS certificates to configure
- ✅ Battle-tested security (30+ years)
- ✅ Audit trail (SSH logs)

**HTTP disadvantages:**
- ❌ Need to configure bearer tokens
- ❌ Need to rotate tokens
- ❌ Need TLS for security
- ❌ More attack surface (HTTP port open)

### Simplicity

**SSH advantages:**
- ✅ Already configured (mac.home.arpa, workstation.home.arpa)
- ✅ No daemon required on pools (M0)
- ✅ Direct execution (rbees-pool)
- ✅ No port forwarding needed
- ✅ Works through firewalls (SSH port usually open)

**HTTP disadvantages:**
- ❌ Need to run pool-managerd daemon
- ❌ Need to open port :9200
- ❌ Need to configure authentication
- ❌ More processes to manage

### Homelab Fit

**Your setup:**
- Trusted network (home)
- SSH already works
- 3 hosts (blep, mac, workstation)
- Not multi-tenant (yet)

**SSH is perfect for:**
- Trusted networks
- Small number of hosts
- Infrequent control operations
- Human operators

**HTTP is better for:**
- Untrusted networks
- Many hosts (10+)
- High-frequency operations
- Programmatic access

---

## Proposed Architecture

### Control Plane: SSH (M0) → HTTP (M2+)

**M0 (Homelab):**
```
rbees-ctl (blep)
    ↓ SSH (control operations)
rbees-pool (mac/workstation)
    ↓ direct execution
hf download, git pull, rbees-workerd spawn
```

**M2+ (Production):**
```
rbees-ctl (blep)
    ↓ HTTP (control operations)
rbees-orcd (daemon :8080)
    ↓ HTTP (control operations)
pool-managerd (daemon :9200)
    ↓ spawn
rbees-workerd (worker)
```

### Data Plane: Always HTTP

**All modes:**
```
llama-orch-sdk (client)
    ↓ HTTP POST /v2/tasks
rbees-orcd (daemon :8080)
    ↓ HTTP POST /execute
rbees-workerd (worker :8001)
    ↓ SSE stream
llama-orch-sdk (client)
```

---

## Implementation Plan

### M0: SSH Control + Direct Execution

**Binaries:**
1. `rbees-ctl` (CLI on blep)
   - Commands pools via SSH
   - No HTTP client for control
   - Uses: orchestrator-core (shared types)

2. `rbees-pool` (CLI on pools)
   - Executes directly (no daemon)
   - Model downloads (hf CLI)
   - Git operations (git CLI)
   - Worker spawn (direct process spawn)
   - Uses: pool-core (shared types)

3. `rbees-workerd` (worker daemon)
   - HTTP server for inference
   - Spawned by rbees-pool

**No daemons for control plane in M0.**

### M1: Add Pool Manager Daemon (Optional)

**Add:**
4. `pool-managerd` (daemon on pools)
   - HTTP server :9200
   - Worker lifecycle via HTTP
   - Heartbeat to orchestrator (HTTP)
   - Uses: pool-core

**Update:**
- `rbees-pool` can call pool-managerd HTTP API (optional)
- `rbees-ctl` can call pool-managerd HTTP API via SSH tunnel (optional)

**Still supports SSH fallback.**

### M2: Add Orchestrator Daemon

**Add:**
5. `rbees-orcd` (daemon on blep)
   - HTTP server :8080
   - Job scheduling
   - SQLite state
   - Uses: orchestrator-core

**Update:**
- `rbees-ctl` calls rbees-orcd HTTP API for jobs
- `rbees-orcd` calls pool-managerd HTTP API
- `llama-orch-sdk` calls rbees-orcd HTTP API for inference

**SSH still available for direct pool control.**

---

## Recommended Decision

### Use SSH for Control Plane (M0)

**Reasons:**
1. **Your homelab is trusted** - SSH is perfect for this
2. **SSH already configured** - No new setup needed
3. **No daemons needed** - Simpler for M0
4. **Secure by default** - SSH keys, no passwords
5. **Spec allows it** - Home Mode (SYS-3.1.1) is flexible

**Update specs:**
- Add SYS-5.8.x: SSH Control Protocol (M0)
- Clarify SYS-5.2.x: HTTP required for M2+, optional for M0
- Document SSH as valid control mechanism for trusted networks

### Keep HTTP for Data Plane (All Modes)

**Reasons:**
1. **Inference needs streaming** - SSE over HTTP
2. **SDK already exists** - llama-orch-sdk
3. **Spec defines it** - SYS-5.1.x, SYS-5.4.x
4. **Clients expect it** - Web UI, applications

---

## Final Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ CONTROL PLANE (Operator Control)                                │
├─────────────────────────────────────────────────────────────────┤
│ M0: SSH-based (homelab)                                          │
│   rbees-ctl → SSH → rbees-pool → direct execution                │
│                                                                   │
│ M2+: HTTP-based (production, optional)                           │
│   rbees-ctl → HTTP → rbees-orcd → HTTP → pool-managerd      │
│                                                                   │
│ Security: SSH keys (M0), Bearer tokens + TLS (M2+)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DATA PLANE (Inference API)                                      │
├─────────────────────────────────────────────────────────────────┤
│ Always HTTP-based (all modes)                                    │
│   llama-orch-sdk → HTTP → rbees-orcd → HTTP → worker         │
│   (SSE streaming for tokens)                                     │
│                                                                   │
│ Security: API tokens (M3+), localhost trust (M0)                │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Control plane and data plane are separate. Use the right tool for each.

---

## Command Examples

### M0: SSH Control

```bash
# On blep (orchestrator host)
llorch pool models download tinyllama --host mac
  → ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool models download tinyllama"
  → hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...

llorch pool worker spawn metal --host mac --model tinyllama
  → ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool worker spawn metal --model tinyllama"
  → rbees-workerd --backend metal ... &

# On mac (pool host)
rbees-pool models download tinyllama
  → hf download ...

rbees-pool worker spawn metal --model tinyllama
  → rbees-workerd --backend metal ... &
```

### M2+: HTTP Control (Optional)

```bash
# Start daemons
rbees-orcd --config orch.toml    # :8080
pool-managerd --config pool.toml    # :9200

# CLI calls HTTP
llorch pool worker spawn metal --host mac --model tinyllama
  → HTTP: POST http://localhost:8080/pools/mac/workers/spawn
  → rbees-orcd: POST http://mac.home.arpa:9200/workers/spawn
  → pool-managerd spawns worker
```

### Inference: Always HTTP

```rust
// Applications always use HTTP (llama-orch-sdk)
let client = Client::new("http://localhost:8080");
let response = client.enqueue(EnqueueRequest {
    prompt: "Hello",
    model: "llama3",
    max_tokens: 100,
}).await?;

// Stream tokens via SSE
let mut stream = client.enqueue_stream(request).await?;
while let Some(event) = stream.next().await {
    print!("{}", event.text);
}
```

---

## Summary

**Recommendation:** SSH for control plane (M0), HTTP for data plane (always)

**Benefits:**
- ✅ Secure (SSH keys)
- ✅ Simple (no daemons needed for M0)
- ✅ Homelab-friendly (already configured)
- ✅ Progressive (can add HTTP later)
- ✅ Separation of concerns (control vs inference)

**Trade-offs:**
- ⚠️ Spec deviation (need to update SYS-5.2.x)
- ⚠️ SSH dependency (but already have it)
- ⚠️ Two protocols (but for different purposes)

**Action items:**
1. Update specs to allow SSH control in M0
2. Implement rbees-pool with direct execution
3. Implement rbees-ctl with SSH client
4. Keep HTTP APIs in specs for M2+
5. llama-orch-sdk remains HTTP-only (inference)

---

**Version**: 1.0  
**Status**: Recommendation (awaiting decision)  
**Last Updated**: 2025-10-09

---

**End of Decision Document**
