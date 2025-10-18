# Orchestrator Standards - What rbee Already Does

**Created by:** TEAM-113  
**Date:** 2025-10-18  
**Purpose:** Document industry standards that rbee ALREADY follows as an orchestrator

---

## 🎯 What rbee IS

**rbee is an orchestrator for GPU/CPU inference workloads.**

### Components
- **queen-rbee** - Control plane (API server, scheduler, registry)
- **rbee-hive** - Node manager (spawns workers, monitors health)
- **llm-worker-rbee** - Worker process (runs inference)

**We orchestrate inference workloads. We don't need another orchestrator.**

---

## ✅ Industry Standards rbee Already Follows

### 1. Server-Worker Architecture
**Standard:** Distributed systems pattern used by GPUStack, Kubernetes, Docker Swarm

**What rbee does:**
- queen-rbee = Server (control plane)
- rbee-hive = Worker node manager
- llm-worker-rbee = Worker process
- Communication via HTTP APIs

**Status:** ✅ Already implemented

---

### 2. SQLite for Persistence
**Standard:** Embedded database used by Ollama, GPUStack, many orchestrators

**What rbee does:**
- Model catalog in SQLite
- Beehive registry in SQLite
- Worker registry in-memory (ephemeral)

**Status:** ✅ Already implemented

---

### 3. HTTP REST APIs
**Standard:** Industry standard for service communication

**What rbee does:**
- queen-rbee exposes HTTP API
- rbee-hive exposes HTTP API
- llm-worker-rbee exposes HTTP API
- JSON request/response format

**Status:** ✅ Already implemented

---

### 4. Bearer Token Authentication (RFC 6750)
**Standard:** HTTP authentication for APIs

**What rbee does:**
- Authorization: Bearer <token> header
- Timing-safe token comparison
- Token fingerprinting in logs (never raw tokens)

**Status:** ✅ Already implemented (queen-rbee, rbee-hive)

---

### 5. Health Endpoints
**Standard:** HTTP health checks for monitoring

**What rbee does:**
- `/health` endpoint on all components
- Returns 200 OK when healthy
- Simple, effective

**Status:** ✅ Already implemented

---

### 6. Prometheus Metrics
**Standard:** Metrics exposition format

**What rbee does:**
- `/metrics` endpoint exposes Prometheus format
- Worker state metrics
- Model download metrics
- Ready for Grafana dashboards

**Status:** ✅ Already implemented

**Prometheus naming conventions (for reference):**
- `_total` suffix for counters
- `_seconds` suffix for durations
- `_bytes` suffix for sizes
- Use labels for dimensions

---

### 7. Audit Logging
**Standard:** Security event logging for compliance

**What rbee does:**
- Audit logging library exists
- Logs authentication events
- Logs worker lifecycle events
- Never logs secrets (uses fingerprints)

**Status:** ✅ Being wired in Week 2

---

### 8. Secrets Management
**Standard:** Secure credential handling

**What rbee does:**
- Load secrets from files (not environment variables)
- File permission validation
- Memory zeroization
- Systemd credentials support

**Status:** ✅ Already implemented

---

### 9. Input Validation
**Standard:** Validate all external inputs

**What rbee does:**
- Validate model references
- Validate identifiers (node names, worker IDs)
- Validate paths (prevent traversal)
- Sanitize strings (prevent log injection)

**Status:** ✅ Already implemented (TEAM-113 wired it)

---

### 10. BDD Testing (Gherkin)
**Standard:** Behavior-driven development testing

**What rbee does:**
- 29 feature files
- ~300 scenarios
- Given-When-Then format
- Cucumber framework

**Status:** ✅ Already implemented

---

## 📚 Reference: Other Orchestrators

### GPUStack
**What it is:** GPU cluster manager (similar to rbee)

**Similarities:**
- Server-worker architecture
- SQLite persistence
- HTTP APIs
- Scheduler assigns work to workers

**Differences:**
- GPUStack does automatic CPU fallback (we don't)
- GPUStack has separate scheduler component (ours is integrated)
- GPUStack targets enterprise (we target simplicity)

**Takeaway:** Similar architecture, different design choices

---

### Kubernetes
**What it is:** Generic container orchestrator

**Similarities:**
- Control plane + node agents
- Scheduler assigns work
- Health monitoring
- Resource management

**Differences:**
- Kubernetes orchestrates ANY workload (we do inference only)
- Kubernetes is complex (we're simple)
- Kubernetes needs cluster setup (we're self-contained)
- Kubernetes uses YAML (we use code/config files)

**Takeaway:** rbee is the simple alternative to Kubernetes for inference

---

### Ollama
**What it is:** Simple LLM runner (not an orchestrator)

**Similarities:**
- SQLite for model catalog
- GGUF model support
- Simple UX

**Differences:**
- Ollama is single-node (we're distributed)
- Ollama is CLI-focused (we're API-focused)
- Ollama doesn't orchestrate (we do)

**Takeaway:** Different use case (single-node vs distributed)

---

## 🚫 What rbee Does NOT Do (By Design)

### 1. Automatic CPU Fallback
**What it is:** Automatically switch from GPU to CPU if GPU unavailable

**Why we don't:** User explicitly chooses worker type (GPU or CPU)

---

### 2. Generic Container Orchestration
**What it is:** Run any Docker container (like Kubernetes)

**Why we don't:** We're focused on inference workloads only

---

### 3. Multi-Tenancy / RBAC
**What it is:** Multiple users with different permissions

**Why we don't (yet):** v0.1.0 targets single-user or trusted environments

**Note:** Could add in future if needed

---

### 4. Distributed Tracing (OpenTelemetry)
**What it is:** Trace requests across services

**Why we don't (yet):** Logs + metrics are sufficient for v0.1.0

**Note:** Could add in future if needed

---

## 🎯 rbee's Design Philosophy

### Simple Over Complex
- Self-contained (no cluster setup)
- Focused (inference only)
- Minimal dependencies

### Explicit Over Automatic
- User chooses worker type (no fallback)
- User chooses model
- User chooses backend

### Practical Over Perfect
- SQLite over distributed database
- Bearer tokens over mTLS
- Simple health checks over complex probes

---

## 📊 Comparison Matrix

| Feature | rbee | Kubernetes | GPUStack | Ollama |
|---------|------|------------|----------|--------|
| **Purpose** | Inference orchestrator | Generic orchestrator | GPU cluster manager | Single-node LLM |
| **Complexity** | Simple | Complex | Medium | Simple |
| **Setup** | Self-contained | Cluster required | Cluster required | Single binary |
| **Workloads** | Inference only | Any container | Inference only | Single model |
| **Distribution** | Multi-node | Multi-node | Multi-node | Single-node |
| **Auth** | Bearer tokens | mTLS/RBAC | API keys | None |
| **Persistence** | SQLite | etcd | SQLite/PostgreSQL | SQLite |
| **Target** | Home lab, small teams | Enterprise | Enterprise | Personal use |

---

## ✅ Summary

### What rbee Already Does (Industry-Standard)
1. ✅ Server-worker architecture
2. ✅ SQLite persistence
3. ✅ HTTP REST APIs
4. ✅ Bearer token authentication
5. ✅ Health endpoints
6. ✅ Prometheus metrics
7. ✅ Audit logging
8. ✅ Secrets management
9. ✅ Input validation
10. ✅ BDD testing

### rbee's Position
- **Orchestrator** for inference workloads
- **Simpler** than Kubernetes
- **Focused** on GPU/CPU inference
- **Self-contained** (no external dependencies)

### Industry Standards That Apply
- HTTP APIs
- Prometheus metrics
- Bearer token auth (RFC 6750)
- SQLite persistence
- Server-worker pattern

### Industry Standards That DON'T Apply
- Kubernetes-specific patterns (we're not in Kubernetes)
- Generic container orchestration (we're inference-focused)
- Enterprise complexity (we're simple by design)

---

**Maintained by:** TEAM-113  
**Last Updated:** 2025-10-18  
**References:** GPUStack (similar), Kubernetes (different approach), Ollama (different scale)
