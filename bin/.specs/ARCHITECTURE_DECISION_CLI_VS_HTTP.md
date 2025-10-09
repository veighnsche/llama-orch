# Architecture Decision: CLI vs HTTP for Binary Control

**Status**: Decision Required  
**Date**: 2025-10-09  
**Context**: Determining control interface for rbees-orcd, pool-managerd, and rbees-workerd

---

## Executive Summary

We need to decide how operators interact with the three runtime binaries:
- **rbees-orcd** - The brain (scheduling, admission, job management)
- **pool-managerd** - Control plane (worker lifecycle, GPU inventory)
- **rbees-workerd** - Worker (inference execution)

**Options:**
1. **HTTP APIs only** - All control via HTTP endpoints
2. **CLI wrappers** - Separate CLI binaries that call HTTP APIs
3. **Hybrid** - Built-in CLI + HTTP APIs in same binary

---

## Current Architecture (From Specs)

### From `bin/.specs/00_llama-orch.md`

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATORD (The Brain - All Intelligent Decisions)           │
│ • HTTP API: POST /v2/tasks (client submissions)                 │
│ • HTTP API: POST /pools/{id}/workers/spawn (pool commands)      │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP Commands
                     ↓
┌────────────────────────┐
│ POOL-MANAGERD         │
│ • HTTP API: POST /workers/spawn                                 │
│ • HTTP API: GET /status                                         │
│ • Heartbeat: POST orchestrator/heartbeat                        │
└─────┬──────────────────┘
      │ Process spawn
      ↓
┌──────┐
│WORKER│ (rbees-workerd)
│ • HTTP API: POST /execute                                       │
│ • HTTP API: POST /cancel                                        │
│ • Callback: POST pool-manager/workers/ready                     │
└──────┘
```

**Key insight:** The specs define HTTP APIs for inter-component communication, but say nothing about operator control interfaces.

---

## Option 1: HTTP APIs Only

### Architecture

```
Operator
    ↓ curl/httpie/custom client
rbees-orcd HTTP API (:8080)
    ↓ HTTP
pool-managerd HTTP API (:9200)
    ↓ spawn
rbees-workerd HTTP API (:8001)
```

### Pros

**✅ Simplicity**
- Single interface for everything (HTTP)
- No CLI code to maintain
- Clear separation: HTTP for all communication

**✅ Language Agnostic**
- Any language can interact (curl, Python, Go, etc.)
- Easy to build custom tooling
- OpenAPI spec is the contract

**✅ Remote-First**
- HTTP works over network by default
- No SSH required for remote operations
- Same interface local and remote

**✅ Programmatic**
- Easy to automate (scripts, CI/CD)
- Easy to build UIs (web dashboards)
- Easy to integrate with other systems

**✅ Spec Compliance**
- Specs already define HTTP APIs (SYS-5.x)
- No additional interface to specify
- Clear contract via OpenAPI

### Cons

**❌ Developer Experience**
- Verbose: `curl -X POST http://localhost:8080/v2/tasks -H "Content-Type: application/json" -d '{"model":"llama3","prompt":"hello"}'`
- No tab completion
- No built-in help
- Error messages are JSON (not human-friendly)

**❌ Discovery**
- How do operators know what endpoints exist?
- Must read OpenAPI spec or docs
- No `--help` flag

**❌ Common Operations**
- Simple tasks require complex HTTP calls
- No shortcuts for common workflows
- Must remember JSON structure

**❌ Local Development**
- Still need scripts for local dev workflow
- Git operations, model downloads, builds still need tooling
- HTTP doesn't help with "download model" or "build worker"

### Example Usage

```bash
# Submit job
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "prompt": "Hello world",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# List jobs
curl http://localhost:8080/v2/tasks

# Get pool status
curl http://localhost:9200/status

# Spawn worker
curl -X POST http://localhost:9200/workers/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "worker-1",
    "model_ref": "hf:meta-llama/Llama-3-8B",
    "gpu_id": 0
  }'
```

---

## Option 2: Separate CLI Binaries

### Architecture

```
Operator
    ↓ CLI commands
rbees-orcd-cli (separate binary)
    ↓ HTTP calls
rbees-orcd HTTP API (:8080)
    ↓ HTTP
pool-managerd HTTP API (:9200)
    ↓ spawn
rbees-workerd HTTP API (:8001)
```

**Binaries:**
- `rbees-orcd` - Daemon (HTTP server)
- `rbees-orcd-cli` - CLI client (HTTP client)
- `pool-managerd` - Daemon (HTTP server)
- `pool-managerd-cli` - CLI client (HTTP client)
- `rbees-workerd` - Worker daemon (HTTP server)

### Pros

**✅ Developer Experience**
- Intuitive: `llorch jobs submit --model llama3 --prompt "hello"`
- Tab completion
- Built-in help (`--help`)
- Human-friendly error messages

**✅ Discovery**
- `llorch --help` shows all commands
- `llorch jobs --help` shows job commands
- Self-documenting

**✅ Shortcuts**
- Common operations are simple
- `llorch jobs list` vs long curl command
- Can provide workflows: `llorch dev setup`

**✅ Separation of Concerns**
- Daemon is pure HTTP server
- CLI is pure HTTP client
- Clear boundaries

**✅ Optional**
- Operators can use CLI or HTTP directly
- CLI is convenience, not requirement
- Power users can bypass CLI

**✅ Development Tooling**
- CLI can include dev commands (setup, doctor, check)
- Git operations, model downloads, builds
- Not just runtime control

### Cons

**❌ Maintenance Burden**
- More code to maintain (CLI + daemon)
- Two binaries per component (6 total)
- CLI must stay in sync with HTTP API

**❌ Duplication**
- CLI logic duplicates HTTP API contracts
- Changes require updating both
- More testing surface

**❌ Deployment Complexity**
- Must distribute multiple binaries
- Version compatibility concerns
- CLI version must match daemon version

**❌ Indirection**
- CLI → HTTP → Daemon (extra layer)
- Debugging requires understanding both
- Error messages may be less clear

### Example Usage

```bash
# Submit job (CLI)
llorch jobs submit \
  --model llama3 \
  --prompt "Hello world" \
  --max-tokens 100 \
  --temperature 0.7

# List jobs (CLI)
llorch jobs list

# Get pool status (CLI)
llorch pool status --host mac

# Spawn worker (CLI)
llorch pool worker spawn \
  --host mac \
  --backend metal \
  --model llama3 \
  --gpu 0

# Development commands (CLI only, not HTTP)
llorch dev setup
llorch dev doctor
llorch pool models download tinyllama --host mac
llorch pool git pull --host workstation
```

---

## Option 3: Hybrid (Built-in CLI + HTTP)

### Architecture

```
Operator
    ↓ CLI mode OR HTTP mode
rbees-orcd (single binary)
    ├─ CLI mode: rbees-orcd jobs submit ...
    └─ Daemon mode: rbees-orcd serve
```

**Single binary with dual modes:**
- `rbees-orcd serve` - Run as HTTP daemon
- `rbees-orcd jobs submit` - CLI client mode (calls HTTP API)

### Pros

**✅ Single Binary**
- Only one binary to distribute
- No version mismatch issues
- Simpler deployment

**✅ Best of Both Worlds**
- HTTP API for programmatic access
- CLI for human operators
- Same binary, different modes

**✅ Guaranteed Sync**
- CLI and HTTP in same codebase
- Can't get out of sync
- Shared types and logic

**✅ Flexibility**
- Operators choose interface
- Can mix and match
- CLI calls local or remote HTTP

### Cons

**❌ Binary Size**
- Larger binary (includes CLI + daemon)
- More dependencies
- Slower compilation

**❌ Complexity**
- Single binary with multiple modes
- More complex argument parsing
- Potential for confusion (serve vs client mode)

**❌ Coupling**
- CLI and daemon tightly coupled
- Changes affect both modes
- Harder to evolve independently

**❌ Testing**
- Must test both modes
- Integration tests more complex
- More edge cases

### Example Usage

```bash
# Start daemon
rbees-orcd serve --config /etc/llorch/orchestrator.toml

# CLI client mode (calls HTTP API)
rbees-orcd jobs submit \
  --model llama3 \
  --prompt "Hello world" \
  --endpoint http://localhost:8080

# Or use HTTP directly
curl -X POST http://localhost:8080/v2/tasks -d '{...}'
```

---

## Comparison Matrix

| Aspect | HTTP Only | Separate CLI | Hybrid |
|--------|-----------|--------------|--------|
| **Developer Experience** | ❌ Poor | ✅ Excellent | ✅ Excellent |
| **Maintenance Burden** | ✅ Low | ❌ High | ⚠️ Medium |
| **Binary Count** | ✅ 3 binaries | ❌ 6 binaries | ✅ 3 binaries |
| **Deployment Complexity** | ✅ Simple | ❌ Complex | ✅ Simple |
| **Spec Compliance** | ✅ Perfect | ✅ Perfect | ✅ Perfect |
| **Discovery** | ❌ Poor | ✅ Excellent | ✅ Excellent |
| **Programmatic Access** | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Remote Operations** | ✅ Native | ⚠️ Via SSH | ⚠️ Via HTTP |
| **Development Tooling** | ❌ Needs scripts | ✅ Built-in | ✅ Built-in |
| **Version Sync** | ✅ N/A | ❌ Must match | ✅ Guaranteed |

---

## Real-World Examples

### Kubernetes (Separate CLI)
```bash
kubectl get pods              # CLI
curl $APISERVER/api/v1/pods   # HTTP API
```
- **Model**: Separate CLI (`kubectl`) + HTTP API server
- **Why**: Best DX, programmatic access via HTTP
- **Downside**: Must keep kubectl in sync with API

### Docker (Hybrid)
```bash
docker ps                     # CLI mode
docker daemon                 # Daemon mode (deprecated)
dockerd                       # Daemon (separate binary now)
```
- **Model**: Started hybrid, moved to separate binaries
- **Why**: Separation of concerns, easier to maintain
- **Evolution**: Recognized separate binaries are cleaner

### Systemd (HTTP + CLI)
```bash
systemctl start nginx         # CLI
curl -X POST /org/freedesktop/systemd1/unit/nginx.service/Start  # D-Bus (like HTTP)
```
- **Model**: CLI + D-Bus API
- **Why**: Both human and programmatic interfaces
- **Note**: D-Bus is like HTTP for system services

### Prometheus (HTTP Only)
```bash
curl http://localhost:9090/api/v1/query?query=up  # HTTP only
```
- **Model**: HTTP API only, no CLI
- **Why**: Designed for programmatic access
- **Downside**: Poor DX, need external tools (promtool)

---

## Recommendation

### For llama-orch: **Option 2 (Separate CLI Binaries)**

**Rationale:**

1. **Spec Compliance**
   - Specs define HTTP APIs (SYS-5.x) ✅
   - CLI is additional convenience, not replacement
   - Daemons remain pure HTTP servers

2. **Developer Experience**
   - Operators need good DX for homelab/dev
   - `llorch jobs submit` >> long curl command
   - Built-in help and discovery

3. **Development Tooling**
   - Need tooling for git, models, builds anyway
   - CLI can include dev commands
   - Not just runtime control

4. **Separation of Concerns**
   - Daemons: long-running, HTTP servers, state management
   - CLI: short-lived, HTTP clients, no state
   - Clear boundaries

5. **Evolution Path**
   - Start with CLI for M0 (development)
   - Daemons can evolve independently
   - Can add web UI later (also HTTP client)

6. **Real-World Validation**
   - Kubernetes model is proven
   - Docker moved to separate binaries
   - Industry standard pattern

### Proposed Structure

```
bin/
├── rbees-orcd/           # Daemon (HTTP server)
│   ├── src/
│   │   ├── main.rs         # HTTP server only
│   │   ├── api/            # HTTP endpoints
│   │   ├── scheduler/      # Scheduling logic
│   │   └── state/          # State management
│   └── Cargo.toml
│
├── rbees-orcd-cli/       # CLI client (HTTP client)
│   ├── src/
│   │   ├── main.rs         # CLI entry point
│   │   ├── commands/       # CLI commands
│   │   │   ├── jobs.rs     # rbees jobs ...
│   │   │   ├── pools.rs    # rbees pool ...
│   │   │   └── dev.rs      # rbees dev ...
│   │   └── client.rs       # HTTP client
│   └── Cargo.toml
│
├── pool-managerd/           # Daemon (HTTP server)
│   └── ...
│
├── pool-managerd-cli/       # CLI client (HTTP client)
│   └── ...
│
└── rbees-workerd/          # Worker daemon (HTTP server)
    └── ...                  # No CLI needed (controlled by pool-managerd)
```

### Command Examples

```bash
# Orchestrator daemon (systemd manages this)
rbees-orcd --config /etc/llorch/orchestrator.toml

# Orchestrator CLI (operators use this)
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list
llorch jobs cancel job-123
llorch pool status --host mac
llorch pool worker spawn --host mac --backend metal
llorch dev setup
llorch dev doctor

# Pool manager daemon (systemd manages this)
pool-managerd --config /etc/llorch/pool-manager.toml

# Pool manager CLI (local operations on pool host)
rbees-pool models download tinyllama
rbees-pool models list
rbees-pool git pull
rbees-pool worker spawn metal --model tinyllama --gpu 0

# Worker daemon (pool-managerd spawns this)
rbees-workerd \
  --worker-id worker-1 \
  --model /path/to/model.gguf \
  --gpu 0 \
  --callback-url http://pool-manager:9200/workers/ready
```

---

## Implementation Plan

### Phase 1: HTTP APIs (M0)
- Implement daemons with HTTP APIs
- Follow specs (SYS-5.x)
- Test with curl/httpie

### Phase 2: CLI Clients (M0+)
- Implement CLI binaries
- Call HTTP APIs
- Provide good DX

### Phase 3: Development Tooling (M0+)
- Add dev commands to CLI
- Git operations, model downloads, builds
- Replace bash scripts

### Phase 4: Web UI (M2+)
- Build web dashboard
- Also calls HTTP APIs
- Same as CLI, different interface

---

## Decision

**Recommended**: Option 2 (Separate CLI Binaries)

**Rationale**: Best developer experience while maintaining spec compliance and separation of concerns. CLI is convenience layer over HTTP APIs, not replacement.

**Action Items**:
1. Keep daemon specs focused on HTTP APIs (SYS-5.x)
2. Create separate CLI specs for operator tooling
3. Implement daemons first (HTTP only)
4. Add CLI clients for DX
5. Delete bash scripts after CLI complete

---

**Version**: 1.0  
**Status**: Recommendation (awaiting decision)  
**Last Updated**: 2025-10-09

---

**End of Architecture Decision**
