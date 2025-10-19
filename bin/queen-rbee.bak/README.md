# queen-rbee - Orchestrator Daemon

**Status:** Scaffold (M1 milestone)  
**Binary:** `queen-rbee`  
**Port:** 8080

---

## Purpose

The orchestrator daemon is the brain of the rbee system:
- Job scheduling and admission control
- Worker registry management
- SSE stream relay
- Job queue persistence (SQLite)

---

## Architecture

```
Client (SDK)
    ↓ HTTP POST /v2/tasks
queen-rbee (daemon)
    ↓ HTTP POST /execute (direct to workers)
llm-worker-rbee (worker daemon)
    ↓ SSE stream
queen-rbee (relay)
    ↓ SSE stream
Client
```

---

## Usage

```bash
# Start daemon
queen-rbee --port 8080 --database rbee-orchestrator.db

# With config file
queen-rbee --config /etc/rbee/orchestrator.toml
```

---

## Implementation Status

- [x] Scaffold created
- [ ] HTTP server (Axum)
- [ ] Worker registry
- [ ] Job queue
- [ ] SQLite persistence
- [ ] Admission control
- [ ] Scheduling algorithm
- [ ] SSE relay
- [ ] Signal handlers

---

## Milestone

**M1:** Build orchestrator daemon (after M0 completion)

---

**rbee: Your distributed swarm**
