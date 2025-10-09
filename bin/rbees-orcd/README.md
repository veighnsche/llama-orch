# rbees-orcd - Orchestrator Daemon

**Status:** Scaffold (M1 milestone)  
**Binary:** `rbees-orcd`  
**Port:** 8080

---

## Purpose

The orchestrator daemon is the brain of the rbees system:
- Job scheduling and admission control
- Worker registry management
- SSE stream relay
- Job queue persistence (SQLite)

---

## Architecture

```
Client (SDK)
    ↓ HTTP POST /v2/tasks
rbees-orcd (daemon)
    ↓ HTTP POST /execute (direct to workers)
rbees-workerd (worker daemon)
    ↓ SSE stream
rbees-orcd (relay)
    ↓ SSE stream
Client
```

---

## Usage

```bash
# Start daemon
rbees-orcd --port 8080 --database rbees-orchestrator.db

# With config file
rbees-orcd --config /etc/rbees/orchestrator.toml
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

**rbees: Your distributed swarm**
