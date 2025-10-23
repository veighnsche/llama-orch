# rbee-hive

**Status:** 🚧 IN DEVELOPMENT (TEAM-261)  
**Purpose:** Daemon for managing LLM worker instances on a single machine

## 🎯 Architectural Decision (Oct 23, 2025)

**Decision:** Keep as HTTP daemon, remove hive heartbeat

**Why Daemon?**
- ✅ **Performance:** 1-5ms response time (vs 80-350ms for CLI)
- ✅ **UX:** Real-time SSE streaming for progress updates
- ✅ **Security:** No command injection risk
- ✅ **Consistency:** Matches queen/worker patterns

**Simplification:**
- ❌ **No hive heartbeat** - Workers send heartbeats directly to queen
- ✅ **Queen is single source of truth** for worker state
- ✅ **Simpler architecture** - No state aggregation

**See:** `bin/.plan/TEAM_261_SIMPLIFICATION_AUDIT.md` for details

## Overview

rbee-hive is a daemon that manages multiple LLM worker instances on a single machine.

**CRITICAL:** This is a DAEMON ONLY binary. NO CLI functionality!

## Binary Structure

```
bin/20_rbee_hive/
├── src/
│   ├── main.rs          (Entry point - daemon args only)
│   ├── http_server.rs   (HTTP routes & API - IN BINARY)
│   └── lib.rs           (Re-exports from crates)
└── Cargo.toml
```

**IMPORTANT:** HTTP server entry point is implemented DIRECTLY in the binary,
not as a separate crate.

## Dependencies

- rbee-hive-crates/worker-lifecycle
- rbee-hive-crates/worker-registry
- rbee-hive-crates/model-catalog
- rbee-hive-crates/model-provisioner
- rbee-hive-crates/monitor
- rbee-hive-crates/download-tracker
- rbee-hive-crates/device-detection
- rbee-hive-crates/vram-checker
- rbee-hive-crates/worker-catalog

## Usage

```bash
# Daemon arguments only
rbee-hive --config /path/to/config.toml
```

## Implementation Status

- [ ] Core functionality
- [ ] Tests
- [ ] Documentation
- [ ] Examples

## Notes

**NO CLI crate!** This is daemon-only. All management is done via HTTP API or queen-rbee.
