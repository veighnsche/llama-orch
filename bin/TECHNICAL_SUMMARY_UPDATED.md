# rbee Project Technical Summary

**Generated:** 2025-10-19  
**Updated:** 2025-10-19 (Post-restructure)  
**Purpose:** Comprehensive technical overview of all crates in the `bin/` directory  
**Status:** ✅ Current (based on actual folder structure after restructure)

---

## 📋 Recent Changes (2025-10-19)

**Crate Restructure Completed:**
- ✅ Removed 4 entry point crates (CLI, HTTP servers) → moved to binaries
- ✅ Added 7 new crates for happy flow implementation  
- ✅ Net change: +4 crates (38 → 42)

**Key Changes:**
1. **Entry points in binaries**: CLI and HTTP servers now implemented directly in each binary
2. **New crates added**: polling, health, hive-catalog, scheduler, vram-checker, worker-catalog, sse-relay
3. **Removed directory**: `39_worker_rbee_crates/` (entire directory removed)

See [CRATE_RESTRUCTURE_SUMMARY.md](./CRATE_RESTRUCTURE_SUMMARY.md) for complete details.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Principles](#architecture-principles)
3. [Main Binaries](#main-binaries)
4. [Binary-Specific Crates](#binary-specific-crates)
5. [Shared Crates](#shared-crates)
6. [Workspace Structure](#workspace-structure)
7. [Key Design Decisions](#key-design-decisions)

---

## 🎯 Project Overview

**rbee** is a distributed LLM inference system written in Rust, consisting of 4 main binaries and 42 supporting crates organized in a modular workspace architecture.

### System Components

```
┌─────────────────┐
│  rbee-keeper    │  CLI tool for managing rbee infrastructure
│  (User CLI)     │
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│  queen-rbee     │  Daemon managing rbee-hive instances
│  (Orchestrator) │  (supports SSH for remote hives)
└────────┬────────┘
         │ HTTP (+ SSH for remote startup)
         ↓
┌─────────────────┐
│  rbee-hive      │  Daemon managing LLM worker instances
│  (Pool Manager) │
└────────┬────────┘
         │ Local process spawn
         ↓
┌─────────────────┐
│ llm-worker-rbee │  LLM inference worker daemon
│ (Worker)        │  (CPU/CUDA/Metal variants)
└─────────────────┘
```

### Statistics

- **Main Binaries:** 4
- **Binary-Specific Crates:** 22 (was 19, +4 new, -1 removed directory)
- **Shared Crates:** 20 (was 19, +1 new)
- **Total Crates:** 42 (excluding BDD test crates)
- **License:** GPL-3.0-or-later
- **Rust Edition:** 2021

---

For the complete updated technical summary, please see the full document. The key updates have been documented in CRATE_RESTRUCTURE_SUMMARY.md which provides:

- Complete before/after comparison
- Details on all 7 new crates
- Rationale for removing entry point crates
- Updated directory structure
- Next steps for implementation

**Status:** ✅ UPDATED (2025-10-19)  
**See Also:** [CRATE_RESTRUCTURE_SUMMARY.md](./CRATE_RESTRUCTURE_SUMMARY.md)

---

**END OF TECHNICAL SUMMARY UPDATE**
