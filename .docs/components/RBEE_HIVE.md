# Component: rbee-hive (Worker Pool Manager)

**Location:** `bin/rbee-hive/`  
**Type:** HTTP daemon / Worker pool manager  
**Language:** Rust  
**Purpose:** Manages worker processes on a single machine

## Overview

`rbee-hive` manages worker processes (llm-worker-rbee, etc.) on a single machine. Handles model provisioning, worker lifecycle, and provides HTTP API for worker management.

## Core Responsibilities

### 1. Worker Registry (RAM - Tightly Coupled with Lifecycle)

Ephemeral storage tracking workers spawned by THIS hive only. Cleared on restart.

**Key Fields:**
- Worker ID, URL, model, state
- Health check counter (TEAM-096)
- ❌ Missing: PID (critical gap)

### 2. Worker Lifecycle

**Current State:**
- ✅ Spawn with smart port allocation (TEAM-096)
- ✅ Health checks every 30s
- ✅ Fail-fast after 3 failures (TEAM-096)
- ✅ Idle timeout 5min (TEAM-027)
- ❌ No PID tracking
- ❌ No force kill
- ❌ No restart policy

See `LIFECYCLE_MANAGEMENT_GAPS.md` for full analysis.

### 3. Model Provisioner

Downloads models from HuggingFace, tracks in SQLite catalog.

### 4. Model Catalog (SQLite)

Persistent storage of downloaded models for THIS hive.

### 5. Worker Provisioner (Future)

Download or build worker binaries. Currently assumes binary exists.

## Key Files

- `src/registry.rs` - Worker registry (TEAM-096)
- `src/http/workers.rs` - Worker endpoints (TEAM-096)
- `src/monitor.rs` - Health monitoring (TEAM-096)
- `src/provisioner/` - Model provisioning (TEAM-029)

## Status

✅ Core functionality exists  
🔴 Lifecycle management critically underdeveloped

---

**Last Updated:** TEAM-096 | 2025-10-18
