# Component Documentation

**Location:** `.docs/components/`  
**Purpose:** Comprehensive documentation of all rbee ecosystem components  
**Last Updated:** TEAM-096 | 2025-10-18

## Overview

This directory contains detailed documentation for all components, subsystems, and shared libraries in the rbee ecosystem. Each document includes architecture, API, maturity assessment, and integration guidance.

---

## Quick Start

**Start here:** `COMPONENT_INDEX.md` - Master index of all 15 components

---

## Documentation Structure

### Core Binaries (3 docs)
User-facing tools and daemons:
- `RBEE_KEEPER.md` - CLI tool for managing rbee ecosystem
- `QUEEN_RBEE.md` - Central orchestrator daemon
- `RBEE_HIVE.md` - Worker pool manager daemon

### Registries (4 docs)
Persistent and ephemeral data storage:
- `BEEHIVE_REGISTRY.md` - SQLite hive registration (queen-rbee)
- `WORKER_REGISTRY_QUEEN.md` - RAM worker tracking across all hives
- `WORKER_REGISTRY_HIVE.md` - RAM worker tracking per hive
- `MODEL_CATALOG.md` - SQLite model download tracking

### Provisioning & Management (2 docs)
Resource provisioning and lifecycle:
- `MODEL_PROVISIONER.md` - HuggingFace model downloads with SSE
- `WORKER_LIFECYCLE.md` - Worker process lifecycle management

### Communication (2 docs)
Inter-component communication:
- `SSE_STREAMING.md` - Real-time streaming (downloads, inference)
- `LOCAL_VS_NETWORK_MODE.md` - Deployment mode strategies

### Shared Libraries (2 docs)
Reusable security and utility crates:
- `SHARED_CRATES.md` - 9 production-ready shared crates
- `SHARED_CRATES_INTEGRATION.md` - Integration guide with examples

### Analysis Documents (3 docs)
Technical analysis and fixes:
- `LIFECYCLE_MANAGEMENT_GAPS.md` - Worker lifecycle gaps analysis
- `TEAM_096_PORT_ALLOCATION_FIX.md` - Port allocation bug fix
- `TEAM_096_SUMMARY.md` - TEAM-096 work summary

---

## Component Status Summary

### ✅ Production Ready (9 components)
- rbee-keeper (CLI)
- queen-rbee (Orchestrator)
- Beehive Registry (SQLite)
- Worker Registry (queen-rbee, RAM)
- Model Catalog (SQLite)
- Model Provisioner
- SSE Streaming
- Local/Network Mode
- **9 Shared Crates** (auth-min, jwt-guardian, secrets-management, etc.)

### 🟡 Functional with Gaps (2 components)
- rbee-hive (Worker pool manager) - Missing PID tracking
- Worker Lifecycle - Missing force kill, restart policy

### 🔴 Not Yet Implemented (1 component)
- Scheduler (future milestone)

---

## Key Findings

### What's Already Built
1. **Security Infrastructure** - 3 production-ready security crates (auth-min, jwt-guardian, secrets-management)
2. **Validation** - Input validation, deadline propagation, audit logging
3. **Orchestration** - Complete queen-rbee with hive/worker registries
4. **Provisioning** - Model downloads with progress streaming
5. **Communication** - SSH support, SSE streaming, local/network modes

### Critical Gaps
1. **Worker Lifecycle** - No PID tracking in HTTP daemon mode (CLI mode has it)
2. **Integration** - Shared crates ready but not integrated into main components
3. **Scheduler** - Not yet implemented (future milestone)

### Recommended Actions
1. **P0:** Add PID tracking to rbee-hive worker lifecycle
2. **P1:** Integrate security crates (auth-min, secrets-management, input-validation)
3. **P1:** Add audit logging to queen-rbee and rbee-hive
4. **P2:** Implement scheduler for multi-hive load balancing

---

## How to Use This Documentation

### For New Engineers
1. Read `COMPONENT_INDEX.md` - Get overview of all components
2. Read component docs for areas you're working on
3. Check `SHARED_CRATES.md` - See what's already available
4. Use `SHARED_CRATES_INTEGRATION.md` - Learn how to integrate

### For Architecture Review
1. Read `COMPONENT_INDEX.md` - System overview
2. Read `LIFECYCLE_MANAGEMENT_GAPS.md` - Known issues
3. Review maturity assessments in each component doc

### For Security Audit
1. Read `SHARED_CRATES.md` - Security infrastructure
2. Check integration status in `COMPONENT_INDEX.md`
3. Review each component's security section

### For Integration Work
1. Read `SHARED_CRATES_INTEGRATION.md` - Integration patterns
2. Follow examples for your component
3. Update component docs after integration

---

## Document Format

Each component document includes:

1. **Overview** - Purpose and scope
2. **Architecture** - Diagrams and structure
3. **Data Models** - Schemas and types
4. **API Methods** - Complete interface
5. **Lifecycle** - How it works end-to-end
6. **Integration Points** - Connections to other components
7. **Maturity Assessment** - Status, strengths, limitations, recommendations
8. **Testing** - How to test
9. **Related Components** - Cross-references
10. **Historical Context** - Team attributions

---

## Historical Context

This documentation was created by TEAM-096 after discovering that:
1. Many components were already implemented but undocumented
2. 9 production-ready shared crates existed but weren't integrated
3. Previous teams (TEAM-027, 029, 030, 043, 046, 052, 080, 085) built substantial infrastructure

**All historical team contributions are preserved and credited in the documentation.**

---

## Maintenance

### When to Update
- ✅ After implementing new components
- ✅ After integrating shared crates
- ✅ After fixing critical gaps
- ✅ After architecture changes

### How to Update
1. Update relevant component doc
2. Update `COMPONENT_INDEX.md` status
3. Add team attribution (TEAM-XXX)
4. Update maturity assessment

---

## Quick Reference

| Need | Read This |
|------|-----------|
| **System overview** | `COMPONENT_INDEX.md` |
| **Security crates** | `SHARED_CRATES.md` |
| **Integration guide** | `SHARED_CRATES_INTEGRATION.md` |
| **Known issues** | `LIFECYCLE_MANAGEMENT_GAPS.md` |
| **Worker lifecycle** | `WORKER_LIFECYCLE.md` |
| **Model downloads** | `MODEL_PROVISIONER.md` |
| **Registries** | `*_REGISTRY.md` files |
| **Communication** | `SSE_STREAMING.md`, `LOCAL_VS_NETWORK_MODE.md` |

---

## Statistics

- **Total Components:** 15 documented
- **Total Documents:** 17 files
- **Shared Crates:** 9 production-ready
- **Lines of Documentation:** ~8,000+ lines
- **Teams Credited:** 10+ teams (TEAM-027 through TEAM-096)

---

## Contributing

When adding new components:
1. Create component doc using existing format
2. Update `COMPONENT_INDEX.md`
3. Add cross-references to related components
4. Include maturity assessment
5. Credit your team (TEAM-XXX)

---

**Created by:** TEAM-096 | 2025-10-18  
**Purpose:** Provide comprehensive component documentation for future engineering teams
