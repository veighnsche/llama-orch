# TEAM-314: Contract Implementation Plan

**Status:** 📋 PLAN  
**Date:** 2025-10-27  
**Purpose:** Step-by-step plan to implement missing contracts

---

## Overview

This plan implements 3 new contract crates in priority order:

1. **daemon-contract** (CRITICAL) - Generic daemon lifecycle
2. **ssh-contract** (HIGH) - SSH-related types
3. **keeper-config-contract** (MEDIUM) - Keeper configuration

---

## Phase 1: daemon-contract (Week 1)

### Priority: 🔴 CRITICAL

**Goal:** Create generic daemon lifecycle contracts

**Deliverables:**
- New crate: `bin/97_contracts/daemon-contract/`
- Generic `DaemonHandle` type
- Status, install, config types
- Migration of `QueenHandle`
- Addition of `HiveHandle`

**Files to Create:**
```
bin/97_contracts/daemon-contract/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs           # Main exports
    ├── handle.rs        # DaemonHandle (generic)
    ├── status.rs        # StatusRequest/StatusResponse
    ├── install.rs       # InstallConfig/InstallResult
    ├── lifecycle.rs     # HttpDaemonConfig
    └── shutdown.rs      # ShutdownConfig
```

**Estimated Time:** 2-3 days

**See:** `TEAM_314_DAEMON_CONTRACT_IMPLEMENTATION.md` for details

---

## Phase 2: ssh-contract (Week 1-2)

### Priority: 🟡 HIGH

**Goal:** Eliminate SSH type duplication

**Deliverables:**
- New crate: `bin/97_contracts/ssh-contract/`
- `SshTarget` type (moved from ssh-config)
- `SshTargetStatus` enum
- Remove duplication in tauri_commands

**Files to Create:**
```
bin/97_contracts/ssh-contract/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs           # Main exports
    ├── target.rs        # SshTarget
    └── status.rs        # SshTargetStatus
```

**Estimated Time:** 1 day

**See:** `TEAM_314_SSH_CONTRACT_IMPLEMENTATION.md` for details

---

## Phase 3: keeper-config-contract (Week 2)

### Priority: 🟢 MEDIUM

**Goal:** Stable keeper configuration schema

**Deliverables:**
- New crate: `bin/97_contracts/keeper-config-contract/`
- `KeeperConfig` type (moved from rbee-keeper)
- Schema validation
- Migration guide

**Files to Create:**
```
bin/97_contracts/keeper-config-contract/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs           # Main exports
    ├── config.rs        # KeeperConfig
    └── validation.rs    # Schema validation
```

**Estimated Time:** 1 day

**See:** `TEAM_314_KEEPER_CONFIG_CONTRACT_IMPLEMENTATION.md` for details

---

## Timeline

```
Week 1:
  Day 1-2: daemon-contract (handle.rs, status.rs)
  Day 3:   daemon-contract (install.rs, lifecycle.rs)
  Day 4:   daemon-contract (migration, testing)
  Day 5:   ssh-contract (complete)

Week 2:
  Day 1:   keeper-config-contract (complete)
  Day 2-3: Testing, documentation
  Day 4-5: Buffer for issues
```

---

## Success Criteria

### Phase 1: daemon-contract
- ✅ Generic `DaemonHandle` works for queen, hive, workers
- ✅ `QueenHandle` is type alias to `DaemonHandle`
- ✅ `HiveHandle` added and working
- ✅ All tests pass
- ✅ No breaking changes for consumers

### Phase 2: ssh-contract
- ✅ `SshTarget` moved to contract
- ✅ Duplication removed from tauri_commands
- ✅ ssh-config uses contract
- ✅ All tests pass

### Phase 3: keeper-config-contract
- ✅ `KeeperConfig` moved to contract
- ✅ Schema validation added
- ✅ rbee-keeper uses contract
- ✅ All tests pass

---

## Risk Mitigation

### Risk 1: Breaking Changes

**Mitigation:**
- Use type aliases for compatibility
- Gradual migration (one crate at a time)
- Comprehensive testing

### Risk 2: Circular Dependencies

**Mitigation:**
- Contracts depend on nothing (except serde)
- Implementation crates depend on contracts
- Clear dependency graph

### Risk 3: Performance Impact

**Mitigation:**
- Contracts are zero-cost (just types)
- No runtime overhead
- Benchmark critical paths

---

## Dependencies

### daemon-contract
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
```

### ssh-contract
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
```

### keeper-config-contract
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
```

---

## Testing Strategy

### Unit Tests
- Each contract type has tests
- Serialization/deserialization
- Validation logic

### Integration Tests
- Contracts work with consumers
- No breaking changes
- Migration path works

### Documentation Tests
- Examples compile
- README examples work

---

## Documentation Requirements

Each contract crate must have:
1. **README.md** - Purpose, usage, examples
2. **Rustdoc** - All public types documented
3. **Examples** - At least 2 usage examples
4. **Migration Guide** - How to migrate from old code

---

## Post-Implementation

### Cleanup
- Remove old duplicate types
- Update all consumers
- Archive old code

### Documentation
- Update architecture docs
- Update API docs
- Create migration guides

### Monitoring
- Watch for issues
- Collect feedback
- Plan improvements

---

## Next Steps

1. Review this plan
2. Read detailed implementation docs:
   - `TEAM_314_DAEMON_CONTRACT_IMPLEMENTATION.md`
   - `TEAM_314_SSH_CONTRACT_IMPLEMENTATION.md`
   - `TEAM_314_KEEPER_CONFIG_CONTRACT_IMPLEMENTATION.md`
3. Start with Phase 1 (daemon-contract)
4. Follow search instructions in `TEAM_314_SEARCH_INSTRUCTIONS.md`

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** PLAN 📋
