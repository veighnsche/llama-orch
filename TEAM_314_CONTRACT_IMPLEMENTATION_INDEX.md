# TEAM-314: Contract Implementation - Complete Guide

**Status:** 📚 INDEX  
**Date:** 2025-10-27  
**Purpose:** Master index for all contract implementation documents

---

## Documents Overview

This is a complete guide for implementing missing contracts in the rbee ecosystem.

---

## 📋 Planning Documents

### 1. Main Plan
**File:** `TEAM_314_CONTRACT_IMPLEMENTATION_PLAN.md`

**Contents:**
- Overview of all 3 contracts
- Timeline (2 weeks)
- Success criteria
- Risk mitigation
- Dependencies

**Read First:** ✅ Start here

---

### 2. Missing Contracts Analysis
**File:** `TEAM_314_MISSING_CONTRACTS_ANALYSIS.md`

**Contents:**
- 7 missing contracts identified
- Priority ranking
- Impact analysis
- Recommendations

**Purpose:** Understand what's missing and why

---

## 🔴 Phase 1: daemon-contract (CRITICAL)

### Implementation Guide
**File:** `TEAM_314_DAEMON_CONTRACT_IMPLEMENTATION.md`

**Contents:**
- Step-by-step implementation
- Source code locations
- Complete code examples
- Migration guide
- Testing instructions

**Types Implemented:**
- `DaemonHandle` (generic)
- `StatusRequest/StatusResponse`
- `InstallConfig/InstallResult/UninstallConfig`
- `HttpDaemonConfig`
- `ShutdownConfig`

**Estimated Time:** 2-3 days

**Priority:** 🔴 CRITICAL - Do this first!

---

## 🟡 Phase 2: ssh-contract (HIGH)

### Implementation Guide
**File:** `TEAM_314_SSH_CONTRACT_IMPLEMENTATION.md`

**Contents:**
- Step-by-step implementation
- Duplication removal
- Complete code examples
- Migration guide
- Testing instructions

**Types Implemented:**
- `SshTarget`
- `SshTargetStatus`

**Estimated Time:** 1 day

**Priority:** 🟡 HIGH - Do after daemon-contract

---

## 🟢 Phase 3: keeper-config-contract (MEDIUM)

### Implementation Guide
**File:** `TEAM_314_KEEPER_CONFIG_CONTRACT_IMPLEMENTATION.md`

**Contents:**
- Step-by-step implementation
- Schema validation
- Complete code examples
- Migration guide
- Testing instructions

**Types Implemented:**
- `KeeperConfig`
- `ValidationError`

**Estimated Time:** 1 day

**Priority:** 🟢 MEDIUM - Do last

---

## 🔍 Search Instructions

### Search Guide
**File:** `TEAM_314_SEARCH_INSTRUCTIONS.md`

**Contents:**
- How to find type definitions
- How to find all usages
- How to check for parity
- How to verify migration
- Complete search examples

**Purpose:** Ensure nothing is missed during implementation

**Critical Commands:**
```bash
# Find type definitions
rg "pub struct TypeName" --type rust

# Find all usages
rg "TypeName" --type rust

# Find duplicates
rg "pub struct TypeName" --type rust --count-matches

# Verify migration
rg "OldTypeName" --type rust  # Should be 0 after
```

---

## 📊 Implementation Order

```
Week 1:
├── Day 1-2: daemon-contract
│   ├── Create crate
│   ├── Implement DaemonHandle
│   ├── Implement status types
│   └── Migrate QueenHandle
│
├── Day 3: daemon-contract (continued)
│   ├── Implement install types
│   ├── Implement lifecycle types
│   └── Add HiveHandle
│
├── Day 4: daemon-contract (testing)
│   ├── Write tests
│   ├── Update documentation
│   └── Verify migration
│
└── Day 5: ssh-contract
    ├── Create crate
    ├── Implement SshTarget
    ├── Remove duplication
    └── Test migration

Week 2:
├── Day 1: keeper-config-contract
│   ├── Create crate
│   ├── Implement KeeperConfig
│   ├── Add validation
│   └── Test migration
│
├── Day 2-3: Testing & Documentation
│   ├── Integration tests
│   ├── Update READMEs
│   └── Create migration guides
│
└── Day 4-5: Buffer
    └── Fix any issues
```

---

## 🎯 Quick Start

### For daemon-contract

1. Read `TEAM_314_DAEMON_CONTRACT_IMPLEMENTATION.md`
2. Run search commands from `TEAM_314_SEARCH_INSTRUCTIONS.md`
3. Create crate structure
4. Implement types (copy from sources)
5. Migrate consumers
6. Test

### For ssh-contract

1. Read `TEAM_314_SSH_CONTRACT_IMPLEMENTATION.md`
2. Run search commands
3. Create crate
4. Implement types
5. Remove duplicates
6. Test

### For keeper-config-contract

1. Read `TEAM_314_KEEPER_CONFIG_CONTRACT_IMPLEMENTATION.md`
2. Run search commands
3. Create crate
4. Implement config
5. Add validation
6. Test

---

## ✅ Success Criteria

### daemon-contract
- [ ] Generic `DaemonHandle` works for queen, hive, workers
- [ ] `QueenHandle` is type alias
- [ ] `HiveHandle` added
- [ ] All status, install, lifecycle types moved
- [ ] All tests pass
- [ ] No breaking changes

### ssh-contract
- [ ] `SshTarget` moved to contract
- [ ] Duplication removed
- [ ] All consumers updated
- [ ] All tests pass

### keeper-config-contract
- [ ] `KeeperConfig` moved to contract
- [ ] Validation added
- [ ] rbee-keeper uses contract
- [ ] All tests pass

---

## 📚 Related Documents

### Analysis Documents
- `TEAM_314_MISSING_CONTRACTS_ANALYSIS.md` - What's missing
- `TEAM_314_DAEMON_HANDLE_PROPOSAL.md` - Generic handle proposal

### Migration Documents
- `TEAM_314_SSH_CLIENT_MIGRATION.md` - SSH client to shared crate

---

## 🔧 Tools Required

### ripgrep (rg)
```bash
cargo install ripgrep
```

### fd (optional)
```bash
cargo install fd-find
```

### ast-grep (optional)
```bash
cargo install ast-grep
```

---

## 📖 Reading Order

**For Implementation:**
1. `TEAM_314_CONTRACT_IMPLEMENTATION_PLAN.md` (overview)
2. `TEAM_314_SEARCH_INSTRUCTIONS.md` (how to search)
3. `TEAM_314_DAEMON_CONTRACT_IMPLEMENTATION.md` (implement)
4. `TEAM_314_SSH_CONTRACT_IMPLEMENTATION.md` (implement)
5. `TEAM_314_KEEPER_CONFIG_CONTRACT_IMPLEMENTATION.md` (implement)

**For Understanding:**
1. `TEAM_314_MISSING_CONTRACTS_ANALYSIS.md` (what's missing)
2. `TEAM_314_DAEMON_HANDLE_PROPOSAL.md` (why generic handle)
3. `TEAM_314_CONTRACT_IMPLEMENTATION_PLAN.md` (how to fix)

---

## 🚀 Getting Started

```bash
# 1. Read the plan
cat TEAM_314_CONTRACT_IMPLEMENTATION_PLAN.md

# 2. Read search instructions
cat TEAM_314_SEARCH_INSTRUCTIONS.md

# 3. Start with daemon-contract
cat TEAM_314_DAEMON_CONTRACT_IMPLEMENTATION.md

# 4. Create the crate
cd /home/vince/Projects/llama-orch/bin/97_contracts
cargo new --lib daemon-contract

# 5. Follow the implementation guide
# ... (see daemon-contract implementation doc)
```

---

## 📞 Support

If you get stuck:

1. **Check search instructions** - Make sure you found all usages
2. **Check implementation guide** - Follow step-by-step
3. **Run tests** - Verify each step
4. **Check existing contracts** - Look at shared-contract for examples

---

## 📝 Summary

**Total Documents:** 6

**Implementation Guides:** 3
- daemon-contract (CRITICAL)
- ssh-contract (HIGH)
- keeper-config-contract (MEDIUM)

**Support Documents:** 3
- Main plan
- Missing contracts analysis
- Search instructions

**Estimated Time:** 2 weeks

**Priority Order:**
1. 🔴 daemon-contract (2-3 days)
2. 🟡 ssh-contract (1 day)
3. 🟢 keeper-config-contract (1 day)

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** INDEX 📚
