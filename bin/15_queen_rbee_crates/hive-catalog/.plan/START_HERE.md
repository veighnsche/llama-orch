# Hive Catalog → File-Based Config Migration

**Mission:** Eliminate SQLite-based `hive-catalog` and replace with file-based configuration following Unix best practices.

## The Problem

Currently, hive registration uses SQLite (`hive-catalog` crate), which is:
- ❌ Over-engineered for simple config storage
- ❌ Requires programmatic editing via CLI
- ❌ Not transparent to users
- ❌ Doesn't follow standard Unix config patterns

## The Solution

Replace with **file-based configuration**:

```
~/.config/rbee/
├── config.toml          # Queen-level config (queen port, etc.)
├── hives.conf           # SSH/hive definitions (SSH config style)
└── capabilities.yaml    # Auto-generated device capabilities
```

## Key Changes

### Before
```bash
rbee hive install --id my-hive --ssh-host 192.168.1.100 --ssh-user admin
# Writes to SQLite database
```

### After
```bash
# 1. User manually edits ~/.config/rbee/hives.conf:
Host workstation
    HostName 192.168.1.100
    Port 22
    User admin
    HivePort 8081

# 2. Install using alias:
rbee hive install -h workstation
```

## Team Assignments

| Phase | Team | Focus | Duration |
|-------|------|-------|----------|
| Phase 1 | TEAM-188 | Config file design & parser | 4-6h |
| Phase 2 | TEAM-189 | Replace SQLite in job_router | 6-8h |
| Phase 3 | TEAM-190 | Preflight validation | 3-4h |
| Phase 4 | TEAM-191 | Capabilities auto-generation | 4-5h |
| Phase 5 | TEAM-192 | Code peer review | 2-3h |
| Phase 6 | TEAM-193 | Documentation | 2-3h |
| Phase 7 | TEAM-194 | Self-destruct (delete hive-catalog) | 1-2h |

## Success Criteria

- ✅ No SQLite dependencies
- ✅ Users manually edit `hives.conf`
- ✅ Alias-based hive operations
- ✅ Auto-generated capabilities cache
- ✅ Preflight validation on queen startup
- ✅ `hive-catalog` crate deleted

## Read Next

Start with **PHASE_1_TEAM_188.md** for config parser implementation.

---

**Created by:** TEAM-187  
**Date:** 2025-10-21  
**Status:** 🚀 Ready to start
