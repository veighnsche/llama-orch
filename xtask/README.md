# xtask — Workspace Utility Tasks

**TEAM-111** - Cleaned up and modernized  
**Status:** Active

---

## 🎯 Purpose

Developer tooling for the llama-orch workspace. Provides commands for:
- Regenerating contracts and schemas
- Running CI checks
- Managing development workflow
- Engine management

---

## 📋 Available Commands

### Regeneration Tasks
```bash
cargo xtask regen                # Regenerate all artifacts
cargo xtask regen-openapi        # Regenerate OpenAPI types
cargo xtask regen-schema         # Regenerate config schema
cargo xtask spec-extract         # Extract spec requirements
```

### Development Tasks
```bash
cargo xtask dev:loop             # Full dev workflow (fmt, clippy, regen, test, links)
cargo xtask docs:index           # Regenerate README index
```

### CI Tasks
```bash
cargo xtask ci:auth              # Test auth-min crate
cargo xtask ci:determinism       # Run determinism suite
cargo xtask ci:haiku:cpu         # Run haiku e2e tests
```

### Pact Tasks
```bash
cargo xtask pact:verify          # Verify pact contracts
```

### Engine Tasks
```bash
cargo xtask engine:status        # Check engine status
cargo xtask engine:down          # Stop engines
```

### BDD Test Tasks (NEW!)
```bash
cargo xtask bdd:test             # Run all BDD tests (live output)
cargo xtask bdd:test --quiet     # Run quietly (summary only)
cargo xtask bdd:test --tags @auth  # Run tests with specific tag
cargo xtask bdd:test --feature lifecycle  # Run specific feature
```

---

## 🔧 Common Workflows

### Daily Development
```bash
# Before committing
cargo xtask dev:loop
```

### After Changing Contracts
```bash
# Regenerate OpenAPI types
cargo xtask regen-openapi

# Regenerate config schema
cargo xtask regen-schema

# Or regenerate everything
cargo xtask regen
```

### Running Tests
```bash
# Run BDD tests
cargo xtask bdd:test

# Run BDD tests with tag filter
cargo xtask bdd:test --tags @p0

# Run determinism tests
cargo xtask ci:determinism

# Run haiku e2e tests
cargo xtask ci:haiku:cpu

# Verify pact contracts
cargo xtask pact:verify
```

---

## 🧹 Recent Cleanup (TEAM-111)

**Removed deprecated/stub commands:**
- ❌ `ci:haiku:gpu` - Never implemented
- ❌ `pact:publish` - Never implemented
- ❌ `engine:plan` - Never implemented
- ❌ `engine:up` - Never implemented

**Kept working commands:**
- ✅ All regeneration tasks
- ✅ Development workflow tasks
- ✅ CI test tasks
- ✅ Working engine tasks (status, down)

See `.archive/CLEANUP_PLAN.md` for details.

---

## 📚 Documentation

- Spec: [.specs/00_llama-orch.md](../.specs/00_llama-orch.md)
- Requirements: [requirements/00_llama-orch.yaml](../requirements/00_llama-orch.yaml)
- Cleanup Plan: [.archive/CLEANUP_PLAN.md](.archive/CLEANUP_PLAN.md)

---

## 🚀 Recent Additions (TEAM-111)

**NEW: BDD Test Runner** ✅
- `bdd:test` - Run BDD tests with live output
- `--quiet` flag - Run quietly with spinner
- `--tags` flag - Filter by tags
- `--feature` flag - Run specific feature

**Features:**
- ✅ Live output streaming (default)
- ✅ Quiet mode with progress spinner
- ✅ Failure-focused reporting
- ✅ Auto-generated rerun commands
- ✅ Timestamped log files
- ✅ Comprehensive error handling

---

**Status:** Active - Cleaned and ready for new tasks  
**Owners:** @llama-orch-maintainers
