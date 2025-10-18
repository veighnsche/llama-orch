# xtask Cleanup & Modernization Plan

**TEAM-111** - Cleaning up old tasks  
**Date:** 2025-10-18

---

## 🔍 Current State Analysis

### Tasks Currently Defined
1. ✅ **regen-openapi** - Used in CI, docs
2. ✅ **regen-schema** - Used in CI, docs
3. ✅ **regen** - Runs both above + spec-extract
4. ✅ **spec-extract** - Runs tools-spec-extract
5. ✅ **dev:loop** - Used in CI, README (fmt, clippy, regen, tests, link check)
6. ✅ **docs:index** - Used in CI
7. ❌ **ci:haiku:gpu** - STUB (not implemented)
8. ✅ **ci:haiku:cpu** - Runs e2e haiku tests
9. ✅ **ci:determinism** - Runs determinism suite
10. ✅ **ci:auth** - Self-check for auth-min crate
11. ✅ **pact:verify** - Runs pact provider verification
12. ❌ **pact:publish** - STUB (not implemented)
13. ❌ **engine:plan** - STUB (not implemented)
14. ❌ **engine:up** - STUB (not implemented)
15. ✅ **engine:status** - Works (checks health, reads PID files)
16. ✅ **engine:down** - Works (kills processes)

### Usage Analysis

**Actively Used (in CI/docs):**
- `regen-openapi`
- `regen-schema`
- `dev:loop`
- `docs:index`

**Potentially Useful:**
- `spec-extract`
- `ci:haiku:cpu`
- `ci:determinism`
- `ci:auth`
- `pact:verify`
- `engine:status`
- `engine:down`

**DEPRECATED/STUBS:**
- `ci:haiku:gpu` - stub
- `pact:publish` - stub
- `engine:plan` - stub
- `engine:up` - stub

---

## 🎯 Cleanup Actions

### Phase 1: Remove Dead Code
- [ ] Remove `ci:haiku:gpu` (stub)
- [ ] Remove `pact:publish` (stub)
- [ ] Remove `engine:plan` (stub)
- [ ] Remove `engine:up` (stub)

### Phase 2: Consolidate
- [ ] Keep `regen` as umbrella command
- [ ] Keep `dev:loop` as main dev workflow
- [ ] Keep working CI tasks
- [ ] Keep working engine tasks (status, down)

### Phase 3: Add New Task
- [ ] Add `bdd:test` - Run BDD tests (port bash script)

---

## 📋 Recommended Structure

### Core Tasks (Keep)
```
regen               - Regenerate all artifacts
regen-openapi       - Regenerate OpenAPI types
regen-schema        - Regenerate config schema
spec-extract        - Extract spec requirements
```

### Development Tasks (Keep)
```
dev:loop            - Full dev workflow (fmt, clippy, regen, test, links)
docs:index          - Regenerate README index
```

### CI Tasks (Keep)
```
ci:auth             - Test auth-min crate
ci:determinism      - Run determinism suite
ci:haiku:cpu        - Run haiku e2e tests
```

### Pact Tasks (Keep)
```
pact:verify         - Verify pact contracts
```

### Engine Tasks (Keep)
```
engine:status       - Check engine status
engine:down         - Stop engines
```

### New Tasks (Add)
```
bdd:test            - Run BDD tests
bdd:test-quiet      - Run BDD tests (quiet mode)
bdd:test-tags       - Run BDD tests with tags
```

---

## 🗑️ To Remove

1. **ci:haiku:gpu** - Never implemented, no GPU tests exist
2. **pact:publish** - Never implemented, not in use
3. **engine:plan** - Never implemented, unclear purpose
4. **engine:up** - Never implemented, unclear purpose

---

## ✅ Final Task List

After cleanup, we should have:

### Regeneration (4 tasks)
- `regen` - All regeneration
- `regen-openapi` - OpenAPI types
- `regen-schema` - Config schema
- `spec-extract` - Spec requirements

### Development (2 tasks)
- `dev:loop` - Full dev workflow
- `docs:index` - README index

### CI (3 tasks)
- `ci:auth` - Auth tests
- `ci:determinism` - Determinism tests
- `ci:haiku:cpu` - Haiku e2e tests

### Pact (1 task)
- `pact:verify` - Contract verification

### Engine (2 tasks)
- `engine:status` - Check status
- `engine:down` - Stop engines

### BDD (3 tasks - NEW)
- `bdd:test` - Run BDD tests
- `bdd:test-quiet` - Run quietly
- `bdd:test-tags` - Run with tag filter

**Total: 15 tasks** (down from 16, but more useful)

---

## 🚀 Next Steps

1. Clean up xtask (remove stubs)
2. Port BDD bash script to Rust xtask
3. Update documentation
4. Update CI pipelines if needed
