# Proof Bundle Policy Summary

**Date**: 2025-10-02  
**Status**: NORMATIVE

---

## Critical Rule: Crate-Local Proof Bundles

### ✅ REQUIRED

Proof bundles MUST be generated inside each crate's `.proof_bundle/` directory:

```
<crate-root>/.proof_bundle/<test-type>/<run-id>/
```

### ❌ FORBIDDEN

Proof bundles MUST NOT be generated in the repository root:

```
/.proof_bundle/  ← NEVER
```

---

## Examples

### Correct Locations

```
bin/worker-orcd-crates/vram-residency/.proof_bundle/unit/20251002-123000-abc123/
bin/worker-orcd-crates/vram-residency/.proof_bundle/bdd/20251002-123000-abc123/
bin/pool-managerd/.proof_bundle/unit/20251002-123000-abc123/
test-harness/bdd/.proof_bundle/bdd/20251002-123000-abc123/
test-harness/chaos/.proof_bundle/chaos/20251002-123000-abc123/
```

### Incorrect Locations

```
/.proof_bundle/unit/...  ← WRONG
/.proof_bundle/bdd/...   ← WRONG
```

---

## Implementation

### Default Behavior

```rust
use proof_bundle::{ProofBundle, TestType};

// Automatically resolves to: <crate-root>/.proof_bundle/unit/<run-id>/
let pb = ProofBundle::for_type(TestType::Unit)?;
```

### Resolution Logic

1. Get crate root via `CARGO_MANIFEST_DIR`
2. Create `.proof_bundle/` subdirectory in crate root
3. Create test type subdirectory (e.g., `unit/`, `bdd/`)
4. Create run ID subdirectory

---

## Enforcement

### CI/CD Check

```bash
# Verify no root-level proof bundles
if [ -d ".proof_bundle" ]; then
    echo "ERROR: Proof bundles must be crate-local"
    exit 1
fi
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
if [ -d ".proof_bundle" ]; then
    echo "ERROR: See test-harness/proof-bundle/.specs/PB-001-proof-bundle-location.md"
    exit 1
fi
```

---

## Rationale

1. **Crate isolation** — Each crate owns its proof bundles
2. **Clear ownership** — No ambiguity about which crate generated what
3. **No conflicts** — Crates don't interfere with each other
4. **Easy cleanup** — Can clean per-crate
5. **CI/CD friendly** — Per-crate artifact upload

---

## References

- **Full Spec**: [PB-001-proof-bundle-location.md](./PB-001-proof-bundle-location.md)
- **Library Docs**: `test-harness/proof-bundle/README.md`
- **System Overview**: `/.proof_bundle/README.md`

---

**Status**: ✅ NORMATIVE  
**Enforcement**: REQUIRED  
**Violations**: Block PR merge
