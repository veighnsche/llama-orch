# PB-001: Proof Bundle Location Policy

**Status**: Normative  
**Created**: 2025-10-02  
**Updated**: 2025-10-02

---

## Requirement

**CRITICAL**: Proof bundles MUST be generated inside each crate's `.proof_bundle/` directory, NOT in the repository root.

---

## Policy

### ✅ CORRECT: Crate-Local Proof Bundles

Each crate that generates proof bundles MUST place them in:

```
<crate-root>/.proof_bundle/<test-type>/<run-id>/
```

**Examples**:

```
bin/worker-orcd-crates/vram-residency/.proof_bundle/unit/20251002-123000-abc123/
bin/worker-orcd-crates/vram-residency/.proof_bundle/bdd/20251002-123000-abc123/
test-harness/bdd/.proof_bundle/bdd/20251002-123000-abc123/
test-harness/chaos/.proof_bundle/chaos/20251002-123000-abc123/
```

### ❌ INCORRECT: Root-Level Proof Bundles

**NEVER** place proof bundles in the repository root:

```
# ❌ WRONG
/.proof_bundle/unit/...
/.proof_bundle/bdd/...
```

---

## Rationale

### 1. Crate Isolation

Each crate's proof bundles are isolated:
- ✅ Easy to find (always in crate's `.proof_bundle/`)
- ✅ Clear ownership (crate owns its proof bundles)
- ✅ No conflicts between crates
- ✅ Can be cleaned per-crate

### 2. Repository Organization

Root-level proof bundles would:
- ❌ Clutter repository root
- ❌ Mix evidence from different crates
- ❌ Make it unclear which crate generated what
- ❌ Complicate cleanup and maintenance

### 3. CI/CD Integration

Crate-local proof bundles enable:
- ✅ Per-crate artifact upload
- ✅ Parallel test execution
- ✅ Selective proof bundle generation
- ✅ Clear audit trail per crate

---

## Implementation

### Default Behavior

`ProofBundle::for_type()` automatically resolves to crate-local directory:

```rust
use proof_bundle::{ProofBundle, TestType};

// Automatically creates: <crate-root>/.proof_bundle/unit/<run-id>/
let pb = ProofBundle::for_type(TestType::Unit)?;
```

**Resolution logic**:
1. Get crate root via `CARGO_MANIFEST_DIR`
2. Create `.proof_bundle/` subdirectory
3. Create test type subdirectory (e.g., `unit/`)
4. Create run ID subdirectory (e.g., `20251002-123000-abc123/`)

### Environment Override

If needed, override with `LLORCH_PROOF_DIR`:

```bash
# Override to custom location (use with caution)
LLORCH_PROOF_DIR=/tmp/proof cargo test
```

**Warning**: Only use overrides for debugging. Production CI/CD should use default crate-local paths.

---

## Examples

### Example 1: vram-residency Unit Tests

```rust
// File: bin/worker-orcd-crates/vram-residency/tests/proof_bundle_generator.rs

use proof_bundle::{ProofBundle, TestType};

#[test]
fn generate_proof_bundle() -> anyhow::Result<()> {
    // Creates: bin/worker-orcd-crates/vram-residency/.proof_bundle/unit/<run-id>/
    let pb = ProofBundle::for_type(TestType::Unit)?;
    
    pb.write_json("metadata", &serde_json::json!({
        "crate": "vram-residency",
        "test_type": "unit",
    }))?;
    
    Ok(())
}
```

**Result**:
```
bin/worker-orcd-crates/vram-residency/.proof_bundle/unit/20251002-123000-abc123/
├── metadata.json
└── ...
```

### Example 2: vram-residency BDD Tests

```rust
// File: bin/worker-orcd-crates/vram-residency/bdd/src/main.rs

use proof_bundle::{ProofBundle, TestType};

#[tokio::main]
async fn main() {
    // Creates: bin/worker-orcd-crates/vram-residency/.proof_bundle/bdd/<run-id>/
    let pb = ProofBundle::for_type(TestType::Bdd)?;
    
    // Run BDD tests...
    
    pb.write_json("metadata", &serde_json::json!({
        "crate": "vram-residency",
        "test_type": "bdd",
        "features": 10,
    }))?;
}
```

**Result**:
```
bin/worker-orcd-crates/vram-residency/.proof_bundle/bdd/20251002-123000-abc123/
├── metadata.json
├── scenarios.ndjson
└── ...
```

### Example 3: test-harness/bdd

```rust
// File: test-harness/bdd/tests/integration_test.rs

use proof_bundle::{ProofBundle, TestType};

#[test]
fn generate_bdd_proof_bundle() -> anyhow::Result<()> {
    // Creates: test-harness/bdd/.proof_bundle/bdd/<run-id>/
    let pb = ProofBundle::for_type(TestType::Bdd)?;
    
    pb.write_json("metadata", &serde_json::json!({
        "crate": "test-harness-bdd",
        "test_type": "bdd",
    }))?;
    
    Ok(())
}
```

**Result**:
```
test-harness/bdd/.proof_bundle/bdd/20251002-123000-abc123/
├── metadata.json
└── ...
```

---

## Verification

### Check Proof Bundle Location

```bash
# Find all proof bundles in repository
find . -type d -name ".proof_bundle"

# Expected output (examples):
# ./bin/worker-orcd-crates/vram-residency/.proof_bundle
# ./test-harness/bdd/.proof_bundle
# ./test-harness/chaos/.proof_bundle

# ❌ Should NOT see:
# ./.proof_bundle  (root-level)
```

### Verify Crate-Local

```bash
# For a specific crate
cd bin/worker-orcd-crates/vram-residency
ls -la .proof_bundle/

# Should show:
# .proof_bundle/unit/
# .proof_bundle/bdd/
```

---

## CI/CD Integration

### Upload Crate-Specific Artifacts

```yaml
# .github/workflows/vram-residency-ci.yml

- name: Run Tests
  run: cargo test -p vram-residency

- name: Upload Proof Bundle
  uses: actions/upload-artifact@v3
  with:
    name: vram-residency-proof-bundle
    path: bin/worker-orcd-crates/vram-residency/.proof_bundle/
```

### Parallel Test Execution

```yaml
# Each crate generates its own proof bundle independently
jobs:
  test-vram-residency:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test -p vram-residency
      - uses: actions/upload-artifact@v3
        with:
          name: vram-residency-proof
          path: bin/worker-orcd-crates/vram-residency/.proof_bundle/
  
  test-pool-managerd:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test -p pool-managerd
      - uses: actions/upload-artifact@v3
        with:
          name: pool-managerd-proof
          path: bin/pool-managerd/.proof_bundle/
```

---

## Migration Guide

### If You Have Root-Level Proof Bundles

**Step 1**: Identify root-level proof bundles

```bash
ls -la .proof_bundle/
```

**Step 2**: Move to crate-local directories

```bash
# Example: Move vram-residency proof bundles
mv .proof_bundle/unit/* bin/worker-orcd-crates/vram-residency/.proof_bundle/unit/
mv .proof_bundle/bdd/* bin/worker-orcd-crates/vram-residency/.proof_bundle/bdd/
```

**Step 3**: Remove root-level directory

```bash
rm -rf .proof_bundle/
```

**Step 4**: Update test code to use crate-local paths

```rust
// Old (wrong):
// let pb = ProofBundle::new("/path/to/root/.proof_bundle")?;

// New (correct):
let pb = ProofBundle::for_type(TestType::Unit)?;  // Auto-resolves to crate-local
```

---

## Enforcement

### .gitignore

Each crate should have `.proof_bundle/` in its `.gitignore`:

```gitignore
# bin/worker-orcd-crates/vram-residency/.gitignore
.proof_bundle/
```

**Note**: The root `.gitignore` should NOT have `.proof_bundle/` since bundles are crate-local.

### Pre-commit Hook (Optional)

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for root-level proof bundles
if [ -d ".proof_bundle" ]; then
    echo "ERROR: Proof bundles must be crate-local, not in repository root"
    echo "See: test-harness/proof-bundle/.specs/PB-001-proof-bundle-location.md"
    exit 1
fi
```

---

## Exceptions

### None

There are **NO exceptions** to this policy. All proof bundles MUST be crate-local.

If you believe you need an exception, discuss with the team first.

---

## Refinement Opportunities

1. **Automated verification**: Add CI check to ensure no root-level `.proof_bundle/` exists
2. **Documentation**: Add proof bundle location to crate README templates
3. **Tooling**: Create `xtask` command to verify proof bundle locations
4. **Monitoring**: Track proof bundle sizes per-crate to detect bloat

---

## References

- Root `.proof_bundle/README.md` — Overview of proof bundle system
- `test-harness/proof-bundle/README.md` — Library documentation
- `.docs/testing/TEST_TYPES_GUIDE.md` — Test type definitions

---

**Status**: ✅ NORMATIVE  
**Enforcement**: REQUIRED for all crates  
**Violations**: Block PR merge
