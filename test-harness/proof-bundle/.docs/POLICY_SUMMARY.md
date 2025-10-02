# Proof Bundle Policies — Quick Reference

**Last Updated**: 2025-10-02

---

## Critical Policies

### 1. ✅ Crate-Local Location (PB-001)

**REQUIRED**: Proof bundles MUST be inside each crate's `.proof_bundle/` directory.

```
✅ CORRECT: <crate>/.proof_bundle/<type>/<run-id>/
❌ WRONG:   /.proof_bundle/
```

**See**: `.specs/PB-001-proof-bundle-location.md`

---

### 2. ✅ Always Generate (Pass AND Fail) (PB-002)

**REQUIRED**: Proof bundles MUST be generated for ALL test runs.

```
✅ Generate on pass
✅ Generate on fail (MORE IMPORTANT)
✅ Generate on panic
✅ Generate on timeout
❌ NEVER skip on failure
```

**Why**: Failed tests are MORE important to document than passing tests.

**See**: `.specs/PB-002-always-generate-bundles.md`

---

## Quick Implementation

### Pattern: Always Generate

```rust
use proof_bundle::{ProofBundle, TestType};

#[test]
fn my_test() -> anyhow::Result<()> {
    // ALWAYS create proof bundle FIRST
    let pb = ProofBundle::for_type(TestType::Unit)?;
    
    // Run test (may fail)
    let result = run_test();
    
    // ALWAYS capture result (pass or fail)
    pb.append_ndjson("results", &serde_json::json!({
        "status": if result.is_ok() { "pass" } else { "fail" },
        "error": result.as_ref().err().map(|e| format!("{:?}", e)),
    }))?;
    
    result
}
```

### CI/CD: Always Upload

```yaml
- name: Run Tests
  run: cargo test -p my-crate
  continue-on-error: true

- name: Upload Proof Bundle (ALWAYS)
  if: always()  # Upload even if tests failed
  uses: actions/upload-artifact@v3
  with:
    name: proof-bundle
    path: my-crate/.proof_bundle/
```

---

## Enforcement

**Both policies are NORMATIVE and REQUIRED.**

Violations block PR merge.

---

## Proof Bundle Team Responsibilities

The proof-bundle team is responsible for:

1. ✅ **Auditing test coverage** across all crates
2. ✅ **Ensuring proof bundle quality** (information-rich, human-auditable)
3. ✅ **Maintaining specifications** (PB-001, PB-002, etc.)
4. ✅ **Enforcing consistency** across crates
5. ✅ **Advocating for human auditors** (not just machines)
6. ✅ **Continuous improvement** of proof bundle system

**See**: `RESPONSIBILITIES.md` for full details.

---

## References

- **Specifications**: `.specs/` directory
- **Library Docs**: `README.md`
- **Responsibilities**: `RESPONSIBILITIES.md`
- **Root Overview**: `/.proof_bundle/README.md`

---

**Status**: ACTIVE  
**Enforcement**: REQUIRED
