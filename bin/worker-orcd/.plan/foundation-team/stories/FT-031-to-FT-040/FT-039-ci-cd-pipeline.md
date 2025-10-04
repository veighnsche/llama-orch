# FT-039: CI/CD Pipeline

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 72 - 73  
**Spec Ref**: CI/CD requirements

---

## Story Description

Establish CI/CD pipeline for worker-orcd: automated builds, tests, and deployment. Includes CUDA feature flag handling and multi-GPU test support.

---

## Acceptance Criteria

- [ ] GitHub Actions workflow for CI
- [ ] Automated builds (with/without CUDA)
- [ ] Automated test execution
- [ ] Code coverage reporting
- [ ] Clippy and rustfmt checks
- [ ] Docker image builds
- [ ] Deployment automation
- [ ] Badge in README

---

## Dependencies

**Upstream**: FT-038 (Gate 3, Day 71)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Technical Details

```yaml
# .github/workflows/worker-orcd-ci.yml
name: worker-orcd CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: cargo build --features cuda
      - name: Test
        run: cargo test --features cuda
      - name: Clippy
        run: cargo clippy --features cuda
```

---

## Definition of Done

- [ ] CI pipeline operational
- [ ] All checks passing
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **CI pipeline started**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "ci_start",
       target: format!("commit-{}", commit_sha),
       human: format!("CI pipeline started for commit {}", commit_sha),
       ..Default::default()
   });
   ```

2. **CI pipeline passed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "ci_complete",
       target: format!("commit-{}", commit_sha),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("CI pipeline PASSED for commit {} ({} ms)", commit_sha, elapsed.as_millis()),
       ..Default::default()
   });
   ```

3. **CI pipeline failed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "ci_complete",
       target: format!("commit-{}", commit_sha),
       error_kind: Some("ci_failed".to_string()),
       human: format!("CI pipeline FAILED for commit {}: {}", commit_sha, failure_reason),
       ..Default::default()
   });
   ```

**Why this matters**: CI/CD pipeline ensures quality. Narration creates audit trail of builds and deployments.

**Note**: This is infrastructure. Narration primarily for CI/CD tracking, not runtime.

---
*Narration guidance added by Narration-Core Team 🎀*
