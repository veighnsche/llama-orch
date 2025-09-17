# DOCS_TODO — Home Profile Documentation Rebuild

> Evidence-backed checklist for rewriting every doc/spec/plan so the single-host home lab profile is the only product narrative. References below point to the current files in `HEAD` that still speak about “reductions”, multi-tenant/enterprise behavior, or rely on doc paths we just removed.

## 0. Baseline References
- `README_LLM.md` (home ruleset) points to `.docs/workflow.md`, `.docs/PROCESS.md`, `.specs/` — keep those files current when further edits land.
- `README.md` links to `.docs/` and `.specs/`; navigation verified after each doc change.
- `TODO.md` references `.docs/HOME_PROFILE.md`, `.docs/HOME_PROFILE_TARGET.md`, `.specs/00_home_profile.md` — all exist after this sweep.

## 1. `.docs/` Status
- ✅ `.docs/HOME_PROFILE.md`, `.docs/HOME_PROFILE_TARGET.md`, `.docs/workflow.md`, `.docs/PROJECT_GUIDE.md`, `.docs/session-policy.md`, `.docs/PROCESS.md` rewritten for home baseline.
- ✅ `.docs/testing/spec-derived-test-catalog.md`, `spec-combination-matrix.md`, `BDD_WIRING.md`, `TESTING_POLICY.md` refreshed; traceability harness has targets again.
- TODO: Review high-level philosophy docs (`.docs/testing/VIBE_CHECK.md`, `WHY_LLMS_ARE_STUPID.md`) and align wording with current policies.
- TODO: Decide whether to restore archived TODO history under `.docs/DONE/` from backup or keep existing files as-is.

## 2. `.specs/`
- ✅ `.specs/00_llama-orch.md` rewritten for home profile baseline.
- ✅ `.specs/00_home_profile.md` now overlays home-specific requirements without reduction language.
- ✅ `.specs/metrics/otel-prom.md` trimmed to minimal metric set (Active/Retired lifecycle, no fairness gauges).
- ✅ Requirements regenerated (`requirements/00_llama-orch.yaml`, `requirements/00_home_profile.yaml`).
- Progress: `.specs/30_pool-managerd.md`, `.specs/40-43*.md`, `.specs/60-config-schema.md`, `.specs/71-metrics-contract.md` spot-checked (no enterprise wording).
- TODO: Review remaining component docs (policy specs, determinism suite) and proposals for legacy WFQ/tenant references; mark superseded or update.

## 3. `.plan/`
- ✅ Updated plans: `00_meta_plan.md`, `00_llama-orch.md`, `20_orchestratord.md`, `30_pool_managerd.md`, `50_policy.md`, `60_config_schema.md`, `71_metrics_contract.md`, `72_bdd_harness.md`, `80_cli.md`, `code-distribution.md` — now aligned with home backlog.
- TODO: Review remaining plan files (`40_worker_adapters.md`, `90_tools.md`, etc.) for any lingering enterprise terminology.

## 4. Scripts & Harness
- ✅ `search_overkill.sh` retitled as audit helper (legacy scans remain to flag regressions).
- ✅ `cargo test -p test-harness-bdd traceability` (run after BDD docs refresh).
- TODO: After final doc/spec adjustments, run `bash ci/scripts/check_links.sh` and confirm all references resolve.

## 5. Validation Checklist (to run after the rewrite)
1. `cargo test -p test-harness-bdd traceability` — ensures the rebuilt spec/test catalogs reference every requirement ID.
2. `bash ci/scripts/check_links.sh` — validates that updated doc links resolve.
3. `cargo xtask spec-extract && git diff --exit-code requirements/` — regenerates requirement indexes from the new specs.
4. Manual spot-check: navigate via `README.md` and `README_LLM.md` to verify all links/tables still point to valid docs.
5. `ci/scripts/archive_todo.sh` — confirm it can archive `TODO.md` into the recreated `.docs/DONE/` directory.

## 6. Open Questions
- Restore archived TODO history under `.docs/DONE/` or start fresh?
- Are existing ORCH/OC IDs sufficient, or do we need renumbering once implementation catches up?
- When trimming metrics further, ensure `ci/metrics.lint.json` and docs stay in sync (still to audit after code changes).

> Update this TODO as you recreate documents. Include file references (path + line numbers) when you complete or adjust tasks so future reviewers can track the changes quickly.
