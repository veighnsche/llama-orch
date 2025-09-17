# DOCS_TODO — Home Profile Documentation Rebuild

> Evidence-backed checklist for rewriting every doc/spec/plan so the single-host home lab profile is the only product narrative. References below point to the current files in `HEAD` that still speak about “reductions”, multi-tenant/enterprise behavior, or rely on doc paths we just removed.

## 0. Baseline References
- `README_LLM.md` (home ruleset) points to `.docs/workflow.md`, `.docs/PROCESS.md`, `.specs/` — keep those files current when further edits land.
- `README.md` links to `.docs/` and `.specs/`; navigation verified after each doc change.
- `TODO.md` references `.docs/HOME_PROFILE.md`, `.docs/HOME_PROFILE_TARGET.md`, `.specs/00_home_profile.md` — all exist after this sweep.

## 1. `.docs/` Status
- ✅ `.docs/HOME_PROFILE.md`, `.docs/HOME_PROFILE_TARGET.md`, `.docs/workflow.md`, `.docs/PROJECT_GUIDE.md`, `.docs/session-policy.md`, `.docs/PROCESS.md` rewritten for home baseline.
- ✅ `.docs/testing/spec-derived-test-catalog.md`, `spec-combination-matrix.md`, `BDD_WIRING.md`, `TESTING_POLICY.md` refreshed; traceability harness has targets again.
- ✅ High-level philosophy docs (`.docs/testing/VIBE_CHECK.md#L1`, `.docs/testing/WHY_LLMS_ARE_STUPID.md#L1`) already align with current policies; repo scan shows no enterprise or tenant terminology.
- ✅ Archived TODO history confirmed under `.docs/DONE/TODO-0.md#L1`; keeping restored files as the canonical record.

## 2. `.specs/`
- ✅ `.specs/00_llama-orch.md` rewritten for home profile baseline.
- ✅ `.specs/00_home_profile.md` now overlays home-specific requirements without reduction language.
- ✅ `.specs/metrics/otel-prom.md` trimmed to minimal metric set (Active/Retired lifecycle, no fairness gauges).
- ✅ Requirements regenerated (`requirements/00_llama-orch.yaml`, `requirements/00_home_profile.yaml`).
- Progress: `.specs/30_pool-managerd.md`, `.specs/40-43*.md`, `.specs/60-config-schema.md`, `.specs/71-metrics-contract.md` spot-checked (no enterprise wording).
- ✅ Policy/determinism specs (`.specs/50-plugins-policy-host.md#L1`, `.specs/70-determinism-suite.md#L1`) and scheduling proposal (`.specs/proposals/2025-09-15-spec-v3.2-catalog-scheduling.md#L1`) read clean with home-profile language.

## 3. `.plan/`
- ✅ Updated plans: `00_meta_plan.md`, `00_llama-orch.md`, `20_orchestratord.md`, `30_pool_managerd.md`, `50_policy.md`, `60_config_schema.md`, `71_metrics_contract.md`, `72_bdd_harness.md`, `80_cli.md`, `code-distribution.md` — now aligned with home backlog.
- ✅ Remaining plan files (`.plan/40_worker_adapters.md#L1`, `.plan/90_tools.md#L1`) align with the home profile; no enterprise terminology remains.

## 4. Scripts & Harness
- ✅ `search_overkill.sh` retitled as audit helper (legacy scans remain to flag regressions).
- ✅ `cargo test -p test-harness-bdd traceability` (run after BDD docs refresh).
- ✅ Link audit run (`bash ci/scripts/check_links.sh`, script at `ci/scripts/check_links.sh#L1`) completed without errors.

## 5. Validation Checklist (to run after the rewrite)
1. `cargo test -p test-harness-bdd traceability` — ensures the rebuilt spec/test catalogs reference every requirement ID.
2. `bash ci/scripts/check_links.sh` — validates that updated doc links resolve.
3. `cargo xtask spec-extract && git diff --exit-code requirements/` — regenerates requirement indexes from the new specs.
4. Manual spot-check: navigate via `README.md` and `README_LLM.md` to verify all links/tables still point to valid docs.
5. `ci/scripts/archive_todo.sh` — confirm it can archive `TODO.md` into the recreated `.docs/DONE/` directory.

## 6. Open Questions
- Are existing ORCH/OC IDs sufficient, or do we need renumbering once implementation catches up?
- When trimming metrics further, ensure `ci/metrics.lint.json` and docs stay in sync (still to audit after code changes).

> Update this TODO as you recreate documents. Include file references (path + line numbers) when you complete or adjust tasks so future reviewers can track the changes quickly.
