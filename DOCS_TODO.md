# DOCS_TODO — Home Profile Documentation Rebuild

> Evidence-backed checklist for rewriting every doc/spec/plan so the single-host home lab profile is the only product narrative. References below point to the current files in `HEAD` that still speak about “reductions”, multi-tenant/enterprise behavior, or rely on doc paths we just removed.

## 0. Baseline References
- `README_LLM.md:11-173` defines the governing workflow and points contributors to `.docs/workflow.md`, `.specs/`, and `.plan/`; all rebuilt docs must stay consistent.
- `README.md:8-9` links to `.docs/` and `.specs/`; navigation must remain valid once the new docs are in place.
- `TODO.md:147-154` expects `.docs/HOME_PROFILE.md`, `.specs/00_home_profile.md`, `.docs/HOME_PROFILE_TARGET.md` to exist.

## 1. `.docs/` Rebuild
- `.docs/HOME_PROFILE.md:2-3,85-109` still frames v2.1 as a “reductive rewrite of the production SPEC” with a “differences vs production” appendix. Rewrite it so the home profile is the baseline product story (no references to reductions or production profiles).
- `.docs/HOME_PROFILE_TARGET.md:20` hardcodes “Home Profile v2.1”; adjust once the new spec/version naming is settled and ensure the validation target matches the rewritten docs.
- `.docs/FULL_PROCESS_SPEC_CHANGE.md` (entire file) remains accurate but references `.docs/testing/*`; keep the flow but rename and restyle it as the canonical home-profile `PROCESS.md` doc.
- `.docs/workflow.md:59-61` lists enterprise personas (Platform Engineer, ML Engineer, SRE) and the document calls out fairness, quotas, dashboards, policy plugins, etc. (lines 117-120). Rework personas and stage descriptions so they reference the home lab story while preserving the testing gates.
- `.docs/workflow.md:125-158` still mandates fairness gauges, preemption counters, capability discovery for clients, quotas, etc. Reconcile with the simplified metric/scheduling story we want for home use.
- `.docs/PROJECT_GUIDE.md:2` points newcomers to legacy paths (`specs/orchestrator-spec.md`, `docs/workflow.md`). Update it after the new files are in place.
- `.docs/session-policy.md:1-6` is concise but must be rechecked against the final session/budget requirements.
- `.docs/testing/spec-derived-test-catalog.md` and `.docs/testing/spec-combination-matrix.md` are still referenced by `test-harness/bdd/tests/traceability.rs:12-25`; regenerate/rewrite them once specs are updated so the traceability test passes again.
- `test-harness/bdd/tests/traceability.rs:12-25` and `ci/scripts/archive_todo.sh:9-10` expect `.docs/testing/...` and `.docs/DONE/` to exist; when rebuilding, recreate these directories and ensure archived TODOs are restored or stubbed.
- `README_LLM.md:145-155` and `README.md:53-55` mention `.docs/DONE/` and `.docs/testing/`; keep those references alive.

## 2. `.specs/` Rebuild
- `.specs/00_llama-orch.md:1-15` still describes “multi-tenant orchestration” with fairness, quotas, and enterprise resilience goals. Rewrite the scope/goals for single-host home lab and remove enterprise references while keeping the requirement IDs.
- `.specs/00_home_profile.md:1-63` frames v2.1 as keeping catalog/drain/artifacts “relative to v1 reduction.” Replace that narrative with a clean description of the baseline home profile (no comparisons to reductions).
- `.specs/metrics/otel-prom.md:55-83` enumerates metrics (`model_state` with Draft/Canary/Deprecated, `admission_share`, `deadlines_met_ratio`, `preemptions_total`, `resumptions_total`) that contradict the slim home profile. Decide which metrics remain and update the contract accordingly.
- `.specs/10-orchestrator-core.md`, `.specs/20-orchestratord.md`, `.specs/30-pool-managerd.md`, `.specs/40-43*.md`, `.specs/50-51*.md`, `.specs/60-config-schema.md`, `.specs/70-determinism-suite.md`, `.specs/71-metrics-contract.md`: review each component spec to ensure they no longer reference multi-tenant fairness, quotas, or enterprise-only knobs. (Use `git show HEAD:.specs/<file>` for details; many still cite fairness/preemption requirements.)
- `.specs/metrics/` and `.specs/proposals/` directories: confirm the necessary appendices/templates exist with the new terminology.
- `requirements/*.yaml` (e.g., `requirements/00_llama-orch.yaml`) still mirror enterprise language—regenerate them after rewriting the specs.

## 3. `.plan/` Rebuild
- `.plan/20_orchestratord.md:28-33` schedules “Stage 9 — Scheduling & fairness” and “Stage 11 — Config & quotas.” Adjust the plan to match the simplified home backlog while keeping the test gates that remain relevant (admission, SSE, catalog, artifacts, capability discovery).
- `.plan/00_llama-orch.md`, `.plan/30_pool_managerd.md`, `.plan/40_worker_adapters.md`, `.plan/50_policy.md`, `.plan/60_config_schema.md`, `.plan/70_determinism_suite.md`, `.plan/71_metrics_contract.md`, `.plan/80_cli.md`, `.plan/90_tools.md`, and `.plan/code-distribution.md` all reference current specs—rewrite each to match the new docs and home-lab focus.

## 4. Scripts and Tests to Update After Rebuild
- `search_overkill.sh:2-47` still announces itself as a “Home Profile reduction helper” and scans for fairness/preemption removal. Once the new docs/specs are published, rewrite or replace the script so it reflects the home profile baseline (and only scans for concepts we truly plan to remove).
- `test-harness/bdd/tests/traceability.rs` should pass once the spec/test catalog docs are reinstated; verify after regenerating docs.
- `ci/scripts/check_links.sh` and `ci/scripts/archive_todo.sh` rely on stable doc paths; rerun after files are recreated.

## 5. Validation Checklist (to run after the rewrite)
1. `cargo test -p test-harness-bdd traceability` — ensures the rebuilt spec/test catalogs reference every requirement ID.
2. `bash ci/scripts/check_links.sh` — validates that updated doc links resolve.
3. `cargo xtask spec-extract && git diff --exit-code requirements/` — regenerates requirement indexes from the new specs.
4. Manual spot-check: navigate via `README.md` and `README_LLM.md` to verify all links/tables still point to valid docs.
5. `ci/scripts/archive_todo.sh` — confirm it can archive `TODO.md` into the recreated `.docs/DONE/` directory.

## 6. Open Questions
- Should we restore the archived TODO history under `.docs/DONE/` from backup, or start fresh with a new numbering scheme?
- Do we keep the existing requirement IDs (ORCH-/OC- prefixes) or renumber them for the home profile? Renumbering ripples into code/tests/requirements YAML.
- Which of the “advanced metrics” in `.specs/metrics/otel-prom.md:55-83` survive in the home profile? Decide before updating `ci/metrics.lint.json` and the metrics spec.

> Update this TODO as you recreate documents. Include file references (path + line numbers) when you complete or adjust tasks so future reviewers can track the changes quickly.
