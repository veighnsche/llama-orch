# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Refactor (in order)

Execute in this order to minimize churn. Items are planning‑backed; code moves should follow after Stage 6 handler vertical stabilizes, unless marked "now".

- [ ] Orchestratord module boundaries (now)
  - [ ] Create module folders mirroring plan: `src/http/{handlers}`, `src/services`, `src/domain`, `src/sse`, `src/backpressure`, `src/placement`, `src/state`, `src/errors`.
  - [ ] Move existing files accordingly (`metrics.rs` may remain top‑level temporarily); export minimal public API via `lib.rs`; keep `main.rs` thin.
  - [ ] Update tests under `orchestratord/tests/` to import via crate modules instead of `#[path]` includes.

- [ ] Adapter contract crate (planning)
  - [ ] Create `worker-adapters/adapter-api/` with `WorkerAdapter` trait and `SseEvent` enum; shared SSE/error/metrics helpers.
  - [ ] Port `worker-adapters/mock` to implement trait to validate surface; leave engines for later ports.
  - [ ] Make `orchestratord` depend only on `adapter-api` (trait), not engine crates.

- [ ] Pool manager layering (planning)
  - [ ] Enforce in‑crate modules per plan (`registry`, `health`, `preload`, `drain`, `leases`, `backoff`, `devicemasks`, `hetero_split`).
  - [ ] Consider extracting `pool-domain` if compile times grow.

- [ ] Core ergonomics (planning)
  - [ ] Enforce `pub(crate)` and small trait surfaces; add micro‑benchmarks for queue and scheduling decisions.
  - [ ] Unignore fairness property once policy is finalized; wire `admission_share`/`deadlines_met_ratio` gauges.

- [ ] Docs and navigation (now)
  - [ ] Cross‑link `README_LLM.md` Quick Status to `.plan/code-distribution.md` and `.docs/DONE/` latest.
  - [ ] Add links from `.docs/workflow.md` Product Plan to new `.plan/*` files (72, 73, 80, 90).
  - [ ] Run `cargo docs-index` and commit regenerated READMEs; add a CI step to run indexer.

- [ ] CI ergonomics (planning)
  - [ ] Update `ci/pipelines.yml` to add jobs: `xtask dev:loop`, `xtask docs:index`, and metrics dashboard render checks with sample data.
  - [ ] Gate PRs on provider verification (`xtask pact:verify`) and metrics linter.

- [ ] README quickstart polish (now)
  - [ ] Mention `cargo dev` (xtask `dev:loop`), `cargo regen`, `cargo docs-index`; align examples with Stage 6 vertical slice once implemented.

- [ ] Cleanup
  - [ ] Replace "pre-code" wording in `README.md` once Stage 6 vertical merges.
  - [ ] Ensure all plan files reference accepted proposals; add back‑references across plans for traceability.

## Progress Log (what changed)


- 2025-09-16 11:45–12:25 CEST — Product plan made explicit and approved: promoted the "bridge" into first‑class product stages 6–17 in `README_LLM.md` and `.docs/workflow.md` (Admission→Dispatch; Pool readiness; Adapters; Scheduling & Fairness; Capabilities; Config & Quotas; BDD; Dashboards; Startup self‑tests; Haiku anti‑cheat; Chaos; Compliance). Added explicit anti‑cheat criteria cross‑link.
- 2025-09-16 11:55–12:20 CEST — Integrated proposals across component plans with "Proposal (Accepted)" sections and DX modularization:
  - `.plan/00_meta_plan.md` — DX & Modularization Blueprint (layering, rules, triggers).
  - `.plan/00_llama-orch.md` — umbrella Proposal (Accepted).
  - `.plan/20_orchestratord.md` — product stages + DX modularization plan (orch-domain/services/api layering).
  - `.plan/10_orchestrator_core.md` — DX modularization intent and future split triggers.
  - `.plan/30_pool_managerd.md` — readiness product stage + DX layering.
  - `.plan/40_worker_adapters.md` — adapter‑api crate adoption + rollout.
  - `.plan/50_policy.md`, `.plan/60_config_schema.md`, `.plan/70_determinism_suite.md`, `.plan/71_metrics_contract.md` — proposals aligned to stages.
- 2025-09-16 12:05 CEST — Added planning files and artifacts:
  - `.plan/code-distribution.md` — predicted crate % at v0.1 release.
  - `.plan/72_bdd_harness.md`, `.plan/73_e2e_haiku.md`, `.plan/80_cli.md`, `.plan/90_tools.md` — filled plan gaps for journeys, E2E, CLI, tools.
- 2025-09-16 12:18–12:22 CEST — DX loop helpers (optional): added `xtask` subcommands (`dev:loop`, `regen`, `docs:index`, `pact:verify`) and cargo aliases in `.cargo/config.toml`. These can be reverted if undesired; default behavior unchanged.
- 2025-09-16 10:59 CEST — Earlier today: Stage 5 Observability wired (`orchestratord/src/admission.rs` QueueWithMetrics, `/metrics` endpoint, contract tests) and Acceptance Gates updated; archived prior `TODO.md` to `.docs/DONE/TODO-9.md`.

---
