# APOLOGY — On Skipping Explicit Product Stages Initially

I’m sorry for presenting the workflow in a way that under‑represented the ACTUAL PRODUCT work. That was a mistake.

## Why it happened
- The repo enforces Spec → Contract → Tests → Code discipline with hard gates. I front‑loaded scaffolding (specs, contracts, CDC, provider verify, properties, determinism, metrics) to keep us from shipping broken surfaces.
- But by compressing the “bridge” into a single lump (formerly “5.5”), I obscured the core application stages (admission → dispatch, pool manager, adapters, fairness, capabilities, quotas, journeys). That was poor communication.

## Why it was wrong
- The workflow must make PRODUCT work first‑class and visible. Hiding it behind a single in‑between label diminishes its scope and priority.
- The README and plans should let anyone (human or LLM) see instantly where we are and what’s left to ship.

## What I changed
- Promoted the product into dedicated stages with gates: Stages 6–17 in `README_LLM.md` and `.docs/workflow.md`:
  - 6 Admission → Dispatch
  - 7 Pool Manager Readiness
  - 8 Worker Adapters Conformance (llamacpp → vLLM → TGI → Triton)
  - 9 Scheduling & Fairness
  - 10 Capability Discovery
  - 11 Config & Quotas
  - 12 BDD Journeys
  - 13 Dashboards & Alerts
  - 14 Startup Self‑Tests
  - 15 Real‑Model E2E (Haiku, anti‑cheat)
  - 16 Chaos & Load (nightly)
  - 17 Compliance & Release
- Aligned per‑spec plans with these product stages:
  - `.plan/20_orchestratord.md`, `.plan/30_pool_managerd.md`, `.plan/40_worker_adapters.md` expanded with deliverables, tests, acceptance criteria.
- Added missing plan files to cover gaps:
  - `.plan/72_bdd_harness.md`, `.plan/73_e2e_haiku.md`, `.plan/80_cli.md`, `.plan/90_tools.md`.
- Added `.plan/code-distribution.md` to set expectations about where code will live at release.

## Going forward
- Stage 6–14 are now the active product backlog. I’ll keep the root `TODO.md` and `.plan/*` in lockstep so progress is visible and actionable.
- If priorities shift, I will update the stages (with IDs and gates) rather than bury them.

Thank you for calling this out. It improved the plan and made the product path explicit.
