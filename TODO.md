# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] Docs vs requirements outputs — adopt per-spec YAML files under `requirements/*.yaml` (e.g., `00_llama-orch.yaml`) and update references in `.docs/workflow.md` accordingly. Rationale: simpler, deterministic artifacts per spec; aggregated `index.yaml` is unnecessary at this time.
- [x] Add minimal Grafana dashboards placeholders under `ci/dashboards/` to satisfy Stage 5 gate in `.docs/workflow.md`.
- [ ] Confirm pacts cover all endpoints and status codes for OrchQueue v1 and sessions. Review `cli/consumer-tests/tests/orchqueue_pact.rs` and ensure `contracts/pacts/*.json` exercise 202/400/429/503/500 where applicable; SSE transcript shape for `GET /v1/tasks/{id}/stream`; 204 for cancel and delete.
- [ ] Optional: implement `xtask` helpers `pact:verify` and `pact:publish` to streamline dev loop (non-blocking).

## Progress Log (what changed)

2025-09-15 23:38 — Updated workflow doc to reference per-spec requirement files. Edited `.docs/workflow.md` tool section to: `spec-extract/ → requirements/*.yaml` and Stage 8 to say `requirements/*.yaml` instead of `index.yaml`.

2025-09-15 23:38 — Added Grafana dashboard placeholders to `ci/dashboards/`:
- `orchqueue_admission.json`
- `replica_load_slo.json`

