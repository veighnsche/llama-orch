# placement — Orchestrator inter-worker placement

This crate implements placement logic for `orchestratord`.

## Overview

- Consumes worker snapshots (Ready + Capacity) to decide where to route tasks.
- Prefers workers that already have the requested model resident.
- Applies fairness and tie-breaking to avoid hot-spotting and ensure starvation-freedom.

## Docs

- [FAIRNESS.md](FAIRNESS.md) — Inter-worker fairness & tie-breaking policy

## Inputs

- `GET /worker/ready`: resident handles per worker
- `GET /worker/capacity`: `draining`, `slots_total/free`, VRAM totals/used/free
- Optional: pool-managerd snapshots with host RAM and prefetch hints

## Notes

- Pin overrides: callers can target a specific worker/pool; eligibility still applies.
- Draining: misroutes fail fast with `503 ADMISSION_CLOSED` and should retry elsewhere.