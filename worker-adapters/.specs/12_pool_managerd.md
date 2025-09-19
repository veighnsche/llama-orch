# Wiring: worker-adapters ↔ pool-managerd

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. Adapters are used by `orchestratord` for the data plane. `pool-managerd` supervises engine processes and publishes readiness/capacity; it does not call adapters directly.

## Expectations on pool-managerd
- Provide accurate readiness/capacity/health in its registry for `orchestratord` to decide dispatch via adapters.
- Manage engine lifecycle (start/stop, backoff) separately from adapter usage.

## Expectations on adapters
- Expose `health/props` that reflect the underlying engine; these signals are consumed by `orchestratord` and surfaced in `/v1/pools/{id}/health` responses.

## Data Flow
- `pool-managerd` → registry snapshot → `orchestratord` → chooses adapter → adapter talks to engine API.

## Refinement Opportunities
- Optional direct health hint channel between manager and adapters for improved fidelity (documented handshake if added later).
