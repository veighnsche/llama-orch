# Worker Adapters — Shared Overview

This directory contains engine adapters that implement a shared `WorkerAdapter` trait to bridge orchestrator requests to engine‑native APIs (llama.cpp, vLLM, TGI, Triton, OpenAI, etc.).

- Central spec: `worker-adapters/.specs/00_worker_adapters.md`
- Contracts: `worker-adapters/.specs/10_contracts.md`
- Each adapter provides a slim, adapter‑specific README and `.specs/00_<adapter>.md` that documents deviations/quirks.

Adapters currently in this workspace:
- `adapter-api` — shared trait/types
- `llamacpp-http` — stub implementation mapping llama.cpp HTTP
- `vllm-http` — stub
- `tgi-http` — stub
- `triton` — stub
- `mock` — test stub
- `openai-http` — new online adapter (stubbed; see its README)

Testing policy
- Per‑crate unit/behavior tests live in each adapter crate.
- Root BDD harness validates cross‑crate flows; per‑adapter BDD (if any) should remain slim.

See `.specs/` for detailed behavior and contracts shared across adapters.

## Detailed behavior (High / Mid / Low)

- High-level
  - Each adapter implements a common `WorkerAdapter` trait to provide `health`, `props`, `submit`, `cancel`, and `engine_version`. Adapters translate `contracts/api-types::TaskRequest` into engine‑native HTTP/gRPC calls and stream tokens back while preserving `started → token* → end` ordering.

- Mid-level
  - HTTP-based adapters use `worker-adapters/http-util` for a shared `reqwest` client with timeouts, HTTP/2 keep‑alive, capped+jittered retries, streaming decode helpers, and header redaction. Errors are mapped to a shared `WorkerError` taxonomy and logs include fields from `README_LLM.md`.
  - Integration via the in‑process `adapter-host` facade (registry keyed by pool/replica) allows `orchestratord` to rebind adapters on reload/drain, route submit/cancel, and apply narration/metrics wrappers consistently.

- Low-level
  - Streaming decode paths minimize allocations by reusing buffers and parsing lightweight token frames (e.g., `{ "t": <string>, "i": <int> }`). Timeouts are enforced per request; retries apply only to idempotent operations. Sensitive headers (e.g., `Authorization`) are redacted in error logs. Determinism signals (engine_version, sampler profile) are surfaced in `started`/`end` frames or logs.
