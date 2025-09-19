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
