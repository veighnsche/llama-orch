# Code Distribution Prediction at v0.1 Release

Methodology: estimated by reading all plans under `.plan/` and mapping scope/complexity to expected code volume per crate (Rust SLOC + config/templates, excluding large binary artifacts). Includes product, test harnesses, and tooling; values are approximate and sum to 100%.

## Per-Crate Percentages

| Crate | % of total | Rationale |
|---|---:|---|
| `orchestratord/` | 19.0% | Handlers, SSE streaming, backpressure, errors, metrics endpoint, control-plane (drain/reload/health/replicasets); accepts bulk of glue and HTTP plumbing. (README_LLM Stages 6,7,10,14) |
| `orchestrator-core/` | 13.0% | Queue, admission policies, scheduling hooks, fairness, determinism enforcement, core metrics. (Stage 9) |
| `pool-managerd/` | 10.0% | Replica registry, preload/readiness lifecycle, drain/reload/leases, backoff/circuit breaker; device masks/heterogeneous splits. (Stage 7) |
| `worker-adapters/llamacpp-http/` | 8.0% | First real-engine adapter: SSE bridge, errors, determinism normalization, metrics. (Stage 8) |
| `worker-adapters/vllm-http/` | 6.0% | Adapter parity with llama.cpp; OpenAI-compat mapping to internal worker contract. (Stage 8) |
| `worker-adapters/tgi-http/` | 5.0% | Adapter; SSE framing/backpressure normalized. (Stage 8) |
| `worker-adapters/triton/` | 6.0% | Adapter; TRT-LLM specifics; version capture. (Stage 8) |
| `worker-adapters/mock/` | 1.0% | Fault-injecting adapter used by tests. |
| `plugins/policy-host/` | 4.0% | WASI host; deterministic plugin ABI, sandboxing. |
| `plugins/policy-sdk/` | 3.0% | SDK surface + versioning; simple helpers. |
| `contracts/api-types/` | 2.0% | OAPI generated types + small glue. |
| `contracts/config-schema/` | 2.0% | Rust config types + schema emitter; examples/tests. |
| `cli/llama-orch-cli/` | 2.0% | CLI UX for submit/stream/cancel; quickstart. |
| `cli/consumer-tests/` | 2.0% | Pact tests + helpers. |
| `test-harness/determinism-suite/` | 3.0% | LCG generator, parsers, suite runner, fixtures. (Stage 4) |
| `test-harness/e2e-haiku/` | 3.0% | Haiku anti-cheat harness, nonce/minute/timezone handling. (Stage 15) |
| `test-harness/metrics-contract/` | 2.0% | Metrics linter/spec alignment tests. (Stage 5) |
| `test-harness/chaos/` | 2.0% | Chaos/load harness; scenarios + assertions. (Stage 16) |
| `test-harness/bdd/` | 3.0% | Cucumber runner, step defs, feature scaffolding. (Stage 12) |
| `tools/spec-extract/` | 1.5% | Requirements extraction; idempotent outputs. (Stage 17) |
| `tools/openapi-client/` | 1.5% | Trybuild UI and client codegen tests. |
| `tools/readme-index/` | 0.2% | Repo navigation generator. |
| `xtask/` | 0.8% | CI helpers, shortcuts (e.g., ci:haiku:cpu, ci:determinism). |

Total: 100.0%

### Product vs Tests/Tooling Split

- Product (operational crates): ~83%
  - `orchestratord/`, `orchestrator-core/`, `pool-managerd/`, all `worker-adapters/*`, `plugins/*`, `contracts/*`, `cli/*`.
- Tests & Harnesses: ~13%
  - `test-harness/*`: determinism, haiku, metrics, chaos, bdd.
- Tooling: ~4%
  - `tools/*`, `xtask/`.

## Notes & Assumptions

- Adapters are the second-largest block collectively; first engine prioritized is llama.cpp (8%), others scaled by parity.
- `orchestratord/` largest single crate due to HTTP/SSE, control-plane, and integration glue.
- `orchestrator-core/` smaller than `orchestratord/` but significant due to scheduling/fairness.
- Percentages include tests within each crate; harness crates are counted separately under tests.

## Plan Gaps Observed and Filled

- BDD harness and E2E Haiku lacked dedicated plan files. Added:
  - `.plan/72_bdd_harness.md` — BDD journeys, runner usage, acceptance.
  - `.plan/73_e2e_haiku.md` — Anti‑cheat E2E protocol and gates.
- CLI and tooling plans were implicit; added:
  - `.plan/80_cli.md` — CLI UX, CDC consumer CLI usage, packaging.
  - `.plan/90_tools.md` — spec-extract, openapi-client, readme-index, xtask (release tasks).

These align with README_LLM stages 6–17 and `.docs/workflow.md` product stages.
