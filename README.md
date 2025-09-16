# llama-orch (pre-code) — Developer Quickstart

This repository contains the pre-code scaffolding for an LLM Orchestrator. Use the commands below
to regenerate contracts, validate specs, and run tests.

See also:

- `README_LLM.md` — Decision rules and workflow for LLM developers
- `.docs/workflow.md` — Stages and gates (SPEC→SHIP)

## Quickstart

- Format check

```bash
cargo fmt --all -- --check
```

- Lints (warnings are errors)

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

- Regenerate contracts and requirements

```bash
cargo regen       # alias for: xtask regen (openapi + schema + spec-extract)
```

- Tests (workspace)

```bash
cargo test --workspace --all-features -- --nocapture
```

- Provider verification tests

```bash
cargo test -p orchestratord --test provider_verify -- --nocapture
```

- Consumer/CDC tests

```bash
cargo test -p cli-consumer-tests -- --nocapture
```

- Trybuild UI tests (compile-time checks)

```bash
cargo test -p tools-openapi-client -- --nocapture
```

- BDD harness (placeholder, ensures 0 undefined/ambiguous steps)

```bash
cargo test -p test-harness-bdd -- --nocapture
```

- Docs link checker

```bash
bash ci/scripts/check_links.sh
```

- Full developer loop (fmt, clippy, regen, tests, linkcheck)

```bash
cargo dev        # alias for: xtask dev:loop
```

- Generate/refresh READMEs from the indexer

```bash
cargo docs-index # alias for: run -p tools-readme-index --
```

## Notes for Stage 6 Vertical Slice (coming soon)

Once the Stage 6 Admission→Dispatch vertical is implemented, the Quickstart will include examples for:

- `POST /v1/tasks` (202 with queue position)
- `GET /v1/tasks/:id/stream` (SSE framing)
- `POST /v1/tasks/:id/cancel`

Until then, handlers remain stubs; metrics endpoint `/metrics` is available for liveness/contract checks.

<!-- BEGIN WORKSPACE MAP (AUTO-GENERATED) -->
## Workspace Map

| Path | Crate | Role | Key APIs/Contracts | Tests | Spec Refs |
|------|------|------|---------------------|-------|-----------|
| [`cli/consumer-tests/`](cli/consumer-tests/README.md) | `cli-consumer-tests` | test-harness | — |
orchqueue_pact, snapshot_transcript, snapshots, stub_wiremock | ORCH-3050, ORCH-3051 |
| [`contracts/api-types/`](contracts/api-types/README.md) | `contracts-api-types` | contracts | — |
— | ORCH-3044, ORCH-3030 |
| [`contracts/config-schema/`](contracts/config-schema/README.md) | `contracts-config-schema` |
contracts | Schema | validate_examples | ORCH-3044, ORCH-3030 |
| [`orchestrator-core/`](orchestrator-core/README.md) | `orchestrator-core` | core | — |
props_queue | ORCH-3004, ORCH-3005, ORCH-3008, ORCH-3010, ORCH-3011, ORCH-3016, ORCH-3017,
ORCH-3027, ORCH-3028, ORCH-3044, ORCH-3045 |
| [`orchestratord/`](orchestratord/README.md) | `orchestratord` | core | OpenAPI | provider_verify
| ORCH-3004, ORCH-3005, ORCH-3008, ORCH-3010, ORCH-3011, ORCH-3016, ORCH-3017, ORCH-3027,
ORCH-3028, ORCH-3044, ORCH-3045, ORCH-2002, ORCH-2101, ORCH-2102, ORCH-2103, ORCH-2104 |
| [`plugins/policy-host/`](plugins/policy-host/README.md) | `plugins-policy-host` | plugin | — | —
| ORCH-3048 |
| [`plugins/policy-sdk/`](plugins/policy-sdk/README.md) | `plugins-policy-sdk` | plugin | — | — |
ORCH-3048 |
| [`pool-managerd/`](pool-managerd/README.md) | `pool-managerd` | core | — | — | ORCH-3004,
ORCH-3005, ORCH-3008, ORCH-3010, ORCH-3011, ORCH-3016, ORCH-3017, ORCH-3027, ORCH-3028, ORCH-3044,
ORCH-3045, ORCH-3038, ORCH-3002 |
| [`test-harness/bdd/`](test-harness/bdd/README.md) | `test-harness-bdd` | test-harness | — | bdd,
features | ORCH-3050, ORCH-3051 |
| [`test-harness/chaos/`](test-harness/chaos/README.md) | `test-harness-chaos` | test-harness | — |
— | ORCH-3050, ORCH-3051 |
| [`test-harness/determinism-suite/`](test-harness/determinism-suite/README.md) |
`test-harness-determinism-suite` | test-harness | — | byte_exact, placeholder | ORCH-3050,
ORCH-3051 |
| [`test-harness/e2e-haiku/`](test-harness/e2e-haiku/README.md) | `test-harness-e2e-haiku` |
test-harness | — | e2e_client, placeholder | ORCH-3050, ORCH-3051 |
| [`test-harness/metrics-contract/`](test-harness/metrics-contract/README.md) |
`test-harness-metrics-contract` | test-harness | — | metrics_lint | ORCH-3050, ORCH-3051 |
| [`tools/openapi-client/`](tools/openapi-client/README.md) | `tools-openapi-client` | tool |
OpenAPI | trybuild, ui | — |
| [`tools/readme-index/`](tools/readme-index/README.md) | `tools-readme-index` | tool | — | — | — |
| [`tools/spec-extract/`](tools/spec-extract/README.md) | `tools-spec-extract` | tool | — | — | — |
| [`worker-adapters/llamacpp-http/`](worker-adapters/llamacpp-http/README.md) |
`worker-adapters-llamacpp-http` | adapter | — | — | ORCH-3054, ORCH-3055, ORCH-3056, ORCH-3057,
ORCH-3058 |
| [`worker-adapters/mock/`](worker-adapters/mock/README.md) | `worker-adapters-mock` | adapter | —
| — | ORCH-3054, ORCH-3055, ORCH-3056, ORCH-3057, ORCH-3058 |
| [`worker-adapters/tgi-http/`](worker-adapters/tgi-http/README.md) | `worker-adapters-tgi-http` |
adapter | — | — | ORCH-3054, ORCH-3055, ORCH-3056, ORCH-3057, ORCH-3058 |
| [`worker-adapters/triton/`](worker-adapters/triton/README.md) | `worker-adapters-triton` |
adapter | — | — | ORCH-3054, ORCH-3055, ORCH-3056, ORCH-3057, ORCH-3058 |
| [`worker-adapters/vllm-http/`](worker-adapters/vllm-http/README.md) | `worker-adapters-vllm-http`
| adapter | — | — | ORCH-3054, ORCH-3055, ORCH-3056, ORCH-3057, ORCH-3058 |
| [`xtask/`](xtask/README.md) | `xtask` | tool | — | — | — |

### Glossary

- `cli-consumer-tests` — cli-consumer-tests (test-harness)
- `contracts-api-types` — contracts-api-types (contracts)
- `contracts-config-schema` — contracts-config-schema (contracts)
- `orchestrator-core` — orchestrator-core (core)
- `orchestratord` — orchestratord (core)
- `plugins-policy-host` — plugins-policy-host (plugin)
- `plugins-policy-sdk` — plugins-policy-sdk (plugin)
- `pool-managerd` — pool-managerd (core)
- `test-harness-bdd` — test-harness-bdd (test-harness)
- `test-harness-chaos` — test-harness-chaos (test-harness)
- `test-harness-determinism-suite` — test-harness-determinism-suite (test-harness)
- `test-harness-e2e-haiku` — test-harness-e2e-haiku (test-harness)
- `test-harness-metrics-contract` — test-harness-metrics-contract (test-harness)
- `tools-openapi-client` — tools-openapi-client (tool)
- `tools-readme-index` — tools-readme-index (tool)
- `tools-spec-extract` — tools-spec-extract (tool)
- `worker-adapters-llamacpp-http` — worker-adapters-llamacpp-http (adapter)
- `worker-adapters-mock` — worker-adapters-mock (adapter)
- `worker-adapters-tgi-http` — worker-adapters-tgi-http (adapter)
- `worker-adapters-triton` — worker-adapters-triton (adapter)
- `worker-adapters-vllm-http` — worker-adapters-vllm-http (adapter)
- `xtask` — xtask (tool)

### Getting Started

- Adapter work: see `worker-adapters/*` crates.
- Contracts: see `contracts/*`.
- Core scheduling: see `orchestrator-core/` and `orchestratord/`.

<!-- END WORKSPACE MAP (AUTO-GENERATED) -->
