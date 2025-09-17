# Code Distribution Expectations — Home Profile

This lightweight map helps keep code organised while we implement the home profile.

| Area | Approx. LOC target | Primary responsibilities |
|------|--------------------|---------------------------|
| `orchestrator-core/` | 10–15% | Queue primitives, placement heuristics, determinism helpers |
| `orchestratord/` | 30–35% | HTTP handlers, SSE pipeline, admission metadata, budgets |
| `pool-managerd/` | 15% | Replica registry, drain/reload, readiness tracking |
| `worker-adapters/*` | 20% | Engine adapters (mock + llama.cpp + vLLM + TGI + Triton) |
| `contracts/` | 5% | OpenAPI, config schema, generated types |
| `test-harness/` | 10–15% | BDD, determinism suite, metrics lint, Haiku |
| `cli/` | 5% | CLI tooling, consumer pact tests |

Guidelines:
- Keep domain logic in `orchestrator-core` and pure services to ease testing.
- Avoid leaking engine-specific code into `orchestratord/`; adapters own engine concerns.
- When a directory grows beyond its target, consider extracting helpers or refactoring before adding new features.
