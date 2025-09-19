# 02 — Language Surface Requirements

## Control Flow & Semantics (MUST/SHOULD)

| Feature | Required? | Notes | Source |
|---|---|---|---|
| if/else admission gates | MUST | Reject infeasible ctx/token budgets before enqueue; typed errors and 429 backpressure. | .specs/10-orchestrator-core.md:66–71; .specs/20-orchestratord.md:22–25 |
| match/error taxonomy | MUST | Stable `code` field with specific variants; mapping to HTTP status. | .specs/20-orchestratord.md:50–55 |
| repeat-until (stream until end/cancel) | MUST | SSE emits `token*` until `end` or `error`; cancel path stops tokens. | .specs/20-orchestratord.md:26–31; .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:17–24 |
| parallel/join | SHOULD | Not explicitly defined; implied by multiple pools/replicas and concurrent sessions; no language primitive in docs. | README.md:26–36 (architecture); .specs/00_llama-orch.md:49–53 |
| foreach | MAY | Not specified as a language primitive; clients can iterate over tasks. | README.md:191–201 |
| try/catch | SHOULD | Error taxonomy allows typed handling at client; server maps to JSON envelopes. | .specs/20-orchestratord.md:50–55 |
| semantic routing | SHOULD | Centralized placement policy proposes deterministic scoring/overrides. | .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:35–66 |
| critique→repair loops | MAY | Not present as a server primitive; can be built using artifacts + retries. | .specs/20-orchestratord.md:82–86; README.md:228–235 |

## Artifacts & Facts

- Required artifact types
  - SSE transcripts stored as artifacts (content-addressed). README.md:228–235; .specs/20-orchestratord.md:82–86
  - Catalog entries for models; engine catalog entries (proposed/accepted). .specs/25-catalog-core.md:20–33; .specs/56-engine-catalog.md:20–33
- Schema usage
  - OpenAPI schemas define `SSEStarted/SSEToken/SSEMetrics/SSEEnd/SSEError`. .specs/20-orchestratord.md:46–49
  - Config schema defines provisioning/model fields. .specs/60-config-schema.md:22–30
- Hashing & provenance
  - Artifacts are SHA-256 content-addressed; models/engines carry `model_digest`/`engine_digest` where available. .specs/25-catalog-core.md:46–50; .specs/56-engine-catalog.md:46–50; README.md:228–235
- Fact emission/consumption
  - SSE `metrics` frames carry additive facts (queue depth, on-time probability, budgets, kv warmth). .specs/00_llama-orch.md:82–83; .specs/20-orchestratord.md:91–94
  - Gates: admission rejects when constraints violated. .specs/10-orchestrator-core.md:66–71

## LLM-Native Primitives

- Built-ins
  - generate/stream via WorkerAdapter; `started → token* → end` framing. .specs/40–44 worker adapter specs; .specs/20-orchestratord.md:26–45
  - cancel via HTTP endpoint and per-task token (proposal). .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:17–24,49–58
- Output schema guarantees
  - Typed SSE payloads with required fields; JSON well-formed, ordering guaranteed. .specs/20-orchestratord.md:28–45
- Seeds & pinning
  - Determinism by fixed `seed`, `sampler_profile_version`, `engine_version`, `model_digest`; replica-set pinned versions. .specs/00_llama-orch.md:58–59; .specs/10-orchestrator-core.md:74–76
- Prompt/version pinning
  - Capability discovery includes `api_version`; logs/metrics capture engine version/digests. .specs/00_llama-orch.md:95–96; .specs/metrics/otel-prom.md:7–13
- Rubric/rater/validator
  - Not defined as server primitives; out of scope. Use artifacts + client-side validation.

## Safety, Budgets, Determinism

- Budgets
  - Per-session token/time/cost budgets SHOULD be supported; enforcement at admission/scheduling; surfaced via SSE metrics/headers. .specs/00_llama-orch.md:52–53; .specs/20-orchestratord.md:87–94
- Redaction & policy
  - Logs MUST NOT leak secrets; outbound HTTP tooling MUST be policy-gated. .specs/00_llama-orch.md:75–77; 116–117,137–138
- Sandboxing/network
  - No sandbox primitive; policy hook controls egress. .specs/00_llama-orch.md:116–117,137–138
- Seeds/determinism
  - Determinism by default within a replica set. .specs/00_llama-orch.md:58–59; .specs/10-orchestrator-core.md:74–76; .specs/70-determinism-suite.md:13–16
- Cache rules
  - KV cache bounded; no cross-worker KV migration; `kv_warmth` surfaced. .specs/00_llama-orch.md:50–52

## Gates (Quality)

- Tests green: provider verify, BDD, determinism, metrics lint. README_LLM.md:57–66; .docs/TESTING_POLICY.md:30–37
- Coverage thresholds: not explicitly stated; determinism suite requires 64 seeds corpus. .specs/70-determinism-suite.md:13–16
- Bench regressions: microbenchmarks proposed for SSE path; no hard numeric gate. .specs/proposals/2025-09-19-performance-streamlining.md:107–126
- Security scan: not specified; logs redaction mandatory. .specs/00_llama-orch.md:75–77

## Minimal Example (composed strictly from docs)

```yaml
# Client-side flow sketch (pseudocode YAML using documented endpoints)
- POST /v1/tasks
  body:
    task_id: "t1"
    session_id: "s1"
    engine: "llamacpp"
    model_ref: "hf:org/repo/model.gguf"
    ctx: 4096
    priority: "interactive"
    seed: 42
    max_tokens: 128
  expect:
    202 Accepted
    headers:
      X-Correlation-Id: <uuid>
    body:
      queue_position: <int>
      predicted_start_ms: <int>

- GET /v1/tasks/t1/stream  # SSE
  expect events in order:
    - started { queue_position, predicted_start_ms }
    - token { t, i }  # repeated
    - metrics { queue_depth, on_time_probability, kv_warmth, ... }  # optional, additive
    - end { tokens_out, decode_ms }

- POST /v1/tasks/t1/cancel  # idempotent; race-free
```
Sources: .docs/HOME_PROFILE.md:21–27; .specs/20-orchestratord.md:26–49; 87–94.

## Top 5 Opportunities

- Harden cancel-on-disconnect and bounded backpressure into normative server behavior; add SSE heartbeat guidance for slow clients. .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:17–24,67–76
- Clarify any parallel/join semantics or explicitly scope them out for the language layer. README.md:26–36; .specs/00_llama-orch.md:49–53
- Provide canonical OpenAPI x-examples for SSE frames and admission, including budgets. .specs/20-orchestratord.md:95–99
- Document artifact kinds and tags more tightly (trace/plan/diff) and retrieval patterns. README.md:228–235; .specs/20-orchestratord.md:82–86
- Specify coverage or performance targets for SSE hot path and adapter streaming decoder. .specs/proposals/2025-09-19-performance-streamlining.md:107–126
