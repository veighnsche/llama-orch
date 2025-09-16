pub mod adapters;
pub mod apply_steps;
pub mod catalog;
pub mod config;
pub mod control_plane;
pub mod core_guardrails;
pub mod data_plane;
pub mod deadlines_preemption;
pub mod determinism;
pub mod error_taxonomy;
pub mod lifecycle;
pub mod observability;
pub mod policy_host;
pub mod policy_sdk;
pub mod pool_manager;
pub mod preflight_steps;
pub mod scheduling;
pub mod security;
pub mod world;

use regex::Regex;

/// Registry of all step regex patterns used in BDD features.
/// This is consumed by tests/bdd.rs to fail on undefined/ambiguous steps.
#[cfg_attr(not(test), allow(dead_code))]
pub fn registry() -> Vec<Regex> {
    vec![
        // adapters
        Regex::new(r"^a worker adapter for llama\.cpp$|^a worker adapter for llamacpp$").unwrap(),
        Regex::new(r"^the adapter implements health properties completion cancel metrics$")
            .unwrap(),
        Regex::new(r"^OpenAI-compatible endpoints are internal only$").unwrap(),
        Regex::new(r"^adapter reports engine_version and model_digest$").unwrap(),
        Regex::new(r"^a worker adapter for vllm$").unwrap(),
        Regex::new(
            r"^the adapter implements health/properties/completion/cancel/metrics against vLLM$",
        )
        .unwrap(),
        Regex::new(r"^a worker adapter for tgi$").unwrap(),
        Regex::new(r"^the adapter implements TGI custom API and metrics$").unwrap(),
        Regex::new(r"^a worker adapter for triton$").unwrap(),
        Regex::new(r"^the adapter implements infer/streaming and metrics$").unwrap(),
        // apply_steps
        Regex::new(r"^when apply runs$").unwrap(),
        Regex::new(r"^target filesystem is unsupported$").unwrap(),
        Regex::new(r"^apply completes successfully$").unwrap(),
        // catalog
        Regex::new(r"^a catalog model payload$").unwrap(),
        Regex::new(r"^I create a catalog model$").unwrap(),
        Regex::new(r"^the model is created$").unwrap(),
        Regex::new(r"^I get the catalog model$").unwrap(),
        Regex::new(r"^the manifest signatures and sbom are present$").unwrap(),
        Regex::new(r"^I verify the catalog model$").unwrap(),
        Regex::new(r"^verification starts$").unwrap(),
        Regex::new(r"^catalog_verifications_total metric is exported$").unwrap(),
        Regex::new(r"^strict trust policy is enabled$").unwrap(),
        Regex::new(r"^an unsigned catalog artifact$").unwrap(),
        Regex::new(r"^catalog ingestion fails with UNTRUSTED_ARTIFACT$").unwrap(),
        // config
        Regex::new(r"^a valid example config$").unwrap(),
        Regex::new(r"^schema validation passes$").unwrap(),
        Regex::new(r"^strict mode with unknown field$").unwrap(),
        Regex::new(r"^validation rejects unknown fields$").unwrap(),
        Regex::new(r"^schema is generated twice$").unwrap(),
        Regex::new(r"^outputs are identical$").unwrap(),
        // control_plane
        Regex::new(r"^a Control Plane API endpoint$").unwrap(),
        Regex::new(r"^a pool id$").unwrap(),
        Regex::new(r"^I request pool health$").unwrap(),
        Regex::new(r"^I receive 200 with liveness readiness draining and metrics$").unwrap(),
        Regex::new(r"^I request pool drain with deadline_ms$").unwrap(),
        Regex::new(r"^draining begins$").unwrap(),
        Regex::new(r"^I request pool reload with new model_ref$").unwrap(),
        Regex::new(r"^reload succeeds and is atomic$").unwrap(),
        Regex::new(r"^reload fails and rolls back atomically$").unwrap(),
        Regex::new(r"^I request replicasets$").unwrap(),
        Regex::new(r"^I receive a list of replica sets with load and SLO snapshots$").unwrap(),
        // data_plane & SSE & sessions
        Regex::new(r"^an OrchQueue API endpoint$").unwrap(),
        Regex::new(r"^I enqueue a completion task with valid payload$").unwrap(),
        Regex::new(r"^I receive 202 Accepted with correlation id$").unwrap(),
        Regex::new(r"^I stream task events$").unwrap(),
        Regex::new(r"^I receive SSE events started, token, end$").unwrap(),
        Regex::new(r"^I receive SSE metrics frames$").unwrap(),
        Regex::new(r"^started includes queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^SSE event ordering is per stream$").unwrap(),
        Regex::new(r"^queue full policy is reject$").unwrap(),
        Regex::new(r"^queue full policy is drop-lru$").unwrap(),
        Regex::new(r"^queue full policy is shed-low-priority$").unwrap(),
        Regex::new(r"^an OrchQueue API endpoint under load$").unwrap(),
        Regex::new(r"^I enqueue a task beyond capacity$").unwrap(),
        Regex::new(r"^I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id$")
            .unwrap(),
        Regex::new(r"^the error body includes policy_label retriable and retry_after_ms$").unwrap(),
        Regex::new(r"^an existing queued task$").unwrap(),
        Regex::new(r"^I cancel the task$").unwrap(),
        Regex::new(r"^I receive 204 No Content with correlation id$").unwrap(),
        Regex::new(r"^a session id$").unwrap(),
        Regex::new(r"^I query the session$").unwrap(),
        Regex::new(r"^I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth$")
            .unwrap(),
        Regex::new(r"^I delete the session$").unwrap(),
        // deadlines & preemption
        Regex::new(r"^a task with infeasible deadline$").unwrap(),
        Regex::new(r"^I receive error code DEADLINE_UNMET$").unwrap(),
        Regex::new(r"^SSE metrics include on_time_probability$").unwrap(),
        Regex::new(r"^soft preemption is enabled$").unwrap(),
        Regex::new(r"^under persistent overload$").unwrap(),
        Regex::new(r"^lower priority items are preempted first$").unwrap(),
        Regex::new(r"^preemptions_total and resumptions_total metrics are exported$").unwrap(),
        Regex::new(r"^hard preemption is enabled and adapter proves interruptible_decode$")
            .unwrap(),
        Regex::new(r"^preempted flag and resumable state are surfaced$").unwrap(),
        // determinism
        Regex::new(r"^two replicas pin engine_version sampler_profile_version and model_digest$")
            .unwrap(),
        Regex::new(r"^same prompt parameters and seed are used$").unwrap(),
        Regex::new(r"^token streams are byte-exact across replicas$").unwrap(),
        Regex::new(r"^determinism is not assumed across engine or model updates$").unwrap(),
        Regex::new(r"^replicas across engine or model versions are used$").unwrap(),
        // error taxonomy
        Regex::new(r"^I trigger INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I receive 400 with correlation id and error envelope code INVALID_PARAMS$")
            .unwrap(),
        Regex::new(r"^I trigger POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE$")
            .unwrap(),
        Regex::new(r"^I trigger INTERNAL error$").unwrap(),
        Regex::new(r"^I receive 500 with correlation id and error envelope code INTERNAL$")
            .unwrap(),
        Regex::new(r"^error envelope includes engine when applicable$").unwrap(),
        // lifecycle
        Regex::new(r"^I set model state Deprecated with deadline_ms$").unwrap(),
        Regex::new(r"^new sessions are blocked with MODEL_DEPRECATED$").unwrap(),
        Regex::new(r"^I set model state Retired$").unwrap(),
        Regex::new(r"^pools unload and archives retained$").unwrap(),
        Regex::new(r"^model_state gauge is exported$").unwrap(),
        // observability
        Regex::new(r"^metrics conform to linter names and labels$").unwrap(),
        Regex::new(r"^label cardinality budgets are enforced$").unwrap(),
        Regex::new(r"^started event and admission logs$").unwrap(),
        Regex::new(r"^include queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^logs do not contain secrets or API keys$").unwrap(),
        // placeholders used by tests/features/observability/basic.feature
        Regex::new(r"^noop$").unwrap(),
        Regex::new(r"^nothing happens$").unwrap(),
        Regex::new(r"^it passes$").unwrap(),
        // policy host/sdk
        Regex::new(r"^a policy host$").unwrap(),
        Regex::new(r"^the default plugin ABI is WASI$").unwrap(),
        Regex::new(r"^functions are pure and deterministic over explicit snapshots$").unwrap(),
        Regex::new(r"^ABI versioning is explicit and bumps MAJOR on breaking changes$").unwrap(),
        Regex::new(r"^plugins run in a sandbox with no filesystem or network by default$").unwrap(),
        Regex::new(r"^host bounds CPU time and memory per invocation$").unwrap(),
        Regex::new(r"^host logs plugin id version decision and latency$").unwrap(),
        Regex::new(r"^a policy SDK$").unwrap(),
        Regex::new(r"^public SDK functions are semver-stable within a MAJOR$").unwrap(),
        Regex::new(r"^breaking changes include a migration note and version bump$").unwrap(),
        Regex::new(r"^SDK performs no network or filesystem I/O by default$").unwrap(),
        // pool manager
        Regex::new(r"^pool is Unready due to preload failure$").unwrap(),
        Regex::new(r"^pool readiness is false and last error cause is present$").unwrap(),
        Regex::new(r"^driver error occurs$").unwrap(),
        Regex::new(r"^pool transitions to Unready and restarts with backoff$").unwrap(),
        Regex::new(r"^restart storms are bounded by circuit breaker$").unwrap(),
        Regex::new(r"^device masks are configured$").unwrap(),
        Regex::new(r"^placement respects device masks; no cross-mask spillover occurs$").unwrap(),
        Regex::new(r"^heterogeneous split ratios are configured$").unwrap(),
        Regex::new(r"^per-GPU resident KV is capped for smallest GPU$").unwrap(),
        // preflight
        Regex::new(r"^side effects are not performed \(DryRun is default\)$").unwrap(),
        Regex::new(r"^I run preflight and apply in Commit mode$").unwrap(),
        Regex::new(r"^a critical compatibility violation is detected in preflight$").unwrap(),
        Regex::new(r"^I run the engine with default policy$").unwrap(),
        Regex::new(r"^when preflight runs$").unwrap(),
        // scheduling
        Regex::new(r"^WFQ weights are configured for tenants and priorities$").unwrap(),
        Regex::new(r"^load arrives across tenants and priorities$").unwrap(),
        Regex::new(r"^observed share approximates configured weights$").unwrap(),
        Regex::new(r"^quotas are configured per tenant$").unwrap(),
        Regex::new(r"^requests beyond quota are rejected$").unwrap(),
        Regex::new(r"^session affinity keeps a session on its last good replica$").unwrap(),
        // security
        Regex::new(r"^no API key is provided$").unwrap(),
        Regex::new(r"^I receive 401 Unauthorized$").unwrap(),
        Regex::new(r"^an invalid API key is provided$").unwrap(),
        Regex::new(r"^I receive 403 Forbidden$").unwrap(),
        // core_guardrails
        Regex::new(r"^a task with context length beyond model limit$").unwrap(),
        Regex::new(r"^a task with token budget exceeding configured limit$").unwrap(),
        Regex::new(r"^the request is rejected before enqueue$").unwrap(),
        Regex::new(r"^a running task exceeding watchdog thresholds$").unwrap(),
        Regex::new(r"^the watchdog aborts the task$").unwrap(),
    ]
}

// WorldInventory registration is declared at crate root (lib.rs) to provide
// crate::world::World path that attribute macros expect.
