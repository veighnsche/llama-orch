pub mod adapters;
pub mod catalog;
pub mod control_plane;
pub mod data_plane;
pub mod deadlines_preemption;
pub mod error_taxonomy;
pub mod lifecycle;
pub mod observability;
pub mod pool_manager;
pub mod security;
pub mod world;
pub mod http_util;

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
        Regex::new(r"^I request capabilities$").unwrap(),
        Regex::new(r"^I receive capabilities with engines and API version$").unwrap(),
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
        // deadlines & SSE metrics
        Regex::new(r"^a task with infeasible deadline$").unwrap(),
        Regex::new(r"^I receive error code DEADLINE_UNMET$").unwrap(),
        Regex::new(r"^SSE metrics include on_time_probability$").unwrap(),
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
        // security
        Regex::new(r"^no API key is provided$").unwrap(),
        Regex::new(r"^I receive 401 Unauthorized$").unwrap(),
        Regex::new(r"^an invalid API key is provided$").unwrap(),
        Regex::new(r"^I receive 403 Forbidden$").unwrap(),
        // http-util
        Regex::new(r"^no special http-util configuration$").unwrap(),
        Regex::new(r"^a log line with Authorization Bearer token \"([^\"]+)\"$").unwrap(),
        Regex::new(r"^I apply http-util redaction$").unwrap(),
        Regex::new(r"^the output masks the token and includes its fp6$").unwrap(),
        Regex::new(r"^the output does not contain the raw token$").unwrap(),
        Regex::new(r"^a log line with X-API-Key \"([^\"]+)\"$").unwrap(),
        Regex::new(r"^the output masks X-API-Key and includes its fp6$").unwrap(),
        Regex::new(r"^AUTH_TOKEN is set to \"([^\"]+)\"$").unwrap(),
        Regex::new(r"^AUTH_TOKEN is unset$").unwrap(),
        Regex::new(r"^I apply with_bearer_if_configured to a GET request$").unwrap(),
        Regex::new(r"^the request has Authorization header \"([^\"]+)\"$").unwrap(),
        Regex::new(r"^the request has no Authorization header$").unwrap(),
        Regex::new(r"^I get the http-util client twice$").unwrap(),
        Regex::new(r"^both references point to the same client$").unwrap(),
        // http-util retries and streaming (spec placeholders)
        Regex::new(r"^a transient upstream that returns 503 then succeeds$").unwrap(),
        Regex::new(r"^I invoke with_retries around an idempotent request$").unwrap(),
        Regex::new(r"^attempts follow default policy base 100ms multiplier 2\.0 cap 2s max attempts 4$").unwrap(),
        Regex::new(r"^an upstream that returns 400 Bad Request$").unwrap(),
        Regex::new(r"^no retry occurs$").unwrap(),
        Regex::new(r"^a body stream with started token token metrics end$").unwrap(),
        Regex::new(r"^I decode with stream_decode$").unwrap(),
        Regex::new(r"^ordering is preserved and token indices are strictly increasing$").unwrap(),
        // http-util client defaults (placeholders)
        Regex::new(r"^I inspect http-util client defaults$").unwrap(),
        Regex::new(r"^connect timeout is approximately 5s and request timeout approximately 30s$").unwrap(),
        Regex::new(r"^TLS verification is ON by default$").unwrap(),
        Regex::new(r"^HTTP/2 keep-alive is enabled when server supports ALPN$").unwrap(),
    ]
}

// WorldInventory registration is declared at crate root (lib.rs) to provide
// crate::world::World path that attribute macros expect.
