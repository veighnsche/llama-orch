//! Step registry and stub functions for BDD tests (no runtime execution yet).

use regex::Regex;

/// Return the list of regexes that define recognized Gherkin steps.
/// Keep patterns precise to avoid ambiguity.
pub fn registry() -> Vec<Regex> {
    vec![
        // placeholders
        Regex::new(r"^noop$").unwrap(),
        Regex::new(r"^nothing happens$").unwrap(),
        Regex::new(r"^it passes$").unwrap(),

        // Enqueue + Stream
        Regex::new(r"^an OrchQueue API endpoint$").unwrap(),
        Regex::new(r"^I enqueue a completion task with valid payload$").unwrap(),
        Regex::new(r"^I receive 202 Accepted with correlation id$").unwrap(),
        Regex::new(r"^I stream task events$").unwrap(),
        Regex::new(r"^I receive SSE events started, token, end$").unwrap(),
        Regex::new(r"^I receive SSE metrics frames$").unwrap(),
        Regex::new(r"^started includes queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^SSE event ordering is per stream$").unwrap(),

        // Backpressure 429
        Regex::new(r"^an OrchQueue API endpoint under load$").unwrap(),
        Regex::new(r"^I enqueue a task beyond capacity$").unwrap(),
        Regex::new(r"^I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id$").unwrap(),
        Regex::new(r"^the error body includes policy_label retriable and retry_after_ms$").unwrap(),

        // Cancel
        Regex::new(r"^an existing queued task$").unwrap(),
        Regex::new(r"^I cancel the task$").unwrap(),
        Regex::new(r"^I receive 204 No Content with correlation id$").unwrap(),

        // Sessions
        Regex::new(r"^a session id$").unwrap(),
        Regex::new(r"^I query the session$").unwrap(),
        Regex::new(r"^I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth$").unwrap(),
        Regex::new(r"^I delete the session$").unwrap(),

        // Control Plane
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

        // Error taxonomy (typed errors)
        Regex::new(r"^I trigger INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I receive 400 with correlation id and error envelope code INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I trigger POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I trigger INTERNAL error$").unwrap(),
        Regex::new(r"^I receive 500 with correlation id and error envelope code INTERNAL$").unwrap(),
        Regex::new(r"^error envelope includes engine when applicable$").unwrap(),

        // Security
        Regex::new(r"^no API key is provided$").unwrap(),
        Regex::new(r"^I receive 401 Unauthorized$").unwrap(),
        Regex::new(r"^an invalid API key is provided$").unwrap(),
        Regex::new(r"^I receive 403 Forbidden$").unwrap(),

        // Catalog & Trust
        Regex::new(r"^a catalog model payload$").unwrap(),
        Regex::new(r"^I create a catalog model$").unwrap(),
        Regex::new(r"^the model is created$").unwrap(),
        Regex::new(r"^I get the catalog model$").unwrap(),
        Regex::new(r"^the manifest signatures and sbom are present$").unwrap(),
        Regex::new(r"^I verify the catalog model$").unwrap(),
        Regex::new(r"^verification starts$").unwrap(),

        // Lifecycle states
        Regex::new(r"^I set model state Deprecated with deadline_ms$").unwrap(),
        Regex::new(r"^new sessions are blocked with MODEL_DEPRECATED$").unwrap(),
        Regex::new(r"^I set model state Retired$").unwrap(),
        Regex::new(r"^pools unload and archives retained$").unwrap(),
        Regex::new(r"^model_state gauge is exported$").unwrap(),

        // Scheduling — WFQ and quotas
        Regex::new(r"^WFQ weights are configured for tenants and priorities$").unwrap(),
        Regex::new(r"^load arrives across tenants and priorities$").unwrap(),
        Regex::new(r"^observed share approximates configured weights$").unwrap(),
        Regex::new(r"^quotas are configured per tenant$").unwrap(),
        Regex::new(r"^requests beyond quota are rejected$").unwrap(),

        // Deadlines & SSE metrics
        Regex::new(r"^a task with infeasible deadline$").unwrap(),
        Regex::new(r"^I receive error code DEADLINE_UNMET$").unwrap(),
        Regex::new(r"^SSE metrics include on_time_probability$").unwrap(),

        // Preemption
        Regex::new(r"^soft preemption is enabled$").unwrap(),
        Regex::new(r"^under persistent overload$").unwrap(),
        Regex::new(r"^lower priority items are preempted first$").unwrap(),
        Regex::new(r"^preemptions_total and resumptions_total metrics are exported$").unwrap(),
        Regex::new(r"^hard preemption is enabled and adapter proves interruptible_decode$").unwrap(),
        Regex::new(r"^preempted flag and resumable state are surfaced$").unwrap(),

        // Pool Manager — preload & restart/backoff
        Regex::new(r"^pool is Unready due to preload failure$").unwrap(),
        Regex::new(r"^pool readiness is false and last error cause is present$").unwrap(),
        Regex::new(r"^driver error occurs$").unwrap(),
        Regex::new(r"^pool transitions to Unready and restarts with backoff$").unwrap(),
        Regex::new(r"^restart storms are bounded by circuit breaker$").unwrap(),
        Regex::new(r"^device masks are configured$").unwrap(),
        Regex::new(r"^placement respects device masks; no cross-mask spillover occurs$").unwrap(),
        Regex::new(r"^heterogeneous split ratios are configured$").unwrap(),
        Regex::new(r"^per-GPU resident KV is capped for smallest GPU$").unwrap(),

        // Config schema
        Regex::new(r"^a valid example config$").unwrap(),
        Regex::new(r"^schema validation passes$").unwrap(),
        Regex::new(r"^strict mode with unknown field$").unwrap(),
        Regex::new(r"^validation rejects unknown fields$").unwrap(),
        Regex::new(r"^schema is generated twice$").unwrap(),
        Regex::new(r"^outputs are identical$").unwrap(),

        // Determinism suite
        Regex::new(r"^two replicas pin engine_version sampler_profile_version and model_digest$").unwrap(),
        Regex::new(r"^same prompt parameters and seed are used$").unwrap(),
        Regex::new(r"^token streams are byte-exact across replicas$").unwrap(),

        // Metrics contract & observability
        Regex::new(r"^metrics conform to linter names and labels$").unwrap(),
        Regex::new(r"^label cardinality budgets are enforced$").unwrap(),
        Regex::new(r"^started event and admission logs$").unwrap(),
        Regex::new(r"^include queue_position and predicted_start_ms$").unwrap(),
    ]
}

// --- Stub step functions (no runner yet) ------------------------------------

#[allow(dead_code)]
pub struct World;

#[allow(dead_code)]
impl World {
    pub fn new() -> Self { Self }
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;

    // Placeholders
    pub fn noop(_w: &mut World) {}
    pub fn nothing_happens(_w: &mut World) {}
    pub fn it_passes(_w: &mut World) {}

    // Enqueue + Stream
    pub fn given_api_endpoint(_w: &mut World) {}
    pub fn when_enqueue_valid_completion(_w: &mut World) {}
    pub fn then_202_with_corr(_w: &mut World) {}
    pub fn when_stream_events(_w: &mut World) {}
    pub fn then_sse_started_token_end(_w: &mut World) {}
    pub fn then_sse_metrics_frames(_w: &mut World) {}
    pub fn then_started_includes_queue_pos_eta(_w: &mut World) {}
    pub fn then_sse_ordering_per_stream(_w: &mut World) {}

    // Backpressure 429
    pub fn given_api_under_load(_w: &mut World) {}
    pub fn when_enqueue_beyond_capacity(_w: &mut World) {}
    pub fn then_429_with_headers_and_corr(_w: &mut World) {}
    pub fn then_error_body_policy_label_retry(_w: &mut World) {}

    // Cancel
    pub fn given_existing_queued_task(_w: &mut World) {}
    pub fn when_cancel_task(_w: &mut World) {}
    pub fn then_204_with_corr(_w: &mut World) {}

    // Sessions
    pub fn given_session_id(_w: &mut World) {}
    pub fn when_query_session(_w: &mut World) {}
    pub fn then_session_info_fields(_w: &mut World) {}
    pub fn when_delete_session(_w: &mut World) {}

    // Control Plane
    pub fn given_control_plane_endpoint(_w: &mut World) {}
    pub fn given_pool_id(_w: &mut World) {}
    pub fn when_request_pool_health(_w: &mut World) {}
    pub fn then_health_200_fields(_w: &mut World) {}
    pub fn when_request_pool_drain_with_deadline(_w: &mut World) {}
    pub fn then_draining_begins(_w: &mut World) {}
    pub fn when_request_pool_reload_new_model(_w: &mut World) {}
    pub fn then_reload_succeeds_atomic(_w: &mut World) {}
    pub fn then_reload_fails_rollback_atomic(_w: &mut World) {}
    pub fn when_request_replicasets(_w: &mut World) {}
    pub fn then_replicasets_list_with_load_slo(_w: &mut World) {}

    // Error taxonomy
    pub fn when_trigger_invalid_params(_w: &mut World) {}
    pub fn then_400_corr_invalid_params(_w: &mut World) {}
    pub fn when_trigger_pool_unavailable(_w: &mut World) {}
    pub fn then_503_corr_pool_unavailable(_w: &mut World) {}
    pub fn when_trigger_internal_error(_w: &mut World) {}
    pub fn then_500_corr_internal(_w: &mut World) {}
    pub fn then_error_envelope_includes_engine(_w: &mut World) {}

    // Security
    pub fn given_no_api_key(_w: &mut World) {}
    pub fn then_401_unauthorized(_w: &mut World) {}
    pub fn given_invalid_api_key(_w: &mut World) {}
    pub fn then_403_forbidden(_w: &mut World) {}

    // Catalog & Trust
    pub fn given_catalog_model_payload(_w: &mut World) {}
    pub fn when_create_catalog_model(_w: &mut World) {}
    pub fn then_catalog_model_created(_w: &mut World) {}
    pub fn when_get_catalog_model(_w: &mut World) {}
    pub fn then_catalog_manifest_signatures_sbom_present(_w: &mut World) {}
    pub fn when_verify_catalog_model(_w: &mut World) {}
    pub fn then_verification_starts(_w: &mut World) {}

    // Lifecycle
    pub fn when_set_state_deprecated_with_deadline(_w: &mut World) {}
    pub fn then_new_sessions_blocked_model_deprecated(_w: &mut World) {}
    pub fn when_set_state_retired(_w: &mut World) {}
    pub fn then_pools_unload_archives_retained(_w: &mut World) {}
    pub fn then_model_state_gauge_exported(_w: &mut World) {}

    // Scheduling — WFQ and quotas
    pub fn given_wfq_weights_configured(_w: &mut World) {}
    pub fn when_load_arrives_across_tenants_priorities(_w: &mut World) {}
    pub fn then_observed_share_approximates_weights(_w: &mut World) {}
    pub fn given_quotas_configured_per_tenant(_w: &mut World) {}
    pub fn then_requests_beyond_quota_rejected(_w: &mut World) {}

    // Deadlines & SSE metrics
    pub fn given_task_with_infeasible_deadline(_w: &mut World) {}
    pub fn then_deadline_unmet_error(_w: &mut World) {}
    pub fn then_sse_metrics_include_on_time_probability(_w: &mut World) {}

    // Preemption
    pub fn given_soft_preemption_enabled(_w: &mut World) {}
    pub fn given_persistent_overload(_w: &mut World) {}
    pub fn then_lower_priority_preempted_first(_w: &mut World) {}
    pub fn then_preemptions_and_resumptions_metrics_exported(_w: &mut World) {}
    pub fn given_hard_preemption_with_interruptible_decode(_w: &mut World) {}
    pub fn then_preempted_flag_and_resumable_state(_w: &mut World) {}

    // Pool Manager — preload & restart/backoff
    pub fn given_pool_unready_due_to_preload_failure(_w: &mut World) {}
    pub fn then_pool_readiness_false_last_error_present(_w: &mut World) {}
    pub fn given_driver_error_occurs(_w: &mut World) {}
    pub fn then_pool_unready_and_restarts_with_backoff(_w: &mut World) {}
    pub fn then_restart_storms_bounded_by_circuit_breaker(_w: &mut World) {}
    pub fn given_device_masks_configured(_w: &mut World) {}
    pub fn then_placement_respects_device_masks_no_spill(_w: &mut World) {}
    pub fn given_heterogeneous_split_ratios_configured(_w: &mut World) {}
    pub fn then_per_gpu_kv_capped_smallest_gpu(_w: &mut World) {}

    // Config schema
    pub fn given_valid_example_config(_w: &mut World) {}
    pub fn then_schema_validation_passes(_w: &mut World) {}
    pub fn given_strict_mode_with_unknown_field(_w: &mut World) {}
    pub fn then_validation_rejects_unknown_fields(_w: &mut World) {}
    pub fn given_schema_generated_twice(_w: &mut World) {}
    pub fn then_schema_outputs_identical(_w: &mut World) {}

    // Determinism suite
    pub fn given_two_replicas_pinned_versions_artifacts(_w: &mut World) {}
    pub fn when_same_prompt_params_seed(_w: &mut World) {}
    pub fn then_token_streams_byte_exact(_w: &mut World) {}

    // Metrics & observability
    pub fn then_metrics_conform_names_labels(_w: &mut World) {}
    pub fn then_label_cardinality_budgets_enforced(_w: &mut World) {}
    pub fn given_started_event_and_admission_logs(_w: &mut World) {}
    pub fn then_logs_include_queue_pos_eta(_w: &mut World) {}
}
