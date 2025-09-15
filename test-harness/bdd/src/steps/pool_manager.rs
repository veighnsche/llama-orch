use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^pool is Unready due to preload failure$").unwrap(),
        Regex::new(r"^pool readiness is false and last error cause is present$").unwrap(),
        Regex::new(r"^driver error occurs$").unwrap(),
        Regex::new(r"^pool transitions to Unready and restarts with backoff$").unwrap(),
        Regex::new(r"^restart storms are bounded by circuit breaker$").unwrap(),
        Regex::new(r"^device masks are configured$").unwrap(),
        Regex::new(r"^placement respects device masks; no cross-mask spillover occurs$").unwrap(),
        Regex::new(r"^heterogeneous split ratios are configured$").unwrap(),
        Regex::new(r"^per-GPU resident KV is capped for smallest GPU$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_pool_unready_due_to_preload_failure(_w: &mut World) {}
    pub fn then_pool_readiness_false_last_error_present(_w: &mut World) {}
    pub fn given_driver_error_occurs(_w: &mut World) {}
    pub fn then_pool_unready_and_restarts_with_backoff(_w: &mut World) {}
    pub fn then_restart_storms_bounded_by_circuit_breaker(_w: &mut World) {}
    pub fn given_device_masks_configured(_w: &mut World) {}
    pub fn then_placement_respects_device_masks_no_spill(_w: &mut World) {}
    pub fn given_heterogeneous_split_ratios_configured(_w: &mut World) {}
    pub fn then_per_gpu_kv_capped_smallest_gpu(_w: &mut World) {}
}
