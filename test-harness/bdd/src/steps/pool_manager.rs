use crate::steps::world::World;
use cucumber::{given, then};

#[given(regex = r"^pool is Unready due to preload failure$")]
pub async fn given_pool_unready_due_to_preload_failure(world: &mut World) {
    world.push_fact("pool.preload_fail");
}

#[then(regex = r"^pool readiness is false and last error cause is present$")]
pub async fn then_pool_readiness_false_last_error_present(_world: &mut World) {}

#[given(regex = r"^driver error occurs$")]
pub async fn given_driver_error_occurs(world: &mut World) {
    world.push_fact("pool.driver_error");
}

#[then(regex = r"^pool transitions to Unready and restarts with backoff$")]
pub async fn then_pool_unready_and_restarts_with_backoff(_world: &mut World) {}

#[then(regex = r"^restart storms are bounded by circuit breaker$")]
pub async fn then_restart_storms_bounded_by_circuit_breaker(_world: &mut World) {}

#[given(regex = r"^device masks are configured$")]
pub async fn given_device_masks_configured(world: &mut World) {
    world.push_fact("pool.device_masks");
}

#[then(regex = r"^placement respects device masks; no cross-mask spillover occurs$")]
pub async fn then_placement_respects_device_masks_no_spill(_world: &mut World) {}

#[given(regex = r"^heterogeneous split ratios are configured$")]
pub async fn given_heterogeneous_split_ratios_configured(world: &mut World) {
    world.push_fact("pool.hetero_split");
}

#[then(regex = r"^per-GPU resident KV is capped for smallest GPU$")]
pub async fn then_per_gpu_kv_capped_smallest_gpu(_world: &mut World) {}
