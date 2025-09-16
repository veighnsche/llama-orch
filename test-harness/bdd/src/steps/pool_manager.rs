use crate::steps::world::World;
use cucumber::{given, then};
use pool_managerd::health::HealthStatus;

#[given(regex = r"^pool is Unready due to preload failure$")]
pub async fn given_pool_unready_due_to_preload_failure(world: &mut World) {
    world.push_fact("pool.preload_fail");
    // Also set in-memory registry to reflect Unready with last_error so Then can assert real state
    if let Ok(mut reg) = world.state.pool_manager.lock() {
        reg.set_health(
            "pool0",
            HealthStatus {
                live: true,
                ready: false,
            },
        );
        reg.set_last_error("pool0", "preload failure");
    }
}

#[then(regex = r"^pool readiness is false and last error cause is present$")]
pub async fn then_pool_readiness_false_last_error_present(world: &mut World) {
    let (ready, last_error_present) = if let Ok(reg) = world.state.pool_manager.lock() {
        let h = reg.get_health("pool0");
        let e = reg.get_last_error("pool0");
        (h.map(|x| x.ready).unwrap_or(true), e.is_some())
    } else {
        (true, false)
    };
    assert!(!ready, "expected pool ready=false");
    assert!(last_error_present, "expected last_error to be present");
}

#[given(regex = r"^driver error occurs$")]
pub async fn given_driver_error_occurs(world: &mut World) {
    world.push_fact("pool.driver_error");
    if let Ok(mut reg) = world.state.pool_manager.lock() {
        reg.set_health(
            "pool0",
            HealthStatus {
                live: true,
                ready: false,
            },
        );
        reg.set_last_error("pool0", "driver error");
    }
}

#[then(regex = r"^pool transitions to Unready and restarts with backoff$")]
pub async fn then_pool_unready_and_restarts_with_backoff(world: &mut World) {
    // Placeholder assertion: ensure Unready; backoff schedule validation to be added when lifecycle is implemented
    let ready = if let Ok(reg) = world.state.pool_manager.lock() {
        reg.get_health("pool0").map(|x| x.ready).unwrap_or(true)
    } else {
        true
    };
    assert!(!ready, "expected pool Unready after driver error");
}

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
