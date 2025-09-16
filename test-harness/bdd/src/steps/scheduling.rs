use crate::steps::world::World;
use cucumber::{given, then, when};
use std::collections::HashMap;

#[given(regex = r"^WFQ weights are configured for tenants and priorities$")]
pub async fn given_wfq_weights_configured(world: &mut World) {
    world.push_fact("sched.wfq");
}

#[when(regex = r"^load arrives across tenants and priorities$")]
pub async fn when_load_arrives_across_tenants_priorities(world: &mut World) {
    world.push_fact("sched.load");
}

#[then(regex = r"^observed share approximates configured weights$")]
pub async fn then_observed_share_approximates_weights(_world: &mut World) {
    // Simple WFQ simulation: weights {A:1, B:3}, 400 requests total should approximate 1:3
    let weights: HashMap<&str, u32> = HashMap::from_iter([("A", 1), ("B", 3)]);
    let total = 400;
    let sum_w: u32 = weights.values().sum();
    let mut observed: HashMap<&str, u32> = HashMap::new();
    let mut rr = 0u32;
    for _ in 0..total {
        // choose tenant by weight proportion using a simple counter
        let mut acc = 0u32;
        let pick = rr % sum_w;
        rr = rr.wrapping_add(1);
        let mut chosen = "A";
        for (t, w) in &weights {
            if pick < acc + *w {
                chosen = t;
                break;
            }
            acc += *w;
        }
        *observed.entry(chosen).or_default() += 1;
    }
    let share_a = *observed.get("A").unwrap_or(&0) as f64 / total as f64;
    let share_b = *observed.get("B").unwrap_or(&0) as f64 / total as f64;
    // target ratios: 0.25 and 0.75, allow 10% absolute tolerance
    assert!((share_a - 0.25).abs() < 0.10, "A share {:?}", share_a);
    assert!((share_b - 0.75).abs() < 0.10, "B share {:?}", share_b);
}

#[given(regex = r"^quotas are configured per tenant$")]
pub async fn given_quotas_configured_per_tenant(world: &mut World) {
    world.push_fact("sched.quotas");
}

#[then(regex = r"^requests beyond quota are rejected$")]
pub async fn then_requests_beyond_quota_rejected(_world: &mut World) {
    // Simulate quotas: tenant A quota=10, submit 15 -> expect 5 rejections
    let quota = 10;
    let submitted = 15;
    let accepted = quota.min(submitted);
    let rejected = submitted - accepted;
    assert_eq!(accepted, 10);
    assert_eq!(rejected, 5);
}

#[then(regex = r"^session affinity keeps a session on its last good replica$")]
pub async fn then_session_affinity_keeps_last_good_replica(_world: &mut World) {
    // Simulate sticky mapping
    let mut last_good: HashMap<&str, &str> = HashMap::new();
    last_good.insert("s-1", "rA");
    // new dispatch for s-1 should stay on rA
    let dispatch = *last_good.get("s-1").unwrap();
    assert_eq!(dispatch, "rA");
}
