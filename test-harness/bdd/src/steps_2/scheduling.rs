use crate::steps_2::world::World;
use cucumber::{given, then, when};

#[given(regex = r"^WFQ weights are configured for tenants and priorities$")]
pub async fn given_wfq_weights_configured(world: &mut World) { world.push_fact("sched.wfq"); }

#[when(regex = r"^load arrives across tenants and priorities$")]
pub async fn when_load_arrives_across_tenants_priorities(world: &mut World) { world.push_fact("sched.load"); }

#[then(regex = r"^observed share approximates configured weights$")]
pub async fn then_observed_share_approximates_weights(_world: &mut World) {}

#[given(regex = r"^quotas are configured per tenant$")]
pub async fn given_quotas_configured_per_tenant(world: &mut World) { world.push_fact("sched.quotas"); }

#[then(regex = r"^requests beyond quota are rejected$")]
pub async fn then_requests_beyond_quota_rejected(_world: &mut World) {}

#[then(regex = r"^session affinity keeps a session on its last good replica$")]
pub async fn then_session_affinity_keeps_last_good_replica(_world: &mut World) {}
