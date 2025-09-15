use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^WFQ weights are configured for tenants and priorities$").unwrap(),
        Regex::new(r"^load arrives across tenants and priorities$").unwrap(),
        Regex::new(r"^observed share approximates configured weights$").unwrap(),
        Regex::new(r"^quotas are configured per tenant$").unwrap(),
        Regex::new(r"^requests beyond quota are rejected$").unwrap(),
        Regex::new(r"^session affinity keeps a session on its last good replica$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_wfq_weights_configured(_w: &mut World) {}
    pub fn when_load_arrives_across_tenants_priorities(_w: &mut World) {}
    pub fn then_observed_share_approximates_weights(_w: &mut World) {}
    pub fn given_quotas_configured_per_tenant(_w: &mut World) {}
    pub fn then_requests_beyond_quota_rejected(_w: &mut World) {}
    pub fn then_session_affinity_keeps_last_good_replica(_w: &mut World) {}
}
