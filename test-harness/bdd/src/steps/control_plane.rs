use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
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
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
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
}
