use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^I set model state Deprecated with deadline_ms$").unwrap(),
        Regex::new(r"^new sessions are blocked with MODEL_DEPRECATED$").unwrap(),
        Regex::new(r"^I set model state Retired$").unwrap(),
        Regex::new(r"^pools unload and archives retained$").unwrap(),
        Regex::new(r"^model_state gauge is exported$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn when_set_state_deprecated_with_deadline(_w: &mut World) {}
    pub fn then_new_sessions_blocked_model_deprecated(_w: &mut World) {}
    pub fn when_set_state_retired(_w: &mut World) {}
    pub fn then_pools_unload_archives_retained(_w: &mut World) {}
    pub fn then_model_state_gauge_exported(_w: &mut World) {}
}
