use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^I trigger INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I receive 400 with correlation id and error envelope code INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I trigger POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I trigger INTERNAL error$").unwrap(),
        Regex::new(r"^I receive 500 with correlation id and error envelope code INTERNAL$").unwrap(),
        Regex::new(r"^error envelope includes engine when applicable$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn when_trigger_invalid_params(_w: &mut World) {}
    pub fn then_400_corr_invalid_params(_w: &mut World) {}
    pub fn when_trigger_pool_unavailable(_w: &mut World) {}
    pub fn then_503_corr_pool_unavailable(_w: &mut World) {}
    pub fn when_trigger_internal_error(_w: &mut World) {}
    pub fn then_500_corr_internal(_w: &mut World) {}
    pub fn then_error_envelope_includes_engine(_w: &mut World) {}
}
