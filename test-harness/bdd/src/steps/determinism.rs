use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^two replicas pin engine_version sampler_profile_version and model_digest$").unwrap(),
        Regex::new(r"^same prompt parameters and seed are used$").unwrap(),
        Regex::new(r"^token streams are byte-exact across replicas$").unwrap(),
        Regex::new(r"^determinism is not assumed across engine or model updates$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_two_replicas_pinned_versions_artifacts(_w: &mut World) {}
    pub fn when_same_prompt_params_seed(_w: &mut World) {}
    pub fn then_token_streams_byte_exact(_w: &mut World) {}
    pub fn then_no_cross_version_determinism_assumed(_w: &mut World) {}
}
