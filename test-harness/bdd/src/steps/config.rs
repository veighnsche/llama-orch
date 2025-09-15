use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^a valid example config$").unwrap(),
        Regex::new(r"^schema validation passes$").unwrap(),
        Regex::new(r"^strict mode with unknown field$").unwrap(),
        Regex::new(r"^validation rejects unknown fields$").unwrap(),
        Regex::new(r"^schema is generated twice$").unwrap(),
        Regex::new(r"^outputs are identical$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_valid_example_config(_w: &mut World) {}
    pub fn then_schema_validation_passes(_w: &mut World) {}
    pub fn given_strict_mode_with_unknown_field(_w: &mut World) {}
    pub fn then_validation_rejects_unknown_fields(_w: &mut World) {}
    pub fn given_schema_generated_twice(_w: &mut World) {}
    pub fn then_schema_outputs_identical(_w: &mut World) {}
}
