use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^a catalog model payload$").unwrap(),
        Regex::new(r"^I create a catalog model$").unwrap(),
        Regex::new(r"^the model is created$").unwrap(),
        Regex::new(r"^I get the catalog model$").unwrap(),
        Regex::new(r"^the manifest signatures and sbom are present$").unwrap(),
        Regex::new(r"^I verify the catalog model$").unwrap(),
        Regex::new(r"^verification starts$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_catalog_model_payload(_w: &mut World) {}
    pub fn when_create_catalog_model(_w: &mut World) {}
    pub fn then_catalog_model_created(_w: &mut World) {}
    pub fn when_get_catalog_model(_w: &mut World) {}
    pub fn then_catalog_manifest_signatures_sbom_present(_w: &mut World) {}
    pub fn when_verify_catalog_model(_w: &mut World) {}
    pub fn then_verification_starts(_w: &mut World) {}
}
