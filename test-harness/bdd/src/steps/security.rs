use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^no API key is provided$").unwrap(),
        Regex::new(r"^I receive 401 Unauthorized$").unwrap(),
        Regex::new(r"^an invalid API key is provided$").unwrap(),
        Regex::new(r"^I receive 403 Forbidden$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_no_api_key(_w: &mut World) {}
    pub fn then_401_unauthorized(_w: &mut World) {}
    pub fn given_invalid_api_key(_w: &mut World) {}
    pub fn then_403_forbidden(_w: &mut World) {}
}
