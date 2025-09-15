use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^noop$").unwrap(),
        Regex::new(r"^nothing happens$").unwrap(),
        Regex::new(r"^it passes$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn noop(_w: &mut World) {}
    pub fn nothing_happens(_w: &mut World) {}
    pub fn it_passes(_w: &mut World) {}
}
