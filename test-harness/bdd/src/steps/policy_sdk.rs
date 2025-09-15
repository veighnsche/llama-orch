use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^a policy SDK$").unwrap(),
        Regex::new(r"^public SDK functions are semver-stable within a MAJOR$").unwrap(),
        Regex::new(r"^breaking changes include a migration note and version bump$").unwrap(),
        Regex::new(r"^SDK performs no network or filesystem I/O by default$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_policy_sdk(_w: &mut World) {}
    pub fn then_sdk_semver_stable_within_major(_w: &mut World) {}
    pub fn then_breaking_changes_require_migration_and_bump(_w: &mut World) {}
    pub fn then_sdk_no_io_by_default(_w: &mut World) {}
}
