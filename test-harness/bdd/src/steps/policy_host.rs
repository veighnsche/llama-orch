use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^a policy host$").unwrap(),
        Regex::new(r"^the default plugin ABI is WASI$").unwrap(),
        Regex::new(r"^functions are pure and deterministic over explicit snapshots$").unwrap(),
        Regex::new(r"^ABI versioning is explicit and bumps MAJOR on breaking changes$").unwrap(),
        Regex::new(r"^plugins run in a sandbox with no filesystem or network by default$").unwrap(),
        Regex::new(r"^host bounds CPU time and memory per invocation$").unwrap(),
        Regex::new(r"^host logs plugin id version decision and latency$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_policy_host(_w: &mut World) {}
    pub fn then_default_abi_wasi(_w: &mut World) {}
    pub fn then_functions_pure_deterministic(_w: &mut World) {}
    pub fn then_abi_versioning_major_bump(_w: &mut World) {}
    pub fn then_plugins_sandboxed_no_io(_w: &mut World) {}
    pub fn then_host_bounds_cpu_mem(_w: &mut World) {}
    pub fn then_host_logs_plugin_fields(_w: &mut World) {}
}
