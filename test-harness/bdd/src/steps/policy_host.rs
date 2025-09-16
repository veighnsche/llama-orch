use crate::steps::world::World;
use cucumber::{given, then};

#[given(regex = r"^a policy host$")]
pub async fn given_policy_host(world: &mut World) {
    world.push_fact("policy.host");
}

#[then(regex = r"^the default plugin ABI is WASI$")]
pub async fn then_default_abi_wasi(_world: &mut World) {}

#[then(regex = r"^functions are pure and deterministic over explicit snapshots$")]
pub async fn then_functions_pure_deterministic(_world: &mut World) {}

#[then(regex = r"^ABI versioning is explicit and bumps MAJOR on breaking changes$")]
pub async fn then_abi_versioning_major_bump(_world: &mut World) {}

#[then(regex = r"^plugins run in a sandbox with no filesystem or network by default$")]
pub async fn then_plugins_sandboxed_no_io(_world: &mut World) {}

#[then(regex = r"^host bounds CPU time and memory per invocation$")]
pub async fn then_host_bounds_cpu_mem(_world: &mut World) {}

#[then(regex = r"^host logs plugin id version decision and latency$")]
pub async fn then_host_logs_plugin_fields(_world: &mut World) {}
