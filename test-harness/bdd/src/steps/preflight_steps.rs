use crate::steps::world::World;
use cucumber::{given, then, when};

#[then(regex = r"^side effects are not performed \(DryRun is default\)$")]
pub async fn then_side_effects_not_performed(world: &mut World) {
    for ev in world.all_facts() {
        if let Some(stage) = ev.get("stage").and_then(|v| v.as_str()) {
            assert!(
                !stage.starts_with("apply"),
                "unexpected apply-stage facts found in DryRun-by-default scenario: {}",
                stage
            );
        }
    }
}

#[when(regex = r"^I run preflight and apply in Commit mode$")]
pub async fn when_preflight_and_apply_commit(world: &mut World) {
    crate::steps::preflight_steps::when_preflight(world).await;
    crate::steps::apply_steps::when_apply(world).await;
}

#[given(regex = r"^a critical compatibility violation is detected in preflight$")]
pub async fn given_critical_violation(world: &mut World) {
    crate::steps::apply_steps::given_target_fs_unsupported(world).await;
}

#[when(regex = r"^I run the engine with default policy$")]
pub async fn when_run_engine_default(world: &mut World) {
    crate::steps::preflight_steps::when_preflight(world).await;
}

#[when(regex = r"^when preflight runs$")]
pub async fn when_preflight(world: &mut World) {
    world.push_fact("preflight");
}
