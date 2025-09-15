use crate::steps::world::World;
use cucumber::{given, then};

#[given(regex = r"^a task with context length beyond model limit$")]
pub async fn given_ctx_length_beyond_limit(world: &mut World) {
    world.push_fact("guard.ctx_over_limit");
}

#[given(regex = r"^a task with token budget exceeding configured limit$")]
pub async fn given_token_budget_exceeded(world: &mut World) {
    world.push_fact("guard.token_budget_exceeded");
}

#[then(regex = r"^the request is rejected before enqueue$")]
pub async fn then_rejected_before_enqueue(_world: &mut World) {}

#[given(regex = r"^a running task exceeding watchdog thresholds$")]
pub async fn given_running_task_exceeds_watchdog(world: &mut World) {
    world.push_fact("guard.watchdog_threshold_exceeded");
}

#[then(regex = r"^the watchdog aborts the task$")]
pub async fn then_watchdog_aborts_task(_world: &mut World) {}
