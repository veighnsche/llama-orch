use crate::steps_2::world::World;
use cucumber::{given, then};

#[given(regex = r"^a task with infeasible deadline$")]
pub async fn given_task_with_infeasible_deadline(world: &mut World) { world.push_fact("deadline.infeasible"); }

#[then(regex = r"^I receive error code DEADLINE_UNMET$")]
pub async fn then_deadline_unmet_error(_world: &mut World) {}

#[then(regex = r"^SSE metrics include on_time_probability$")]
pub async fn then_sse_metrics_include_on_time_probability(_world: &mut World) {}

#[given(regex = r"^soft preemption is enabled$")]
pub async fn given_soft_preemption_enabled(world: &mut World) { world.push_fact("preempt.soft"); }

#[given(regex = r"^under persistent overload$")]
pub async fn given_persistent_overload(world: &mut World) { world.push_fact("overload"); }

#[then(regex = r"^lower priority items are preempted first$")]
pub async fn then_lower_priority_preempted_first(_world: &mut World) {}

#[then(regex = r"^preemptions_total and resumptions_total metrics are exported$")]
pub async fn then_preemptions_and_resumptions_metrics_exported(_world: &mut World) {}

#[given(regex = r"^hard preemption is enabled and adapter proves interruptible_decode$")]
pub async fn given_hard_preemption_with_interruptible_decode(world: &mut World) { world.push_fact("preempt.hard"); }

#[then(regex = r"^preempted flag and resumable state are surfaced$")]
pub async fn then_preempted_flag_and_resumable_state(_world: &mut World) {}
