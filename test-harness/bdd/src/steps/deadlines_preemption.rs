use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        // Deadlines & SSE metrics
        Regex::new(r"^a task with infeasible deadline$").unwrap(),
        Regex::new(r"^I receive error code DEADLINE_UNMET$").unwrap(),
        Regex::new(r"^SSE metrics include on_time_probability$").unwrap(),

        // Preemption
        Regex::new(r"^soft preemption is enabled$").unwrap(),
        Regex::new(r"^under persistent overload$").unwrap(),
        Regex::new(r"^lower priority items are preempted first$").unwrap(),
        Regex::new(r"^preemptions_total and resumptions_total metrics are exported$").unwrap(),
        Regex::new(r"^hard preemption is enabled and adapter proves interruptible_decode$").unwrap(),
        Regex::new(r"^preempted flag and resumable state are surfaced$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_task_with_infeasible_deadline(_w: &mut World) {}
    pub fn then_deadline_unmet_error(_w: &mut World) {}
    pub fn then_sse_metrics_include_on_time_probability(_w: &mut World) {}
    pub fn given_soft_preemption_enabled(_w: &mut World) {}
    pub fn given_persistent_overload(_w: &mut World) {}
    pub fn then_lower_priority_preempted_first(_w: &mut World) {}
    pub fn then_preemptions_and_resumptions_metrics_exported(_w: &mut World) {}
    pub fn given_hard_preemption_with_interruptible_decode(_w: &mut World) {}
    pub fn then_preempted_flag_and_resumable_state(_w: &mut World) {}
}
