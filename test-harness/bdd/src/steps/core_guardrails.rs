use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^context length exceeds the model limit$").unwrap(),
        Regex::new(r"^server rejects before enqueue with 400$").unwrap(),
        Regex::new(r"^token budget exceeds configured bounds$").unwrap(),
        Regex::new(r"^watchdog aborts a stuck job with wall and idle timeouts$").unwrap(),
        Regex::new(r"^a stuck job is running$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_ctx_exceeds_limit(_w: &mut World) {}
    pub fn then_reject_before_enqueue_400(_w: &mut World) {}
    pub fn given_token_budget_exceeds_bounds(_w: &mut World) {}
    pub fn then_watchdog_aborts_stuck_job(_w: &mut World) {}
    pub fn given_stuck_job_running(_w: &mut World) {}
}
