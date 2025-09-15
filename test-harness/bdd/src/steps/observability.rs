use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^metrics conform to linter names and labels$").unwrap(),
        Regex::new(r"^label cardinality budgets are enforced$").unwrap(),
        Regex::new(r"^started event and admission logs$").unwrap(),
        Regex::new(r"^include queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^logs do not contain secrets or API keys$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn then_metrics_conform_names_labels(_w: &mut World) {}
    pub fn then_label_cardinality_budgets_enforced(_w: &mut World) {}
    pub fn given_started_event_and_admission_logs(_w: &mut World) {}
    pub fn then_logs_include_queue_pos_eta(_w: &mut World) {}
    pub fn then_logs_do_not_contain_secrets_or_api_keys(_w: &mut World) {}
}
