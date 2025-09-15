use serde_json::json;

#[derive(Debug, Default, cucumber::World)]
pub struct World {
    facts: Vec<serde_json::Value>,
    pub mode_commit: bool,
}

impl World {
    pub fn push_fact<S: AsRef<str>>(&mut self, stage: S) {
        self.facts.push(json!({
            "stage": stage.as_ref(),
        }));
    }

    pub fn all_facts(&self) -> impl Iterator<Item = &serde_json::Value> {
        self.facts.iter()
    }
}
