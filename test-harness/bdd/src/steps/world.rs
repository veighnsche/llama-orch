use anyhow::Result;
use async_trait::async_trait;
use cucumber::World as _; // trait in scope for derive or impl
use serde_json::json;

#[derive(Debug, Default)]
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

#[async_trait]
impl cucumber::World for World {
    type Error = anyhow::Error;

    async fn new() -> Result<Self> {
        Ok(Self::default())
    }
}
