use std::time::SystemTime;

#[derive(Clone, Debug)]
pub struct Session {
    pub ttl_ms_remaining: i64,
    pub turns: i32,
    pub kv_bytes: i64,
    pub kv_warmth: bool,
    pub tokens_budget_remaining: Option<i64>,
    pub time_budget_remaining_ms: Option<i64>,
    pub cost_budget_remaining: Option<f64>,
    pub created_at: SystemTime,
}

impl Session {
    pub fn new_default() -> Self {
        Self {
            ttl_ms_remaining: 600_000,
            turns: 0,
            kv_bytes: 0,
            kv_warmth: false,
            tokens_budget_remaining: Some(50_000),
            time_budget_remaining_ms: Some(600_000),
            cost_budget_remaining: Some(1.0),
            created_at: SystemTime::now(),
        }
    }

    pub fn touch_turn(&mut self) {
        self.turns = self.turns.saturating_add(1);
    }

    pub fn spend_tokens(&mut self, tokens: i64) {
        if let Some(b) = self.tokens_budget_remaining.as_mut() {
            *b = (*b - tokens).max(0);
        }
    }

    pub fn spend_time_ms(&mut self, ms: i64) {
        if let Some(b) = self.time_budget_remaining_ms.as_mut() {
            *b = (*b - ms).max(0);
        }
        self.ttl_ms_remaining = (self.ttl_ms_remaining - ms).max(0);
    }

    pub fn spend_cost(&mut self, cost: f64) {
        if let Some(b) = self.cost_budget_remaining.as_mut() {
            *b = (*b - cost).max(0.0);
        }
    }
}
