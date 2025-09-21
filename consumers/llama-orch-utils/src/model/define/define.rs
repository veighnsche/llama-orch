// Placeholder module for applet model/define.
use serde::{Deserialize, Serialize};
#[cfg(feature = "ts-types")]
use ts_rs::TS;
#[cfg_attr(feature = "ts-types", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub model_id: String,
    pub engine_id: Option<String>,
    pub pool_hint: Option<String>,
}

pub fn run(model_id: String, engine_id: Option<String>, pool_hint: Option<String>) -> ModelRef {
    ModelRef { model_id, engine_id, pool_hint }
}
