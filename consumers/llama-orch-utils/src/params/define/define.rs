// Placeholder module for applet params/define.
use serde::{Deserialize, Serialize};
#[cfg(feature = "ts-types")]
use ts_rs::TS;
#[cfg_attr(feature = "ts-types", derive(TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Params {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u64>,
}

pub fn run(p: Params) -> Params { p }
