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

pub fn run(p: Params) -> Params {
    // M2 DRAFT normalization rules:
    // - temperature: default 0.7; clamp to [0.0, 2.0]
    // - top_p: default 1.0; clamp to [0.0, 1.0]
    // - max_tokens: default 1024 (explicit for M2)
    // - seed: passthrough (no default)

    fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
        if x < lo { lo } else if x > hi { hi } else { x }
    }

    let temperature = clamp_f32(p.temperature.unwrap_or(0.7), 0.0, 2.0);
    let top_p = clamp_f32(p.top_p.unwrap_or(1.0), 0.0, 1.0);
    let max_tokens = p.max_tokens.unwrap_or(1024);

    Params {
        temperature: Some(temperature),
        top_p: Some(top_p),
        max_tokens: Some(max_tokens),
        seed: p.seed,
    }
}
