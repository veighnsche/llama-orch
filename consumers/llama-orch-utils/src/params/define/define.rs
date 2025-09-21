// Placeholder module for applet params/define.
#[derive(Debug, Clone, Default)]
pub struct Params {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u64>,
}

pub fn run(p: Params) -> Params { p }
