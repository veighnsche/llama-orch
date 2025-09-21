// Placeholder module for applet model/define.
#[derive(Debug, Clone)]
pub struct ModelRef {
    pub model_id: String,
    pub engine_id: Option<String>,
    pub pool_hint: Option<String>,
}

pub fn run(model_id: String, engine_id: Option<String>, pool_hint: Option<String>) -> ModelRef {
    ModelRef { model_id, engine_id, pool_hint }
}
