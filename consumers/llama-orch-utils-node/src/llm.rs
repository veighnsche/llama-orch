use napi::bindgen_prelude::{Either, Null};
#[napi(object)]
pub struct SdkMsgNapi {
    pub role: String,
    pub content: String,
}

#[napi(object)]
pub struct InvokeInNapi {
    pub messages: Vec<SdkMsgNapi>,
    pub model: ModelRefInNapi,
    pub params: crate::params::ParamsNapi,
}

// Reuse shapes from orch.rs for result
pub use crate::orch::{ChoiceNapi, UsageNapi, InvokeResultNapi};

// Local mirror for input-only model ref to ensure Option<String> acceptance for null/undefined
#[napi(object)]
pub struct ModelRefInNapi {
    pub model_id: String,
    pub engine_id: Option<Either<String, Null>>,
    pub pool_hint: Option<Either<String, Null>>,
}

#[napi(object)]
pub struct InvokeOutNapi {
    pub result: InvokeResultNapi,
}
