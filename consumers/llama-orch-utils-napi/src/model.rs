use crate as _;
use napi::bindgen_prelude::{Either, Null};

#[napi(object)]
pub struct ModelRefNapi {
    pub model_id: String,
    pub engine_id: Option<Either<String, Null>>,
    pub pool_hint: Option<Either<String, Null>>,
}

impl From<llama_orch_utils::model::define::ModelRef> for ModelRefNapi {
    fn from(value: llama_orch_utils::model::define::ModelRef) -> Self {
        ModelRefNapi {
            model_id: value.model_id,
            engine_id: value.engine_id.map(Either::A),
            pool_hint: value.pool_hint.map(Either::A),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_define_roundtrip() {
        let core = llama_orch_utils::model::define::run("m".into(), None, None);
        let napi_ref: ModelRefNapi = core.into();
        assert_eq!(napi_ref.model_id, "m");
        assert!(napi_ref.engine_id.is_none());
        assert!(napi_ref.pool_hint.is_none());
    }

    #[test]
    fn model_define_with_overrides() {
        let core = llama_orch_utils::model::define::run("m2".into(), Some("llamacpp".into()), Some("pool-a".into()));
        let napi_ref: ModelRefNapi = core.into();
        assert_eq!(napi_ref.model_id, "m2");
        assert_eq!(napi_ref.engine_id.as_deref(), Some("llamacpp"));
        assert_eq!(napi_ref.pool_hint.as_deref(), Some("pool-a"));
    }
}
