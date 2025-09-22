use napi::bindgen_prelude::{Either, Null};

#[napi(object)]
pub struct ParamsNapi {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<i64>,
    pub seed: Option<Either<i64, Null>>, // allow explicit null
}

impl From<ParamsNapi> for llama_orch_utils::params::define::Params {
    fn from(value: ParamsNapi) -> Self {
        // Safe, total conversion. Floats downcast to f32. Integers: negative -> 0.
        let mt = value.max_tokens.map(|v| if v < 0 { 0 } else { v as u32 });
        let seed = match value.seed {
            Some(Either::A(v)) => Some(if v < 0 { 0 } else { v as u64 }),
            Some(Either::B(_)) | None => None,
        };
        llama_orch_utils::params::define::Params {
            temperature: value.temperature.map(|x| x as f32),
            top_p: value.top_p.map(|x| x as f32),
            max_tokens: mt,
            seed,
        }
    }
}

impl From<llama_orch_utils::params::define::Params> for ParamsNapi {
    fn from(value: llama_orch_utils::params::define::Params) -> Self {
        ParamsNapi {
            temperature: value.temperature.map(|x| x as f64),
            top_p: value.top_p.map(|x| x as f64),
            max_tokens: value.max_tokens.map(|x| x as i64),
            seed: value.seed.map(|x| Either::A(x as i64)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_clamps_apply() {
        // Nones -> defaults (0.7, 1.0, 1024), seed passthrough None
        let params = ParamsNapi { temperature: None, top_p: None, max_tokens: None, seed: None };
        let core_in: llama_orch_utils::params::define::Params = params.into();
        let core_out = llama_orch_utils::params::define::run(core_in);
        let out: ParamsNapi = core_out.into();
        // Allow for f32->f64 rounding differences
        let t = out.temperature.unwrap();
        assert!((t - 0.7).abs() < 1e-6, "temperature approx equality failed: {}", t);
        let tp = out.top_p.unwrap();
        assert!((tp - 1.0).abs() < 1e-9);
        assert_eq!(out.max_tokens, Some(1024));
        assert_eq!(out.seed, None);
    }

    #[test]
    fn boundary_values_are_normalized() {
        // temperature clamps to [0.0, 2.0]; top_p clamps to [0.0, 1.0]
        let params = ParamsNapi { temperature: Some(10.0), top_p: Some(-1.0), max_tokens: Some(-50), seed: Some(-3) };
        let core_in: llama_orch_utils::params::define::Params = params.into();
        let core_out = llama_orch_utils::params::define::run(core_in);
        let out: ParamsNapi = core_out.into();
        let t = out.temperature.unwrap();
        assert!((t - 2.0).abs() < 1e-9);
        let tp = out.top_p.unwrap();
        assert!((tp - 0.0).abs() < 1e-9);
        // our conversion makes negatives -> 0 before core; core keeps provided 0
        assert_eq!(out.max_tokens, Some(0));
        assert_eq!(out.seed, Some(0));
    }
}
