#![allow(clippy::needless_return)]

#[macro_use]
extern crate napi_derive;

use napi::bindgen_prelude::*;
use napi::Status;

mod fs;
mod prompt;
mod model;
mod params;
mod orch;
mod llm;

#[napi]
pub fn probe() -> String {
    return "llama-orch-utils-node-ok".to_string();
}

#[napi(namespace = "fs")]
pub fn read_file(input: fs::ReadRequestNapi) -> napi::Result<fs::ReadResponseNapi> {
    let core_req: llama_orch_utils::fs::file_reader::ReadRequest = input.into();
    match llama_orch_utils::fs::file_reader::run(core_req) {
        Ok(resp) => Ok(resp.into()),
        Err(e) => Err(napi::Error::new(Status::GenericFailure, format!("fs.readFile failed: {}", e))),
    }
}

#[napi(namespace = "prompt")]
pub fn message(input: prompt::MessageInNapi) -> napi::Result<prompt::MessageNapi> {
    let core_in = llama_orch_utils::prompt::message::MessageIn::try_from(input)
        .map_err(|e| napi::Error::new(Status::GenericFailure, format!("prompt.message failed: {}", e)))?;
    match llama_orch_utils::prompt::message::run(core_in) {
        Ok(out) => Ok(out.into()),
        Err(e) => Err(napi::Error::new(Status::GenericFailure, format!("prompt.message failed: {}", e))),
    }
}

#[napi(namespace = "prompt")]
pub fn thread(input: prompt::ThreadInNapi) -> napi::Result<prompt::ThreadOutNapi> {
    let core_in = llama_orch_utils::prompt::thread::ThreadIn::try_from(input)
        .map_err(|e| napi::Error::new(Status::GenericFailure, format!("prompt.thread failed: {}", e)))?;
    match llama_orch_utils::prompt::thread::run(core_in) {
        Ok(out) => Ok(out.into()),
        Err(e) => Err(napi::Error::new(Status::GenericFailure, format!("prompt.thread failed: {}", e))),
    }
}

// model.define(model_id, engine_id?, pool_hint?) -> ModelRef
#[napi(namespace = "model", js_name = "define")]
pub fn model_define(
    model_id: String,
    engine_id: Option<String>,
    pool_hint: Option<String>,
) -> napi::Result<model::ModelRefNapi> {
    let m = llama_orch_utils::model::define::run(model_id, engine_id, pool_hint);
    Ok(model::ModelRefNapi::from(m))
}

// params.define(p: Params) -> Params
#[napi(namespace = "params", js_name = "define")]
pub fn params_define(input: params::ParamsNapi) -> napi::Result<params::ParamsNapi> {
    let core_in: llama_orch_utils::params::define::Params = input.into();
    let core_out = llama_orch_utils::params::define::run(core_in);
    Ok(core_out.into())
}

// orch.response_extractor(result: InvokeResult) -> string
#[napi(namespace = "orch")]
pub fn response_extractor(result: orch::InvokeResultNapi) -> napi::Result<String> {
    let core: llama_orch_utils::llm::invoke::InvokeResult = result.into();
    Ok(llama_orch_utils::orch::response_extractor::run(&core))
}

#[napi(namespace = "llm")]
pub fn invoke(_input: llm::InvokeInNapi) -> napi::Result<llm::InvokeOutNapi> {
    Err(napi::Error::from_reason(
        "unimplemented: llm.invoke requires SDK wiring".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_expected_string() {
        assert_eq!(probe(), "llama-orch-utils-node-ok");
    }

    // Compile-time signature drift guard: if any export's signature changes,
    // this test will fail to compile when running `cargo test`.
    #[test]
    fn napi_signature_guard() {
        // fs namespace
        let _rf: fn(fs::ReadRequestNapi) -> napi::Result<fs::ReadResponseNapi> = read_file;

        // prompt namespace
        let _pm: fn(prompt::MessageInNapi) -> napi::Result<prompt::MessageNapi> = message;
        let _pt: fn(prompt::ThreadInNapi) -> napi::Result<prompt::ThreadOutNapi> = thread;

        // model namespace
        let _md: fn(String, Option<String>, Option<String>) -> napi::Result<model::ModelRefNapi> = model_define;

        // params namespace
        let _pd: fn(params::ParamsNapi) -> napi::Result<params::ParamsNapi> = params_define;

        // orch namespace
        let _ore: fn(orch::InvokeResultNapi) -> napi::Result<String> = response_extractor;

        // llm namespace
        let _llmi: fn(llm::InvokeInNapi) -> napi::Result<llm::InvokeOutNapi> = invoke;

        // probe (root)
        let _p: fn() -> String = probe;

        let _ = (_rf, _pm, _pt, _md, _pd, _ore, _llmi, _p);
    }
}
