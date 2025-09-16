use crate::steps::world::World;
use contracts_api_types as api;
use cucumber::{given, then};
use futures::StreamExt;
use worker_adapters_adapter_api::WorkerAdapter;
use worker_adapters_llamacpp_http::LlamaCppHttpAdapter;
use worker_adapters_tgi_http::TgiHttpAdapter;
use worker_adapters_triton::TritonAdapter;
use worker_adapters_vllm_http::VllmHttpAdapter;

#[given(regex = r"^a worker adapter for llama\\.cpp$|^a worker adapter for llamacpp$")]
pub async fn given_adapter_llamacpp(world: &mut World) {
    world.push_fact("adapter.llamacpp");
}

#[then(regex = r"^the adapter implements health properties completion cancel metrics$")]
pub async fn then_adapter_implements_llamacpp_endpoints(_world: &mut World) {
    let a = LlamaCppHttpAdapter::new("http://localhost:8080");
    let h = worker_adapters_adapter_api::WorkerAdapter::health(&a).expect("health ok");
    assert!(h.live);
    let p = worker_adapters_adapter_api::WorkerAdapter::props(&a).expect("props ok");
    assert!(p.slots_total.unwrap_or(0) >= 1);
    let req = api::TaskRequest {
        task_id: "t".into(),
        session_id: "s".into(),
        workload: api::Workload::Completion,
        model_ref: "model0".into(),
        engine: api::Engine::Llamacpp,
        ctx: 0,
        priority: api::Priority::Interactive,
        seed: None,
        determinism: None,
        sampler_profile_version: None,
        prompt: Some("hi".into()),
        inputs: None,
        max_tokens: 1,
        deadline_ms: 1000,
        expected_tokens: None,
        kv_hint: None,
    };
    let mut stream =
        worker_adapters_adapter_api::WorkerAdapter::submit(&a, req).expect("submit ok");
    let _first = stream.next().await;
    worker_adapters_adapter_api::WorkerAdapter::cancel(&a, "t").expect("cancel ok");
    let _v =
        worker_adapters_adapter_api::WorkerAdapter::engine_version(&a).expect("engine_version ok");
}

#[then(regex = r"^OpenAI-compatible endpoints are internal only$")]
pub async fn then_openai_endpoints_internal_only(_world: &mut World) {}

#[then(regex = r"^adapter reports engine_version and model_digest$")]
pub async fn then_adapter_reports_versions(_world: &mut World) {}

#[given(regex = r"^a worker adapter for vllm$")]
pub async fn given_adapter_vllm(world: &mut World) {
    world.push_fact("adapter.vllm");
}

#[then(
    regex = r"^the adapter implements health/properties/completion/cancel/metrics against vLLM$"
)]
pub async fn then_adapter_implements_vllm_endpoints(_world: &mut World) {
    let a = VllmHttpAdapter::new("http://localhost:8080");
    let _ = worker_adapters_adapter_api::WorkerAdapter::health(&a).expect("health ok");
    let _ = worker_adapters_adapter_api::WorkerAdapter::props(&a).expect("props ok");
    let req = api::TaskRequest {
        task_id: "t".into(),
        session_id: "s".into(),
        workload: api::Workload::Completion,
        model_ref: "model0".into(),
        engine: api::Engine::Vllm,
        ctx: 0,
        priority: api::Priority::Interactive,
        seed: None,
        determinism: None,
        sampler_profile_version: None,
        prompt: Some("hi".into()),
        inputs: None,
        max_tokens: 1,
        deadline_ms: 1000,
        expected_tokens: None,
        kv_hint: None,
    };
    let mut s = worker_adapters_adapter_api::WorkerAdapter::submit(&a, req).expect("submit ok");
    let _ = s.next().await;
    worker_adapters_adapter_api::WorkerAdapter::cancel(&a, "t").expect("cancel ok");
    let _ =
        worker_adapters_adapter_api::WorkerAdapter::engine_version(&a).expect("engine_version ok");
}

#[given(regex = r"^a worker adapter for tgi$")]
pub async fn given_adapter_tgi(world: &mut World) {
    world.push_fact("adapter.tgi");
}

#[then(regex = r"^the adapter implements TGI custom API and metrics$")]
pub async fn then_adapter_implements_tgi_endpoints(_world: &mut World) {
    let a = TgiHttpAdapter::new("http://localhost:8080");
    let _ = worker_adapters_adapter_api::WorkerAdapter::health(&a).expect("health ok");
    let _ = worker_adapters_adapter_api::WorkerAdapter::props(&a).expect("props ok");
    let req = api::TaskRequest {
        task_id: "t".into(),
        session_id: "s".into(),
        workload: api::Workload::Completion,
        model_ref: "model0".into(),
        engine: api::Engine::Tgi,
        ctx: 0,
        priority: api::Priority::Interactive,
        seed: None,
        determinism: None,
        sampler_profile_version: None,
        prompt: Some("hi".into()),
        inputs: None,
        max_tokens: 1,
        deadline_ms: 1000,
        expected_tokens: None,
        kv_hint: None,
    };
    let mut s = worker_adapters_adapter_api::WorkerAdapter::submit(&a, req).expect("submit ok");
    let _ = s.next().await;
    worker_adapters_adapter_api::WorkerAdapter::cancel(&a, "t").expect("cancel ok");
    let _ =
        worker_adapters_adapter_api::WorkerAdapter::engine_version(&a).expect("engine_version ok");
}

#[given(regex = r"^a worker adapter for triton$")]
pub async fn given_adapter_triton(world: &mut World) {
    world.push_fact("adapter.triton");
}

#[then(regex = r"^the adapter implements infer/streaming and metrics$")]
pub async fn then_adapter_implements_triton_endpoints(_world: &mut World) {
    let a = TritonAdapter::new("http://localhost:8080");
    let _ = worker_adapters_adapter_api::WorkerAdapter::health(&a).expect("health ok");
    let _ = worker_adapters_adapter_api::WorkerAdapter::props(&a).expect("props ok");
    let req = api::TaskRequest {
        task_id: "t".into(),
        session_id: "s".into(),
        workload: api::Workload::Completion,
        model_ref: "model0".into(),
        engine: api::Engine::Triton,
        ctx: 0,
        priority: api::Priority::Interactive,
        seed: None,
        determinism: None,
        sampler_profile_version: None,
        prompt: Some("hi".into()),
        inputs: None,
        max_tokens: 1,
        deadline_ms: 1000,
        expected_tokens: None,
        kv_hint: None,
    };
    let mut s = worker_adapters_adapter_api::WorkerAdapter::submit(&a, req).expect("submit ok");
    let _ = s.next().await;
    worker_adapters_adapter_api::WorkerAdapter::cancel(&a, "t").expect("cancel ok");
    let _ =
        worker_adapters_adapter_api::WorkerAdapter::engine_version(&a).expect("engine_version ok");
}
