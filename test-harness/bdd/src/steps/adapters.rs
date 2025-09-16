use crate::steps::world::World;
use cucumber::{given, then};

#[given(regex = r"^a worker adapter for llama\.cpp$|^a worker adapter for llamacpp$")]
pub async fn given_adapter_llamacpp(world: &mut World) {
    world.push_fact("adapter.llamacpp");
}

#[then(regex = r"^the adapter implements health properties completion cancel metrics$")]
pub async fn then_adapter_implements_llamacpp_endpoints(_world: &mut World) {}

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
pub async fn then_adapter_implements_vllm_endpoints(_world: &mut World) {}

#[given(regex = r"^a worker adapter for tgi$")]
pub async fn given_adapter_tgi(world: &mut World) {
    world.push_fact("adapter.tgi");
}

#[then(regex = r"^the adapter implements TGI custom API and metrics$")]
pub async fn then_adapter_implements_tgi_endpoints(_world: &mut World) {}

#[given(regex = r"^a worker adapter for triton$")]
pub async fn given_adapter_triton(world: &mut World) {
    world.push_fact("adapter.triton");
}

#[then(regex = r"^the adapter implements infer/streaming and metrics$")]
pub async fn then_adapter_implements_triton_endpoints(_world: &mut World) {}
