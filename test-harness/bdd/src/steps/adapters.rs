use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        // llama.cpp adapter
        Regex::new(r"^a worker adapter for llama.cpp$|^a worker adapter for llamacpp$").unwrap(),
        Regex::new(r"^the adapter implements health properties completion cancel metrics$").unwrap(),
        Regex::new(r"^OpenAI-compatible endpoints are internal only$").unwrap(),
        Regex::new(r"^adapter reports engine_version and model_digest$").unwrap(),
        // vLLM adapter
        Regex::new(r"^a worker adapter for vllm$").unwrap(),
        Regex::new(r"^the adapter implements health/properties/completion/cancel/metrics against vLLM$").unwrap(),
        // TGI adapter
        Regex::new(r"^a worker adapter for tgi$").unwrap(),
        Regex::new(r"^the adapter implements TGI custom API and metrics$").unwrap(),
        // Triton adapter
        Regex::new(r"^a worker adapter for triton$").unwrap(),
        Regex::new(r"^the adapter implements infer/streaming and metrics$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_adapter_llamacpp(_w: &mut World) {}
    pub fn then_adapter_implements_llamacpp_endpoints(_w: &mut World) {}
    pub fn then_openai_endpoints_internal_only(_w: &mut World) {}
    pub fn then_adapter_reports_versions(_w: &mut World) {}
    pub fn given_adapter_vllm(_w: &mut World) {}
    pub fn then_adapter_implements_vllm_endpoints(_w: &mut World) {}
    pub fn given_adapter_tgi(_w: &mut World) {}
    pub fn then_adapter_implements_tgi_endpoints(_w: &mut World) {}
    pub fn given_adapter_triton(_w: &mut World) {}
    pub fn then_adapter_implements_triton_endpoints(_w: &mut World) {}
}
