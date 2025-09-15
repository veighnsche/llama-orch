use tools_openapi_client as cli;
use contracts_api_types as api;

fn main() {
    let c = cli::Client::new("http://localhost:8080");
    let req = api::TaskRequest {
        task_id: "11111111-1111-4111-8111-111111111111".into(),
        session_id: "22222222-2222-4222-8222-222222222222".into(),
        workload: api::Workload::Completion,
        model_ref: "repo:NousResearch/Meta-Llama-3-8B-Instruct".into(),
        engine: api::Engine::Llamacpp,
        ctx: 8192,
        priority: api::Priority::Interactive,
        seed: Some(123),
        determinism: Some(api::DeterminismLevel::Strict),
        sampler_profile_version: Some("v1".into()),
        prompt: Some("hello".into()),
        inputs: None,
        max_tokens: 1,
        deadline_ms: 1000,
        expected_tokens: Some(1),
        kv_hint: Some(api::KVHint::Reuse),
    };
    let _rb = c.create_task(&req);
}
