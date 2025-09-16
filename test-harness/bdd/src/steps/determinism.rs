use crate::steps::world::World;
use contracts_api_types as api;
use cucumber::{given, then, when};
use futures::StreamExt;
use worker_adapters_adapter_api::WorkerAdapter;
use worker_adapters_llamacpp_http::LlamaCppHttpAdapter;

#[given(regex = r"^two replicas pin engine_version sampler_profile_version and model_digest$")]
pub async fn given_two_replicas_pinned_versions_artifacts(world: &mut World) {
    world.push_fact("det.pinned");
}

#[when(regex = r"^same prompt parameters and seed are used$")]
pub async fn when_same_prompt_params_seed(world: &mut World) {
    world.push_fact("det.same_params_seed");
}

#[then(regex = r"^token streams are byte-exact across replicas$")]
pub async fn then_token_streams_byte_exact(_world: &mut World) {
    // Simulate two replicas of the same adapter and assert identical token event sequences
    let a1 = LlamaCppHttpAdapter::new("http://r1");
    let a2 = LlamaCppHttpAdapter::new("http://r2");
    let req = api::TaskRequest {
        task_id: "t".into(),
        session_id: "s".into(),
        workload: api::Workload::Completion,
        model_ref: "model0".into(),
        engine: api::Engine::Llamacpp,
        ctx: 0,
        priority: api::Priority::Interactive,
        seed: Some(42),
        determinism: Some(api::DeterminismLevel::Strict),
        sampler_profile_version: Some("v1".into()),
        prompt: Some("hi".into()),
        inputs: None,
        max_tokens: 1,
        deadline_ms: 1000,
        expected_tokens: None,
        kv_hint: None,
    };
    let mut s1 = a1.submit(req.clone()).expect("submit ok");
    let mut s2 = a2.submit(req).expect("submit ok");
    let mut ev1 = Vec::new();
    let mut ev2 = Vec::new();
    while let Some(e) = s1.next().await {
        ev1.push(format!("{:?}", e.unwrap()));
    }
    while let Some(e) = s2.next().await {
        ev2.push(format!("{:?}", e.unwrap()));
    }
    assert_eq!(ev1, ev2, "token streams differ across replicas");
}

#[then(regex = r"^determinism is not assumed across engine or model updates$")]
pub async fn then_no_cross_version_determinism_assumed(_world: &mut World) {
    // Intentionally no assertion of equality across versions. Presence of this step indicates
    // we do not enforce cross-version byte-exactness.
}

#[given(regex = r"^replicas across engine or model versions are used$")]
pub async fn given_replicas_across_versions(world: &mut World) {
    world.push_fact("det.cross_version");
}
