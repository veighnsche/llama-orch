//! Mock adapter (stub-only, no I/O).

use contracts_api_types as api;
use futures::{stream, StreamExt};
use worker_adapters_adapter_api::{
    TokenEvent, TokenStream, WorkerAdapter, WorkerError, WorkerHealth, WorkerProps,
};

#[derive(Debug, Clone, Default)]
pub struct MockAdapter;

impl WorkerAdapter for MockAdapter {
    fn health(&self) -> Result<WorkerHealth, WorkerError> {
        Ok(WorkerHealth {
            live: true,
            ready: true,
        })
    }
    fn props(&self) -> Result<WorkerProps, WorkerError> {
        Ok(WorkerProps {
            slots_total: Some(1),
            slots_free: Some(1),
        })
    }
    fn submit(&self, _req: api::TaskRequest) -> Result<TokenStream, WorkerError> {
        let events = vec![
            Ok(TokenEvent {
                kind: "started".into(),
                data: serde_json::json!({}),
            }),
            Ok(TokenEvent {
                kind: "token".into(),
                data: serde_json::json!({"t":"hello","i":0}),
            }),
            Ok(TokenEvent {
                kind: "end".into(),
                data: serde_json::json!({"tokens_out":1,"decode_ms":0}),
            }),
        ];
        Ok(stream::iter(events).boxed())
    }
    fn cancel(&self, _task_id: &str) -> Result<(), WorkerError> {
        Ok(())
    }
    fn engine_version(&self) -> Result<String, WorkerError> {
        Ok("mock-stub-v0".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{executor::block_on, StreamExt};

    fn base_req() -> api::TaskRequest {
        api::TaskRequest {
            task_id: "t0".into(),
            session_id: "s0".into(),
            workload: api::Workload::Completion,
            model_ref: "m:v0".into(),
            engine: api::Engine::Llamacpp,
            ctx: 1,
            priority: api::Priority::Interactive,
            seed: None,
            determinism: None,
            sampler_profile_version: None,
            prompt: Some("hi".into()),
            inputs: None,
            max_tokens: 8,
            deadline_ms: 1,
            expected_tokens: Some(1),
            kv_hint: None,
        }
    }

    #[test]
    fn props_and_health_shape() {
        let a = MockAdapter;
        let h = a.health().expect("health");
        assert!(h.live && h.ready);
        let p = a.props().expect("props");
        assert_eq!(p.slots_total, Some(1));
        assert_eq!(p.slots_free, Some(1));
        let v = a.engine_version().expect("engine_version");
        assert!(v.contains("mock-stub"));
    }

    #[test]
    fn submit_yields_started_token_end() {
        let a = MockAdapter;
        let mut s = a.submit(base_req()).expect("stream");

        let ev1 = block_on(s.next()).expect("first").expect("ok");
        assert_eq!(ev1.kind, "started");

        let ev2 = block_on(s.next()).expect("second").expect("ok");
        assert_eq!(ev2.kind, "token");
        assert_eq!(ev2.data["t"], "hello");

        let ev3 = block_on(s.next()).expect("third").expect("ok");
        assert_eq!(ev3.kind, "end");
        assert_eq!(ev3.data["tokens_out"], 1);

        // No more events
        assert!(block_on(s.next()).is_none());
    }
}
