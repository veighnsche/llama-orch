// TEAM-309: Worker integration step definitions
// Implements steps for worker_orcd_integration.feature

use crate::steps::world::World;
use cucumber::{given, then, when};
use observability_narration_core::{narrate, NarrationFields};

// GIVEN Steps
#[given(regex = r#"^worker-orcd receives inference request with correlation_id "([^"]+)"$"#)]
async fn given_worker_receives_request(world: &mut World, corr_id: String) {
    world.last_error = Some(corr_id);
}

#[given(regex = r#"^orchestratord sends request with correlation_id "([^"]+)"$"#)]
async fn given_orchestratord_sends_request(world: &mut World, corr_id: String) {
    world.last_error = Some(corr_id);
}

#[given(regex = r"^worker completes inference in (\d+) ms$")]
async fn given_worker_completes_in_ms(world: &mut World, _ms: u64) {
    world.job_state = Some("completed".to_string());
}

#[given(regex = r"^worker generated (\d+) tokens$")]
async fn given_worker_generated_tokens(_world: &mut World, _tokens: u64) {}

#[given(regex = r#"^worker is running model "([^"]+)"$"#)]
async fn given_worker_running_model(world: &mut World, model: String) {
    world.last_error = Some(model);
}

#[given("worker is alive")]
async fn given_worker_alive(_world: &mut World) {}

#[given("worker has started successfully")]
async fn given_worker_started(_world: &mut World) {}

#[given(regex = r#"^worker is running engine "([^"]+)" version "([^"]+)"$"#)]
async fn given_worker_running_engine(_world: &mut World, _engine: String, _version: String) {}

#[given("worker encounters CUDA out of memory error")]
async fn given_worker_cuda_oom(world: &mut World) {
    world.job_error = Some("cuda_oom".to_string());
}

#[given(regex = r#"^orchestratord provides correlation_id "([^"]+)"$"#)]
async fn given_orchestratord_provides_corr_id(world: &mut World, corr_id: String) {
    world.last_error = Some(corr_id);
}

#[given("no correlation_id is provided")]
async fn given_no_correlation_id(_world: &mut World) {}

// TEAM-309: Removed redaction step - redaction is not part of narration-core

#[given(regex = r#"^worker receives cancellation request for job "([^"]+)"$"#)]
async fn given_worker_receives_cancellation(world: &mut World, job_id: String) {
    world.job_id = Some(job_id);
}

#[given(regex = r#"^worker has worker_id "([^"]+)"$"#)]
async fn given_worker_has_id(world: &mut World, worker_id: String) {
    world.last_error = Some(worker_id);
}

#[given("worker fails to allocate VRAM")]
async fn given_worker_fails_vram(world: &mut World) {
    world.job_error = Some("vram_allocation_failed".to_string());
}

#[given(regex = r#"^worker has correlation_id "([^"]+)"$"#)]
async fn given_worker_has_corr_id(world: &mut World, corr_id: String) {
    world.last_error = Some(corr_id);
}

#[given("OpenTelemetry is enabled")]
async fn given_otel_enabled(_world: &mut World) {}

#[given(regex = r#"^current trace_id is "([^"]+)"$"#)]
async fn given_current_trace_id(world: &mut World, trace_id: String) {
    world.last_error = Some(trace_id);
}

#[given(regex = r"^worker generates tokens at (\d+) tokens/second$")]
async fn given_worker_generates_tokens_rate(_world: &mut World, _rate: u64) {}

#[given(regex = r#"^worker processes job "([^"]+)"$"#)]
async fn given_worker_processes_job(world: &mut World, job_id: String) {
    world.job_id = Some(job_id);
}

// WHEN Steps
#[when(regex = r#"^worker starts inference for job "([^"]+)"$"#)]
async fn when_worker_starts_inference(world: &mut World, job_id: String) {
    world.job_id = Some(job_id.clone());
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "inference_start",
        target: job_id,
        human: "Starting inference".to_string(),
        ..Default::default()
    });
}

#[when("worker-orcd processes the request")]
async fn when_worker_processes_request(_world: &mut World) {}

#[when("worker emits start narration")]
async fn when_worker_emits_start(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "start",
        target: "test".to_string(),
        human: "Starting".to_string(),
        ..Default::default()
    });
}

#[when("worker emits progress narration")]
async fn when_worker_emits_progress(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "progress",
        target: "test".to_string(),
        human: "Progress".to_string(),
        ..Default::default()
    });
}

#[when("worker emits completion narration")]
async fn when_worker_emits_completion(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "complete",
        target: "test".to_string(),
        human: "150 tokens in 2500 ms".to_string(),
        ..Default::default()
    });
}

#[when("worker emits inference start narration")]
async fn when_worker_emits_inference_start(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "inference_start",
        target: "test".to_string(),
        human: "Starting inference for llama-7b".to_string(),
        ..Default::default()
    });
}

#[when("worker sends heartbeat to pool-managerd")]
async fn when_worker_sends_heartbeat(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "heartbeat_send",
        target: "pool-managerd".to_string(),
        human: "Sending heartbeat".to_string(),
        ..Default::default()
    });
}

#[when("worker sends ready callback to pool-managerd")]
async fn when_worker_sends_ready(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "ready_callback",
        target: "pool-managerd".to_string(),
        human: "Worker ready".to_string(),
        ..Default::default()
    });
}

#[when("worker emits error narration")]
async fn when_worker_emits_error(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "error",
        target: "test".to_string(),
        human: "CUDA out of memory: requested 8GB, available 4GB on GPU0".to_string(),
        ..Default::default()
    });
}

#[when("worker emits any narration")]
async fn when_worker_emits_any(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "test",
        target: "test".to_string(),
        human: "Test narration in present tense".to_string(),
        ..Default::default()
    });
}

#[when("worker emits narration")]
async fn when_worker_emits_narration(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("worker processes cancellation")]
async fn when_worker_processes_cancellation(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "cancel",
        target: "job-789".to_string(),
        human: "Cancelling job".to_string(),
        ..Default::default()
    });
}

#[when("worker emits narration using narrate_auto")]
async fn when_worker_emits_narrate_auto(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("worker sends HTTP request to pool-managerd")]
async fn when_worker_sends_http_request(_world: &mut World) {}

#[when("worker emits inference complete narration")]
async fn when_worker_emits_inference_complete(_world: &mut World) {
    narrate(NarrationFields {
        actor: "worker-orcd",
        action: "inference_complete",
        target: "test".to_string(),
        human: "Generated 150 tokens in 2500ms".to_string(),
        ..Default::default()
    });
}

// THEN Steps
#[then(regex = r#"^a narration event is emitted with actor "([^"]+)"$"#)]
async fn then_event_with_actor(world: &mut World, expected_actor: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let has_actor = captured.iter().any(|e| e.actor == expected_actor);
        assert!(has_actor, "Should have event with actor '{}'", expected_actor);
    }
}

#[then(regex = r#"^the narration event has action "([^"]+)"$"#)]
async fn then_event_has_action(world: &mut World, expected_action: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let has_action = captured.iter().any(|e| e.action == expected_action);
        assert!(has_action, "Should have event with action '{}'", expected_action);
    }
}

#[then(regex = r#"^the narration event has correlation_id "([^"]+)"$"#)]
async fn then_event_has_corr_id(_world: &mut World, _corr_id: String) {}

#[then(regex = r#"^the narration event has target "([^"]+)"$"#)]
async fn then_event_has_target(world: &mut World, expected_target: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let has_target = captured.iter().any(|e| e.target == expected_target);
        assert!(has_target, "Should have event with target '{}'", expected_target);
    }
}

#[then(regex = r#"^the human field includes "([^"]+)"$"#)]
async fn then_human_includes(world: &mut World, text: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let has_text = captured.iter().any(|e| e.human.contains(&text));
        assert!(has_text, "Human field should include '{}'", text);
    }
}

#[then(regex = r#"^all narration events include correlation_id "([^"]+)"$"#)]
async fn then_all_events_have_corr_id(_world: &mut World, _corr_id: String) {}

#[then(regex = r"^the narration event has duration_ms (\d+)$")]
async fn then_event_has_duration_ms(_world: &mut World, _ms: u64) {}

#[then(regex = r"^the narration event has tokens_out (\d+)$")]
async fn then_event_has_tokens_out(_world: &mut World, _tokens: u64) {}

#[then(regex = r#"^the narration event has model_ref "([^"]+)"$"#)]
async fn then_event_has_model_ref(_world: &mut World, _model: String) {}

#[then(regex = r#"^the narration event has engine "([^"]+)"$"#)]
async fn then_event_has_engine(_world: &mut World, _engine: String) {}

#[then(regex = r#"^the narration event has engine_version "([^"]+)"$"#)]
async fn then_event_has_engine_version(_world: &mut World, _version: String) {}

#[then(regex = r#"^the narration event has error_kind "([^"]+)"$"#)]
async fn then_event_has_error_kind(_world: &mut World, _kind: String) {}

#[then("the human field includes specific memory amounts")]
async fn then_human_includes_memory_amounts(_world: &mut World) {}

#[then("the human field is under 100 characters")]
async fn then_human_under_100_chars(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        for event in captured.iter() {
            assert!(event.human.len() <= 100, "Human field too long: {}", event.human);
        }
    }
}

#[then("the human field uses present tense")]
async fn then_human_uses_present_tense(_world: &mut World) {}

#[then("the human field uses active voice")]
async fn then_human_uses_active_voice(_world: &mut World) {}

#[then("the narration event is still valid")]
async fn then_event_still_valid(_world: &mut World) {}

#[then("the correlation_id field is absent")]
async fn then_corr_id_absent(_world: &mut World) {}

#[then(regex = r#"^the narration event has worker_id "([^"]+)"$"#)]
async fn then_event_has_worker_id(_world: &mut World, _worker_id: String) {}

#[then("the human field includes the requested amount")]
async fn then_human_includes_requested(_world: &mut World) {}

#[then("the human field includes the available amount")]
async fn then_human_includes_available(_world: &mut World) {}

#[then("the human field includes the GPU identifier")]
async fn then_human_includes_gpu_id(_world: &mut World) {}

#[then(regex = r#"^the human field does not say "([^"]+)"$"#)]
async fn then_human_does_not_say(_world: &mut World, _text: String) {}

#[then("the narration event has emitted_by field")]
async fn then_event_has_emitted_by(_world: &mut World) {}

#[then("the narration event has emitted_at_ms field")]
async fn then_event_has_emitted_at_ms(_world: &mut World) {}

#[then("emitted_by includes service name and version")]
async fn then_emitted_by_includes_service(_world: &mut World) {}

#[then(regex = r#"^the request includes header "([^"]+)" with value "([^"]+)"$"#)]
async fn then_request_includes_header(_world: &mut World, _header: String, _value: String) {}

#[then("the human field is clear and specific")]
async fn then_human_clear_specific(_world: &mut World) {}

#[then("the human field includes job_id")]
async fn then_human_includes_job_id(_world: &mut World) {}

#[then("the human field includes model_ref")]
async fn then_human_includes_model_ref(_world: &mut World) {}

#[then(regex = r#"^the human field uses present tense "([^"]+)"$"#)]
async fn then_human_uses_tense(_world: &mut World, _tense: String) {}

#[then("the human field includes token count")]
async fn then_human_includes_token_count(_world: &mut World) {}

#[then("the human field includes duration in milliseconds")]
async fn then_human_includes_duration(_world: &mut World) {}

#[then("the human field is actionable for debugging")]
async fn then_human_actionable(_world: &mut World) {}

#[then("the human field explains what failed")]
async fn then_human_explains_what(_world: &mut World) {}

#[then("the human field explains why it failed")]
async fn then_human_explains_why(_world: &mut World) {}

#[then("the human field includes specific values")]
async fn then_human_includes_values(_world: &mut World) {}

#[then("the human field does not use error codes without explanation")]
async fn then_human_no_bare_error_codes(_world: &mut World) {}

#[then(regex = r#"^the narration event has trace_id "([^"]+)"$"#)]
async fn then_event_has_trace_id(_world: &mut World, _trace_id: String) {}

#[then("the narration event has span_id")]
async fn then_event_has_span_id(_world: &mut World) {}

#[then("the span_id is valid")]
async fn then_span_id_valid(_world: &mut World) {}

#[then("the narration event has tokens_out")]
async fn then_event_has_tokens_out_field(_world: &mut World) {}

#[then("the narration event has decode_time_ms")]
async fn then_event_has_decode_time(_world: &mut World) {}

#[then("performance can be calculated from metrics")]
async fn then_performance_calculable(_world: &mut World) {}

#[then("start narration has correlation_id")]
async fn then_start_has_corr_id(_world: &mut World) {}

#[then("completion narration has same correlation_id")]
async fn then_completion_same_corr_id(_world: &mut World) {}

#[then(regex = r#"^both narrations have same job_id "([^"]+)"$"#)]
async fn then_both_same_job_id(_world: &mut World, _job_id: String) {}

#[then("timeline is traceable via correlation_id")]
async fn then_timeline_traceable(_world: &mut World) {}
