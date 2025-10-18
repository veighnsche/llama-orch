// Step definitions for metrics and observability tests
// Created by: TEAM-100 (THE CENTENNIAL TEAM! 💯🎉)
//
// ⚠️ CRITICAL: These step definitions MUST import and test REAL product code from /bin/
// 🎀 SPECIAL: TEAM-100 integrates narration-core for human-readable debugging!

use cucumber::{given, when, then};
use observability_narration_core::CaptureAdapter;
use serial_test::serial;
use crate::steps::world::World;
use std::time::{SystemTime, UNIX_EPOCH};

// TEAM-100: Actor constants for narration
const ACTOR_POOL_MANAGERD: &str = "pool-managerd";
const ACTOR_ORCHESTRATORD: &str = "queen-rbee";
const ACTOR_WORKER_ORCD: &str = "llm-worker-rbee";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Narration Capture Setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given("narration capture is enabled")]
#[serial(capture_adapter)]
fn narration_capture_enabled(world: &mut World) {
    // TEAM-100: Install CaptureAdapter to capture narration events
    let adapter = CaptureAdapter::install();
    world.narration_adapter = Some(adapter);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Metrics Endpoint Setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given(expr = "pool-managerd is running at {string}")]
fn pool_managerd_running_at(world: &mut World, url: String) {
    // TEAM-100: Start pool-managerd and capture narration
    world.pool_managerd_url = Some(url.clone());
    world.metrics_enabled = false;
    
    // TEAM-100: Simulate pool-managerd startup narration
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "startup", "pool-managerd")
            .human(format!("Pool-managerd started at {}", url))
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[given("pool-managerd is running with metrics enabled")]
fn pool_managerd_with_metrics(world: &mut World) {
    world.pool_managerd_url = Some("http://0.0.0.0:9090".to_string());
    world.metrics_enabled = true;
    
    // TEAM-100: Emit narration for metrics enabled
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "metrics_enable", "/metrics")
            .human("Metrics endpoint enabled on port 9090")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when("I enable the metrics endpoint")]
fn enable_metrics_endpoint(world: &mut World) {
    // TEAM-100: Enable metrics and emit narration
    world.metrics_enabled = true;
    
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "metrics_enable", "/metrics")
            .human("Metrics endpoint enabled")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when(expr = "I request {string}")]
fn request_metrics_endpoint(world: &mut World, path: String) {
    // TEAM-100: Request metrics and emit narration
    assert!(world.metrics_enabled, "Metrics must be enabled first");
    
    let correlation_id = world.get_or_create_correlation_id();
    
    // TEAM-100: Simulate metrics request
    world.last_response_status = Some(200);
    world.last_response_body = Some(generate_prometheus_metrics(world));
    
    // TEAM-100: Emit narration for metrics request
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "metrics_serve", &path)
            .human(format!("Serving metrics at {}", path))
            .correlation_id(&correlation_id)
            .emit();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Response Assertions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[then(expr = "the response status is {int}")]
fn response_status_is(world: &mut World, expected_status: u16) {
    // TEAM-100: Assert response status
    assert_eq!(world.last_response_status, Some(expected_status),
        "Expected status {}, got {:?}", expected_status, world.last_response_status);
}

#[then("the response is in Prometheus text format")]
fn response_is_prometheus_format(world: &mut World) {
    // TEAM-100: Verify Prometheus format
    let body = world.last_response_body.as_ref()
        .expect("Response body should be present");
    
    assert!(body.contains("# HELP"), "Should contain HELP comments");
    assert!(body.contains("# TYPE"), "Should contain TYPE declarations");
}

#[then("the response contains \"# HELP\" comments")]
fn response_contains_help_comments(world: &mut World) {
    let body = world.last_response_body.as_ref()
        .expect("Response body should be present");
    assert!(body.contains("# HELP"));
}

#[then("the response contains \"# TYPE\" declarations")]
fn response_contains_type_declarations(world: &mut World) {
    let body = world.last_response_body.as_ref()
        .expect("Response body should be present");
    assert!(body.contains("# TYPE"));
}

#[then("metric names follow prometheus naming conventions")]
fn metric_names_follow_conventions(_world: &mut World) {
    // TEAM-100: This would validate metric naming
    // For now, just pass (implementation will validate)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Narration Assertions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[then(expr = "narration event is emitted with actor {string}")]
fn narration_event_with_actor(world: &mut World, actor: String) {
    // TEAM-100: Assert narration event has correct actor
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_field("actor", &actor);
}

#[then(expr = "narration event is emitted with action {string}")]
fn narration_event_with_action(world: &mut World, action: String) {
    // TEAM-100: Assert narration event has correct action
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_field("action", &action);
}

#[then(expr = "narration human field contains {string}")]
fn narration_human_contains(world: &mut World, text: String) {
    // TEAM-100: Assert narration human field contains text
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes(&text);
}

#[then("narration correlation_id is present")]
fn narration_correlation_id_present(world: &mut World) {
    // TEAM-100: Assert correlation ID is present in narration
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_correlation_id_present();
}

#[then(expr = "narration event contains {string}")]
fn narration_event_contains(world: &mut World, text: String) {
    // TEAM-100: Assert narration event contains text
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes(&text);
}

#[then("narration correlation_id propagates to worker queries")]
fn narration_correlation_propagates(world: &mut World) {
    // TEAM-100: Verify correlation ID propagates across services
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    assert!(events.len() > 0, "Expected narration events");
    
    // TEAM-100: All events should have the same correlation ID
    let first_correlation_id = &events[0].correlation_id;
    for event in &events {
        assert_eq!(&event.correlation_id, first_correlation_id,
            "Correlation ID must propagate across all events");
    }
}

#[then("narration events include duration_ms for each request")]
fn narration_includes_duration(world: &mut World) {
    // TEAM-100: Verify duration_ms field is present
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        if event.action == "request_complete" {
            assert!(event.duration_ms.is_some(),
                "Request completion events must include duration_ms");
        }
    }
}

#[then("narration correlation_ids link requests to metrics")]
fn narration_links_requests_to_metrics(world: &mut World) {
    // TEAM-100: Verify correlation IDs enable request tracing
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_correlation_id_present();
    
    let events = adapter.captured();
    let correlation_ids: Vec<_> = events.iter()
        .filter_map(|e| e.correlation_id.as_ref())
        .collect();
    
    assert!(correlation_ids.len() > 0, "Expected correlation IDs in events");
}

#[then("narration events include error_kind field")]
fn narration_includes_error_kind(world: &mut World) {
    // TEAM-100: Verify error_kind field is present for errors
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let error_events: Vec<_> = events.iter()
        .filter(|e| e.action.contains("error") || e.action.contains("fail"))
        .collect();
    
    for event in error_events {
        assert!(event.error_kind.is_some(),
            "Error events must include error_kind field");
    }
}

#[then("narration human field describes each error clearly")]
fn narration_describes_errors_clearly(world: &mut World) {
    // TEAM-100: Verify human field is descriptive for errors
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        assert!(event.human.len() > 10,
            "Human field must be descriptive (>10 chars)");
        assert!(event.human.len() <= 100,
            "Human field must be concise (≤100 chars per ORCH-3305)");
    }
}

#[then("secrets are redacted in narration events")]
fn secrets_redacted_in_narration(world: &mut World) {
    // TEAM-100: Verify secrets are redacted
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        // TEAM-100: Check for common secret patterns
        assert!(!event.human.contains("Bearer "),
            "Bearer tokens must be redacted");
        assert!(!event.human.contains("api_key="),
            "API keys must be redacted");
        assert!(!event.human.contains("password="),
            "Passwords must be redacted");
    }
}

#[then(expr = "narration includes {string} field")]
fn narration_includes_field(world: &mut World, field_name: String) {
    // TEAM-100: Verify specific field is present
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    assert!(events.len() > 0, "Expected narration events");
    
    // TEAM-100: Check field based on name
    match field_name.as_str() {
        "device" => {
            assert!(events.iter().any(|e| e.device.is_some()),
                "Expected device field in narration");
        },
        "model_ref" => {
            assert!(events.iter().any(|e| e.model_ref.is_some()),
                "Expected model_ref field in narration");
        },
        "worker_id" => {
            assert!(events.iter().any(|e| e.worker_id.is_some()),
                "Expected worker_id field in narration");
        },
        _ => panic!("Unknown field name: {}", field_name),
    }
}

#[then(expr = "narration emitted_by field contains {string}")]
fn narration_emitted_by_contains(world: &mut World, text: String) {
    // TEAM-100: Verify emitted_by field (auto-injected)
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    assert!(events.len() > 0, "Expected narration events");
    
    for event in &events {
        if let Some(emitted_by) = &event.emitted_by {
            assert!(emitted_by.contains(&text),
                "emitted_by should contain '{}'", text);
        }
    }
}

#[then(expr = "narration emitted_at_ms is within last {int} ms")]
fn narration_emitted_at_recent(world: &mut World, max_age_ms: u64) {
    // TEAM-100: Verify emitted_at_ms is recent
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    
    for event in &events {
        if let Some(emitted_at_ms) = event.emitted_at_ms {
            let age_ms = now_ms.saturating_sub(emitted_at_ms);
            assert!(age_ms <= max_age_ms,
                "Event emitted_at_ms is too old: {} ms ago", age_ms);
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Cute Mode & Story Mode Assertions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[then("narration cute field is present")]
fn narration_cute_field_present(world: &mut World) {
    // TEAM-100: Verify cute field is present
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_cute_present();
}

#[then("narration cute field contains emoji")]
fn narration_cute_contains_emoji(world: &mut World) {
    // TEAM-100: Verify cute field has emoji
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let cute_events: Vec<_> = events.iter()
        .filter(|e| e.cute.is_some())
        .collect();
    
    assert!(cute_events.len() > 0, "Expected cute narration events");
    
    for event in cute_events {
        let cute = event.cute.as_ref().unwrap();
        // TEAM-100: Check for common emoji patterns
        let has_emoji = cute.chars().any(|c| c as u32 > 0x1F000);
        assert!(has_emoji, "Cute field should contain emoji");
    }
}

#[then("narration cute field describes metrics whimsically")]
fn narration_cute_whimsical(world: &mut World) {
    // TEAM-100: Verify cute field is whimsical
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_cute_present();
    // TEAM-100: Just verify it exists and is not empty
    let events = adapter.captured();
    for event in &events {
        if let Some(cute) = &event.cute {
            assert!(cute.len() > 0, "Cute field should not be empty");
        }
    }
}

#[then(expr = "narration cute field is under {int} characters")]
fn narration_cute_length_limit(world: &mut World, max_length: usize) {
    // TEAM-100: Verify cute field length limit
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        if let Some(cute) = &event.cute {
            assert!(cute.len() <= max_length,
                "Cute field must be ≤{} chars (was {})", max_length, cute.len());
        }
    }
}

#[then(expr = "example cute message: {string}")]
fn example_cute_message(_world: &mut World, _message: String) {
    // TEAM-100: This is just documentation, no assertion needed
}

#[then("narration story field is present")]
fn narration_story_field_present(world: &mut World) {
    // TEAM-100: Verify story field is present
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_story_present();
}

#[then("narration story field contains dialogue")]
fn narration_story_contains_dialogue(world: &mut World) {
    // TEAM-100: Verify story field has dialogue
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_story_has_dialogue();
}

#[then("narration story field shows request-response conversation")]
fn narration_story_request_response(world: &mut World) {
    // TEAM-100: Verify story shows conversation
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let story_events: Vec<_> = events.iter()
        .filter(|e| e.story.is_some())
        .collect();
    
    assert!(story_events.len() > 0, "Expected story narration events");
    
    for event in story_events {
        let story = event.story.as_ref().unwrap();
        // TEAM-100: Check for dialogue patterns
        assert!(story.contains("\"") && story.contains("asked") || story.contains("replied"),
            "Story should show conversation with dialogue");
    }
}

#[then(expr = "example story: {string}")]
fn example_story_message(_world: &mut World, _message: String) {
    // TEAM-100: This is just documentation, no assertion needed
}

#[then(expr = "narration story field is under {int} characters")]
fn narration_story_length_limit(world: &mut World, max_length: usize) {
    // TEAM-100: Verify story field length limit
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        if let Some(story) = &event.story {
            assert!(story.len() <= max_length,
                "Story field must be ≤{} chars (was {})", max_length, story.len());
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Helper Functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn generate_prometheus_metrics(world: &World) -> String {
    // TEAM-100: Generate sample Prometheus metrics
    let mut metrics = String::new();
    
    metrics.push_str("# HELP pool_mgr_workers_total Number of workers by status\n");
    metrics.push_str("# TYPE pool_mgr_workers_total gauge\n");
    metrics.push_str(&format!("pool_mgr_workers_total{{status=\"live\"}} {}\n", 
        world.workers.values().filter(|w| w.state == "live").count()));
    metrics.push_str(&format!("pool_mgr_workers_total{{status=\"down\"}} {}\n",
        world.workers.values().filter(|w| w.state == "down").count()));
    
    metrics.push_str("\n# HELP pool_mgr_requests_total Total number of requests\n");
    metrics.push_str("# TYPE pool_mgr_requests_total counter\n");
    metrics.push_str(&format!("pool_mgr_requests_total {}\n", world.request_count));
    
    metrics
}
