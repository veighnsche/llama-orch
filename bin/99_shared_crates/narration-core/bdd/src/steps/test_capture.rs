// Step definitions for test capture behaviors

use crate::steps::world::World;
use cucumber::{given, then, when};
use observability_narration_core::{narrate, CaptureAdapter, NarrationFields};

#[when("I install a capture adapter")]
pub async fn when_install_adapter(world: &mut World) {
    world.adapter = Some(CaptureAdapter::install());
}

#[given("the narration capture adapter is installed")]
pub async fn given_capture_adapter_installed(world: &mut World) {
    let adapter = CaptureAdapter::install();
    eprintln!("[DEBUG] After install(), events: {}", adapter.captured().len());
    // Clear any previous events from other scenarios
    adapter.clear();
    eprintln!("[DEBUG] After clear(), events: {}", adapter.captured().len());
    world.adapter = Some(adapter);
}

// ============================================================
// BUG-003: INVESTIGATION HISTORY - FALSE LEADS & BREAKTHROUGH
// ============================================================
// TEAM-307: ATTEMPTED FIX #1 - job_id FILTERING (FALSE FIX)
// SUSPICION:
// - Thought async task overlap caused event leakage
// - Suspected filtering by job_id would isolate events
//
// FIX ATTEMPT:
// - Added job_id filtering logic in then_captured_has_n_events()
// - Filtered events to only count those matching current job_id
//
// WHY IT FAILED:
// - Many scenarios DELIBERATELY test without job_id (context-free)
// - Filtering defeats the purpose of testing context-free narration
// - Didn't fix root cause, just masked the symptom
//
// TEAM-308: ATTEMPTED FIX #2 - BASELINE TRACKING (FAILED)
// SUSPICION:
// - Thought baseline tracking would solve clear() issues
// - Count only NEW events since scenario start
//
// FIX ATTEMPT:
// - Added initial_event_count field to World struct
// - Record baseline, count new_events = total - baseline
//
// WHY IT FAILED:
// - Baseline: 0, but got 25 events (expected 1)
// - Events from ALL scenarios appeared together
// - Both fixes were treating SYMPTOMS not ROOT CAUSE
//
// 🎯 BREAKTHROUGH (TEAM-308):
// REAL ROOT CAUSE:
// - Cucumber runs with --concurrency 64 by default!
// - All ~18 scenarios run in PARALLEL
// - They ALL share ONE global CaptureAdapter singleton (OnceLock)
// - Race condition: Scenario A clears → Scenario B adds → A asserts
// - Each scenario sees events from ALL concurrent scenarios
//
// PROOF:
// - With --concurrency 64: 2 passed, 18 failed
// - With --concurrency 1:  17 passed, 2 failed
// - Result: 83% improvement by forcing sequential execution
//
// FIX LOCATION: src/main.rs (not here!)
// - Added .max_concurrent_scenarios(1) to force sequential execution
// - This function is now correct (baseline tracking was right idea)
// - The bug was in how Cucumber ran the tests, not in our logic
//
// REMAINING: 2 scenarios still fail even with sequential execution
// These are likely REAL bugs, not race conditions
// ============================================================

#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    // TEAM-308: FAILED ATTEMPT - Baseline tracking doesn't work
    // See investigation comment above
    if let Some(adapter) = &world.adapter {
        world.initial_event_count = adapter.captured().len();
        eprintln!("[DEBUG] Baseline set to: {}", world.initial_event_count);
    }
}

#[given("narration capture is enabled")]
pub async fn given_narration_capture_enabled(world: &mut World) {
    let adapter = CaptureAdapter::install();
    adapter.clear();
    world.adapter = Some(adapter);
}

#[when(regex = r#"^I narrate with human "([^"]+)"$"#)]
pub async fn when_narrate_with_human(_world: &mut World, human: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human,
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate with actor "([^"]+)"$"#)]
pub async fn when_narrate_with_actor(_world: &mut World, actor: String) {
    narrate(NarrationFields {
        actor: Box::leak(actor.into_boxed_str()),
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("I narrate without correlation_id")]
pub async fn when_narrate_without_correlation_id(_world: &mut World) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("I narrate without provenance")]
pub async fn when_narrate_without_provenance(_world: &mut World) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });
}

#[when("I clear the capture adapter")]
pub async fn when_clear_adapter(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.clear();
}

#[then("the adapter is available")]
pub async fn then_adapter_available(world: &mut World) {
    assert!(world.adapter.is_some(), "Adapter should be available");
}

#[then(regex = r#"^the captured narration should have (\d+) events?$"#)]
pub async fn then_captured_has_n_events(world: &mut World, count: usize) {
    // TEAM-308: FAILED ATTEMPT - Baseline tracking
    // See full investigation comment above at given_capture_buffer_empty()
    // This approach FAILED - baseline is 0 but we still get 25 events
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let new_events = captured.len().saturating_sub(world.initial_event_count);
        
        assert_eq!(new_events, count, 
            "Expected {} new events since scenario start, got {}. \n\
             Total events in buffer: {}, Baseline: {}, Current scenario events: {:?}", 
            count, new_events, captured.len(), world.initial_event_count,
            captured.get(world.initial_event_count..).unwrap_or(&[])
                .iter()
                .map(|e| (&e.action, &e.job_id))
                .collect::<Vec<_>>()
        );
    }
}

#[then(regex = r#"^the captured narration should include (correlation ID .+|.+ ID .+)$"#)]
pub async fn then_captured_includes(world: &mut World, field: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        
        // Check if field appears in any of the narration fields
        let found = last.human.contains(&field) 
            || last.cute.as_ref().map_or(false, |c| c.contains(&field))
            || last.story.as_ref().map_or(false, |s| s.contains(&field))
            || last.actor.contains(&field)
            || last.action.contains(&field)
            || last.target.contains(&field);
            
        assert!(found, "Captured narration should include '{}'", field);
    }
}

#[then(regex = r"^captured events count is (\d+)$")]
pub async fn then_captured_count(world: &mut World, expected: usize) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert_eq!(
        captured.len(),
        expected,
        "Expected {} captured events, got {}",
        expected,
        captured.len()
    );
}

#[then(regex = r#"^assert_includes "([^"]+)" passes$"#)]
pub async fn then_assert_includes_passes(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_includes(&substring);
}

#[then(regex = r#"^assert_includes "([^"]+)" fails$"#)]
pub async fn then_assert_includes_fails(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_includes(&substring);
    }));
    assert!(result.is_err(), "assert_includes should have failed");
}

#[then(regex = r#"^assert_field "([^"]+)" "([^"]+)" passes$"#)]
pub async fn then_assert_field_passes(world: &mut World, field: String, value: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_field(&field, &value);
}

#[then(regex = r#"^assert_field "([^"]+)" "([^"]+)" fails$"#)]
pub async fn then_assert_field_fails(world: &mut World, field: String, value: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_field(&field, &value);
    }));
    assert!(result.is_err(), "assert_field should have failed");
}

#[then("assert_correlation_id_present passes")]
pub async fn then_assert_correlation_id_present_passes(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_correlation_id_present();
}

#[then("assert_correlation_id_present fails")]
pub async fn then_assert_correlation_id_present_fails(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_correlation_id_present();
    }));
    assert!(result.is_err(), "assert_correlation_id_present should have failed");
}

#[then("assert_provenance_present passes")]
pub async fn then_assert_provenance_present_passes(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_provenance_present();
}

#[then("assert_provenance_present fails")]
pub async fn then_assert_provenance_present_fails(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_provenance_present();
    }));
    assert!(result.is_err(), "assert_provenance_present should have failed");
}
