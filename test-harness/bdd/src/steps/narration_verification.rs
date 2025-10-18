//! Narration Verification Steps
//!
//! TEAM-085: Verify that PRODUCT CODE emits proper narration
//!
//! These steps are used INSIDE existing feature files to verify
//! that the product code emits human-readable narration using
//! the observability-narration-core crate.
//!
//! The narration-core team has ultimate editorial authority.

use crate::steps::world::World;
use cucumber::then;
use observability_narration_core::CaptureAdapter;

#[then(expr = "I should see narration {string} from {string}")]
pub async fn then_see_narration_from(world: &mut World, text: String, actor: String) {
    // TEAM-085: Verify product code emitted narration
    // TEAM-102: Fixed method name from drain() to captured()
    let adapter = CaptureAdapter::install();
    let captured = adapter.captured();

    let found = captured.iter().any(|n| n.actor == actor && n.human.contains(&text));

    if !found {
        let available: Vec<String> =
            captured.iter().map(|n| format!("{}:{}", n.actor, n.human)).collect();

        panic!(
            "❌ PRODUCT CODE DID NOT EMIT NARRATION!\n\
             \n\
             Expected: actor='{}' with text='{}'\n\
             \n\
             Available narration:\n{:#?}\n\
             \n\
             💡 FIX: Add narrate() call in the product code:\n\
             \n\
             use observability_narration_core::{{narrate, NarrationFields}};\n\
             \n\
             narrate(NarrationFields {{\n\
                 actor: \"{}\",\n\
                 action: \"your_action\",\n\
                 target: \"your_target\".to_string(),\n\
                 human: \"{}\".to_string(),\n\
                 correlation_id: Some(correlation_id),\n\
                 ..Default::default()\n\
             }});",
            actor, text, available, actor, text
        );
    }

    tracing::info!("✅ Product code emitted narration: '{}' from '{}'", text, actor);
    world.last_narration = Some(captured);
}

#[then(expr = "I should see narration {string} with model_ref {string}")]
pub async fn then_see_narration_with_model_ref(world: &mut World, text: String, model_ref: String) {
    // TEAM-102: Fixed to use install() first
    let adapter = CaptureAdapter::install();
    let captured = adapter.captured();

    let found = captured.iter().any(|n| {
        n.human.contains(&text) && n.model_ref.as_ref().map_or(false, |m| m.contains(&model_ref))
    });

    if !found {
        panic!(
            "❌ PRODUCT CODE DID NOT EMIT NARRATION WITH model_ref!\n\
             \n\
             Expected: text='{}' with model_ref containing '{}'\n\
             \n\
             💡 FIX: Set model_ref field in NarrationFields",
            text, model_ref
        );
    }

    tracing::info!("✅ Narration includes model_ref: '{}'", model_ref);
    world.last_narration = Some(captured);
}

#[then(expr = "I should see narration {string} with duration_ms")]
pub async fn then_see_narration_with_duration(world: &mut World, text: String) {
    // TEAM-102: Fixed to use install() first
    let adapter = CaptureAdapter::install();
    let captured = adapter.captured();

    let found = captured.iter().any(|n| n.human.contains(&text) && n.duration_ms.is_some());

    if !found {
        panic!(
            "❌ PRODUCT CODE DID NOT INCLUDE duration_ms!\n\
             \n\
             Expected: text='{}' with duration_ms field\n\
             \n\
             💡 FIX: Measure timing and set duration_ms in NarrationFields",
            text
        );
    }

    tracing::info!("✅ Narration includes duration_ms");
    world.last_narration = Some(captured);
}

#[then(expr = "I should see narration {string} with worker_id {string}")]
pub async fn then_see_narration_with_worker_id(world: &mut World, text: String, worker_id: String) {
    // TEAM-102: Fixed to use install() first
    let adapter = CaptureAdapter::install();
    let captured = adapter.captured();

    let found = captured.iter().any(|n| {
        n.human.contains(&text) && n.worker_id.as_ref().map_or(false, |w| w.contains(&worker_id))
    });

    if !found {
        panic!(
            "❌ PRODUCT CODE DID NOT INCLUDE worker_id!\n\
             \n\
             Expected: text='{}' with worker_id='{}'\n\
             \n\
             💡 FIX: Set worker_id field in NarrationFields",
            text, worker_id
        );
    }

    tracing::info!("✅ Narration includes worker_id: '{}'", worker_id);
    world.last_narration = Some(captured);
}

#[then(expr = "narration should include error_kind {string}")]
pub async fn then_narration_includes_error_kind(world: &mut World, error_kind: String) {
    let narrations = world
        .last_narration
        .as_ref()
        .expect("No narration captured - call 'I should see narration' first");

    let found =
        narrations.iter().any(|n| n.error_kind.as_ref().map_or(false, |e| e == &error_kind));

    if !found {
        panic!(
            "❌ PRODUCT CODE DID NOT SET error_kind!\n\
             \n\
             Expected: error_kind='{}'\n\
             \n\
             💡 FIX: Set error_kind field in NarrationFields for error scenarios",
            error_kind
        );
    }

    tracing::info!("✅ Narration includes error_kind: '{}'", error_kind);
}

#[then(expr = "all narration events should have correlation IDs")]
pub async fn then_all_narration_has_correlation_ids(world: &mut World) {
    let narrations = world.last_narration.as_ref().expect("No narration captured");

    let missing: Vec<String> = narrations
        .iter()
        .filter(|n| n.correlation_id.is_none())
        .map(|n| format!("{}:{}", n.actor, n.action))
        .collect();

    if !missing.is_empty() {
        panic!(
            "❌ PRODUCT CODE MISSING CORRELATION IDs!\n\
             \n\
             Events without correlation_id: {:#?}\n\
             \n\
             💡 FIX: Propagate correlation_id through the call stack:\n\
             \n\
             1. Extract from HTTP headers: correlation_from_header()\n\
             2. Pass through function calls\n\
             3. Include in all narrate() calls",
            missing
        );
    }

    tracing::info!("✅ All narration events have correlation IDs");
}

#[then(expr = "narration human field should be under 100 characters")]
pub async fn then_narration_under_100_chars(world: &mut World) {
    let narrations = world.last_narration.as_ref().expect("No narration captured");

    let too_long: Vec<(String, usize)> = narrations
        .iter()
        .filter(|n| n.human.len() > 100)
        .map(|n| (n.human.clone(), n.human.len()))
        .collect();

    if !too_long.is_empty() {
        panic!(
            "❌ NARRATION TOO LONG! (Editorial Standard: ≤100 chars)\n\
             \n\
             Violations:\n{:#?}\n\
             \n\
             💡 FIX: The narration-core team has ultimate editorial authority.\n\
             Rewrite to be concise while keeping it informative.\n\
             Use abbreviations, remove filler words, focus on key info.",
            too_long
        );
    }

    tracing::info!("✅ All narration under 100 characters (editorial standard)");
}
