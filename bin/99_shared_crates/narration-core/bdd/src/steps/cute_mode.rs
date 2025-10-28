// TEAM-308: Cute mode step definitions
// Implements steps for cute_mode.feature

use crate::steps::world::World;
use cucumber::{gherkin::Step, given, then, when};
use observability_narration_core::{
    narrate, with_narration_context, CaptureAdapter, NarrationFields,
};
use std::collections::HashMap;

// ============================================================
// WHEN Steps - Emitting cute narration
// ============================================================

#[when(regex = r#"^I emit narration with n!\("([^"]+)", "([^"]+)"\)$"#)]
async fn when_emit_simple_narration(_world: &mut World, action: String, message: String) {
    let action_static: &'static str = Box::leak(action.into_boxed_str());
    narrate(NarrationFields {
        actor: "test",
        action: action_static,
        target: "test".to_string(),
        human: message,
        ..Default::default()
    });
}

#[when(regex = r#"^I emit narration with n!\("([^"]+)", cute: "([^"]+)"\)$"#)]
async fn when_emit_cute_narration(_world: &mut World, action: String, cute: String) {
    let action_static: &'static str = Box::leak(action.into_boxed_str());
    narrate(NarrationFields {
        actor: "test",
        action: action_static,
        target: "test".to_string(),
        human: "Test message".to_string(),
        cute: Some(cute),
        ..Default::default()
    });
}

#[when("I narrate with:")]
async fn when_narrate_with_table(_world: &mut World, step: &Step) {
    if let Some(table) = &step.table {
        let mut fields = HashMap::new();

        // Parse table into HashMap
        for row in &table.rows[1..] {
            // Skip header row
            if row.len() >= 2 {
                fields.insert(row[0].clone(), row[1].clone());
            }
        }

        // Extract fields with defaults
        let actor = fields.get("actor").map(|s| s.as_str()).unwrap_or("test");
        let action = fields.get("action").map(|s| s.as_str()).unwrap_or("test");
        let target = fields.get("target").cloned().unwrap_or_else(|| "test".to_string());
        let human = fields.get("human").cloned().unwrap_or_else(|| "Test message".to_string());
        let cute = fields.get("cute").cloned();
        let story = fields.get("story").cloned();

        // Leak actor and action strings for 'static lifetime
        let actor_static: &'static str = Box::leak(actor.to_string().into_boxed_str());
        let action_static: &'static str = Box::leak(action.to_string().into_boxed_str());

        narrate(NarrationFields {
            actor: actor_static,
            action: action_static,
            target,
            human,
            cute,
            story,
            ..Default::default()
        });
    }
}

#[when(regex = r#"^I emit narration with n!\("([^"]+)", cute: "([^"]+)"\) in context$"#)]
async fn when_emit_cute_in_context(world: &mut World, action: String, cute: String) {
    if let Some(ctx) = world.context.clone() {
        let action_static: &'static str = Box::leak(action.into_boxed_str());
        with_narration_context(ctx, async move {
            narrate(NarrationFields {
                actor: "test",
                action: action_static,
                target: "test".to_string(),
                human: "Test message".to_string(),
                cute: Some(cute),
                ..Default::default()
            });
        })
        .await;
    }
}

#[when("I narrate without cute field")]
async fn when_narrate_without_cute(_world: &mut World) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        cute: None,
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate with cute field "([^"]+)"$"#)]
async fn when_narrate_with_cute(_world: &mut World, cute: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        cute: Some(cute),
        ..Default::default()
    });
}

#[when("I narrate with cute field that is 150 characters long")]
async fn when_narrate_with_long_cute(_world: &mut World) {
    let cute = "A".repeat(150);
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        cute: Some(cute),
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate at WARN level with cute field "([^"]+)"$"#)]
async fn when_narrate_warn_with_cute(_world: &mut World, cute: String) {
    // TEAM-308: Note - NarrationFields doesn't have a level field
    // This step just emits with cute field, level is handled elsewhere
    narrate(NarrationFields {
        actor: "test",
        action: "warn",
        target: "test".to_string(),
        human: "Test message".to_string(),
        cute: Some(cute),
        ..Default::default()
    });
}

#[when(regex = r#"^I narrate at ERROR level with cute field "([^"]+)"$"#)]
async fn when_narrate_error_with_cute(_world: &mut World, cute: String) {
    // TEAM-308: Note - NarrationFields doesn't have a level field
    // This step just emits with cute field, level is handled elsewhere
    narrate(NarrationFields {
        actor: "test",
        action: "error",
        target: "test".to_string(),
        human: "Test message".to_string(),
        cute: Some(cute),
        ..Default::default()
    });
}

// ============================================================
// THEN Steps - Assertions on cute field
// ============================================================

#[then("the captured narration should include cute field")]
async fn then_includes_cute_field(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.cute.is_some(), "Cute field should be present");
    }
}

#[then(regex = r#"^the cute field should contain "([^"]+)"$"#)]
async fn then_cute_contains(world: &mut World, expected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();

        if let Some(cute) = &last.cute {
            assert!(
                cute.contains(&expected),
                "Cute field '{}' should contain '{}'",
                cute,
                expected
            );
        } else {
            panic!("Cute field should be present");
        }
    }
}

#[then(regex = r#"^the cute field should not contain "([^"]+)"$"#)]
async fn then_cute_not_contains(world: &mut World, unexpected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();

        if let Some(cute) = &last.cute {
            assert!(
                !cute.contains(&unexpected),
                "Cute field '{}' should NOT contain '{}'",
                cute,
                unexpected
            );
        }
    }
}

#[then("the cute field should be absent")]
async fn then_cute_absent(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.cute.is_none(), "Cute field should be absent");
    }
}

#[then("the cute field should be present")]
async fn then_cute_present(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.cute.is_some(), "Cute field should be present");
    }
}

#[then(regex = r"^the cute field length should be at most (\d+)$")]
async fn then_cute_length_at_most(world: &mut World, max_length: usize) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();

        if let Some(cute) = &last.cute {
            assert!(
                cute.len() <= max_length,
                "Cute field length {} should be at most {}",
                cute.len(),
                max_length
            );
        } else {
            panic!("Cute field should be present");
        }
    }
}

#[then(regex = r#"^event (\d+) cute field should contain "([^"]+)"$"#)]
async fn then_event_n_cute_contains(world: &mut World, event_num: usize, expected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let index = event_num - 1; // Convert to 0-indexed

        assert!(
            index < captured.len(),
            "Event {} does not exist (only {} events captured)",
            event_num,
            captured.len()
        );

        let event = &captured[world.initial_event_count + index];

        if let Some(cute) = &event.cute {
            assert!(
                cute.contains(&expected),
                "Event {} cute field '{}' should contain '{}'",
                event_num,
                cute,
                expected
            );
        } else {
            panic!("Event {} should have cute field", event_num);
        }
    }
}

// ============================================================
// Additional helper steps
// ============================================================

#[then(regex = r#"^the human field should contain "([^"]+)"$"#)]
async fn then_human_contains(world: &mut World, expected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();

        assert!(
            last.human.contains(&expected),
            "Human field '{}' should contain '{}'",
            last.human,
            expected
        );
    }
}
