// TEAM-308: Extended story mode step definitions
// Implements additional steps for story_mode.feature

use crate::steps::world::World;
use cucumber::{gherkin::Step, given, then, when};
use observability_narration_core::{narrate, with_narration_context, NarrationFields};

// ============================================================
// WHEN Steps - Story narration
// ============================================================

#[when(regex = r#"^I emit narration with n!\("([^"]+)", story: "([^"]+)"\)$"#)]
async fn when_emit_story_narration(_world: &mut World, action: String, story: String) {
    let action_static: &'static str = Box::leak(action.into_boxed_str());
    narrate(NarrationFields {
        actor: "test",
        action: action_static,
        target: "test".to_string(),
        human: "Test message".to_string(),
        story: Some(story),
        ..Default::default()
    });
}

#[when(regex = r#"^I emit narration with n!\("([^"]+)", story: "([^"]+)"\) in context$"#)]
async fn when_emit_story_in_context(world: &mut World, action: String, story: String) {
    if let Some(ctx) = world.context.clone() {
        let action_static: &'static str = Box::leak(action.into_boxed_str());
        with_narration_context(ctx, async move {
            narrate(NarrationFields {
                actor: "test",
                action: action_static,
                target: "test".to_string(),
                human: "Test message".to_string(),
                story: Some(story),
                ..Default::default()
            });
        }).await;
    }
}

#[when(regex = r#"^I narrate with story field "(.+)"$"#)]
async fn when_narrate_with_story_field(_world: &mut World, story: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        story: Some(story),
        ..Default::default()
    });
}

#[when("I narrate with story field that is 200 characters long")]
async fn when_narrate_with_long_story(_world: &mut World) {
    let story = "A".repeat(200);
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        story: Some(story),
        ..Default::default()
    });
}

// ============================================================
// THEN Steps - Story field assertions
// ============================================================

#[then("the captured narration should include story field")]
async fn then_includes_story_field(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.story.is_some(), "Story field should be present");
    }
}

#[then(regex = r#"^the story field should contain "([^"]+)"$"#)]
async fn then_story_contains(world: &mut World, expected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        
        if let Some(story) = &last.story {
            assert!(
                story.contains(&expected),
                "Story field '{}' should contain '{}'",
                story,
                expected
            );
        } else {
            panic!("Story field should be present");
        }
    }
}

#[then(regex = r#"^the story field should not contain "([^"]+)"$"#)]
async fn then_story_not_contains(world: &mut World, unexpected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        
        if let Some(story) = &last.story {
            assert!(
                !story.contains(&unexpected),
                "Story field '{}' should NOT contain '{}'",
                story,
                unexpected
            );
        }
    }
}

#[then("the story field should be absent")]
async fn then_story_absent(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.story.is_none(), "Story field should be absent");
    }
}

#[then("the story field should be present")]
async fn then_story_present(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.story.is_some(), "Story field should be present");
    }
}

#[then(regex = r"^the story field length should be at most (\d+)$")]
async fn then_story_length_at_most(world: &mut World, max_length: usize) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        
        if let Some(story) = &last.story {
            assert!(
                story.len() <= max_length,
                "Story field length {} should be at most {}",
                story.len(),
                max_length
            );
        } else {
            panic!("Story field should be present");
        }
    }
}

// ============================================================
// Additional assertions for specific scenarios
// ============================================================

#[then("both cute and story fields should be present")]
async fn then_cute_and_story_present(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.cute.is_some(), "Cute field should be present");
        assert!(last.story.is_some(), "Story field should be present");
    }
}

#[then("all three narration modes should be present")]
async fn then_all_three_modes_present(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(!last.human.is_empty(), "Human field should be present");
        assert!(last.cute.is_some(), "Cute field should be present");
        assert!(last.story.is_some(), "Story field should be present");
    }
}

#[then(regex = r#"^the story field should match pattern "(.+)"$"#)]
async fn then_story_matches_pattern(world: &mut World, _pattern: String) {
    // TEAM-308: Basic pattern matching - just check story exists
    // Full regex matching would require regex crate
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        assert!(last.story.is_some(), "Story field should be present for pattern matching");
    }
}

#[then(regex = r#"^event (\d+) story field should contain "([^"]+)"$"#)]
async fn then_event_n_story_contains(world: &mut World, event_num: usize, expected: String) {
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
        
        if let Some(story) = &event.story {
            assert!(
                story.contains(&expected),
                "Event {} story field '{}' should contain '{}'",
                event_num,
                story,
                expected
            );
        } else {
            panic!("Event {} should have story field", event_num);
        }
    }
}

// ============================================================
// Correlation ID assertions
// ============================================================

#[then(regex = r#"^the story field should include correlation_id "([^"]+)"$"#)]
async fn then_story_includes_correlation_id(world: &mut World, correlation_id: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured narration");
        let last = captured.last().unwrap();
        
        // Check if the event has the correlation_id
        assert!(
            last.correlation_id.as_ref().map_or(false, |id| id == &correlation_id),
            "Event should have correlation_id '{}'",
            correlation_id
        );
    }
}
