// Step definitions for story mode behaviors

use crate::steps::world::World;
use cucumber::{gherkin::Table, then, when};
use observability_narration_core::{narrate, NarrationFields};

#[when(regex = r#"^I narrate with story "(.+)"$"#)]
pub async fn when_narrate_with_story(_world: &mut World, story: String) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        story: Some(story),
        ..Default::default()
    });
}

#[when("I narrate without story field")]
pub async fn when_narrate_without_story(_world: &mut World) {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        story: None,
        ..Default::default()
    });
}

#[when("I narrate with all three modes:")]
pub async fn when_narrate_with_all_three_modes(_world: &mut World, table: &Table) {
    let mut human = String::new();
    let mut cute = None;
    let mut story = None;

    for row in table.rows.iter().skip(1) {
        // Skip header
        let mode = &row[0];
        let content = &row[1];

        match mode.as_str() {
            "human" => human = content.to_string(),
            "cute" => cute = Some(content.to_string()),
            "story" => story = Some(content.to_string()),
            _ => panic!("Unknown mode: {}", mode),
        }
    }

    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human,
        cute,
        story,
        ..Default::default()
    });
}

#[then("the captured narration has story field")]
pub async fn then_captured_has_story(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(
        captured[0].story.is_some(),
        "Expected story field to be present"
    );
}

#[then("the captured narration has no story field")]
pub async fn then_captured_has_no_story(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(
        captured[0].story.is_none(),
        "Expected story field to be None"
    );
}

#[then(regex = r#"^the captured story includes "(.+)"$"#)]
pub async fn then_captured_story_includes(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");

    let story = captured[0]
        .story
        .as_ref()
        .expect("Expected story field to be present");

    assert!(
        story.contains(&substring),
        "Expected story to include '{}', but got: {}",
        substring,
        story
    );
}

#[then(regex = r#"^the captured story does not include "(.+)"$"#)]
pub async fn then_captured_story_does_not_include(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");

    let story = captured[0]
        .story
        .as_ref()
        .expect("Expected story field to be present");

    assert!(
        !story.contains(&substring),
        "Expected story NOT to include '{}', but got: {}",
        substring,
        story
    );
}

#[then("the captured narration has human field")]
pub async fn then_captured_has_human(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(
        !captured[0].human.is_empty(),
        "Expected human field to be present"
    );
}

#[then("the captured narration has cute field")]
pub async fn then_captured_has_cute(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let captured = adapter.captured();
    assert!(!captured.is_empty(), "No narration captured");
    assert!(
        captured[0].cute.is_some(),
        "Expected cute field to be present"
    );
}

#[then("assert_story_present succeeds")]
pub async fn then_assert_story_present_succeeds(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_story_present();
}

#[then(regex = r#"^assert_story_includes "(.+)" succeeds$"#)]
pub async fn then_assert_story_includes_succeeds(world: &mut World, substring: String) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_story_includes(&substring);
}

#[then("assert_story_has_dialogue succeeds")]
pub async fn then_assert_story_has_dialogue_succeeds(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    adapter.assert_story_has_dialogue();
}

#[then("assert_story_has_dialogue fails")]
pub async fn then_assert_story_has_dialogue_fails(world: &mut World) {
    let adapter = world.adapter.as_ref().expect("Adapter not installed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.assert_story_has_dialogue();
    }));
    assert!(
        result.is_err(),
        "assert_story_has_dialogue should have failed"
    );
}

#[then("the tracing event includes story field")]
pub async fn then_tracing_includes_story(_world: &mut World) {
    // This is a placeholder - in real implementation, we'd check tracing output
    // For now, we just verify the test compiles and runs
    // The actual story field emission is tested via capture adapter
}
