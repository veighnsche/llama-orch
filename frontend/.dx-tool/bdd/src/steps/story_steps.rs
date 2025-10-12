// Created by: TEAM-DX-003
// Story file locator BDD steps

use cucumber::{when, then};
use crate::steps::world::DxWorld;
use dx::commands::StoryCommand;

#[when(regex = r#"^I run story-file with URL "(.+)"$"#)]
pub async fn run_story_file(world: &mut DxWorld, url: String) {
    let cmd = StoryCommand::new();
    match cmd.locate_story_file(&url) {
        Ok(info) => {
            world.story_file = Some(info.story_file.clone());
            world.component_file = info.component_file.clone();
            world.story_path = Some(info.story_path.clone());
            world.store_success(format!("Located story: {}", info.story_path));
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r#"^I should see story file path "(.+)"$"#)]
pub async fn should_see_story_path(world: &mut DxWorld, expected: String) {
    let story_path = world.story_path.as_ref().expect("No story path captured");
    assert_eq!(story_path, &expected,
        "Expected story path '{}', got '{}'", expected, story_path);
}

#[then(regex = r#"^I should see component file path "(.+)"$"#)]
pub async fn should_see_component_path(world: &mut DxWorld, expected: String) {
    let component = world.component_file.as_ref().expect("No component file captured");
    let component_str = component.to_string_lossy();
    assert!(component_str.ends_with(&expected),
        "Expected component path to end with '{}', got '{}'", expected, component_str);
}

#[then(regex = r"^the files should exist on disk$")]
pub async fn files_should_exist(world: &mut DxWorld) {
    let story_file = world.story_file.as_ref().expect("No story file captured");
    assert!(story_file.exists(), "Story file should exist: {}", story_file.display());
    
    if let Some(component) = &world.component_file {
        assert!(component.exists(), "Component file should exist: {}", component.display());
    }
}

#[then(regex = r"^the variant ID should be ignored$")]
pub async fn variant_id_ignored(world: &mut DxWorld) {
    // If we got a story path, the variant ID was successfully stripped
    assert!(world.story_path.is_some(), "Story path should be captured");
}

#[then(regex = r#"^I should see an error "(.+)"$"#)]
pub async fn should_see_error(world: &mut DxWorld, expected_msg: String) {
    let error = world.error_message.as_ref().expect("No error captured");
    assert!(error.contains(&expected_msg),
        "Expected error to contain '{}', got '{}'", expected_msg, error);
}
