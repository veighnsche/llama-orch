// Created by: TEAM-DX-003
// CSS command BDD steps

use cucumber::{given, when, then};
use crate::steps::world::{DxWorld, STORYBOOK_URL};
use dx::commands::CssCommand;

#[given(regex = r"^Storybook is running on port 6006$")]
pub async fn storybook_running(_world: &mut DxWorld) {
    // Assumption: Storybook must be running
    // Could add a health check here if needed
}

#[when(regex = r"^I check if class '(.+)' exists$")]
pub async fn check_class_exists(world: &mut DxWorld, class_name: String) {
    let cmd = CssCommand::new();
    match cmd.check_class_exists(STORYBOOK_URL, &class_name).await {
        Ok(exists) => {
            world.class_exists = Some(exists);
            world.store_success(format!("Class '{}' exists: {}", class_name, exists));
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r"^the class should exist$")]
pub async fn class_should_exist(world: &mut DxWorld) {
    assert!(world.class_exists.unwrap_or(false), "Class should exist");
}

#[then(regex = r"^the class should not exist$")]
pub async fn class_should_not_exist(world: &mut DxWorld) {
    assert!(!world.class_exists.unwrap_or(true), "Class should not exist");
}

#[when(regex = r"^I get styles for selector '(.+)'$")]
pub async fn get_selector_styles(world: &mut DxWorld, selector: String) {
    let cmd = CssCommand::new();
    match cmd.get_selector_styles(STORYBOOK_URL, &selector).await {
        Ok(styles) => {
            world.styles = styles;
            world.store_success(format!("Got {} styles", world.styles.len()));
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r"^I should see style '(.+)' with value '(.+)'$")]
pub async fn should_see_style(world: &mut DxWorld, property: String, value: String) {
    let actual = world.styles.get(&property);
    assert_eq!(actual, Some(&value), 
        "Expected style '{}' to be '{}', got {:?}", property, value, actual);
}

#[when(regex = r"^I list classes for selector '(.+)'$")]
pub async fn list_classes(world: &mut DxWorld, selector: String) {
    let cmd = CssCommand::new();
    match cmd.list_classes(STORYBOOK_URL, &selector).await {
        Ok(classes) => {
            world.classes = classes;
            world.store_success(format!("Got {} classes", world.classes.len()));
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r"^I should see class '(.+)'$")]
pub async fn should_see_class(world: &mut DxWorld, class_name: String) {
    assert!(world.classes.contains(&class_name),
        "Expected to find class '{}' in {:?}", class_name, world.classes);
}

#[then(regex = r"^I should see at least (\d+) classes?$")]
pub async fn should_see_at_least_classes(world: &mut DxWorld, count: usize) {
    assert!(world.classes.len() >= count,
        "Expected at least {} classes, got {}", count, world.classes.len());
}
