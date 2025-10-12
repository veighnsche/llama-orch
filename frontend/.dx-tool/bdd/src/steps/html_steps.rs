// Created by: TEAM-DX-003
// HTML command BDD steps

use cucumber::{when, then};
use crate::steps::world::{DxWorld, STORYBOOK_URL};
use dx::commands::HtmlCommand;

#[when(regex = r"^I query selector '(.+)'$")]
pub async fn query_selector(world: &mut DxWorld, selector: String) {
    let cmd = HtmlCommand::new();
    match cmd.query_selector(STORYBOOK_URL, &selector).await {
        Ok(info) => {
            world.element_tag = Some(info.tag);
            world.element_count = Some(info.count);
            world.store_success(format!("Found {} elements", info.count));
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r"^I should find (\d+) elements?$")]
pub async fn should_find_elements(world: &mut DxWorld, count: usize) {
    assert_eq!(world.element_count, Some(count),
        "Expected {} elements, got {:?}", count, world.element_count);
}

#[then(regex = r"^I should find at least (\d+) elements?$")]
pub async fn should_find_at_least_elements(world: &mut DxWorld, count: usize) {
    let actual = world.element_count.unwrap_or(0);
    assert!(actual >= count,
        "Expected at least {} elements, got {}", count, actual);
}

#[then(regex = r"^the element tag should be '(.+)'$")]
pub async fn element_tag_should_be(world: &mut DxWorld, tag: String) {
    assert_eq!(world.element_tag.as_deref(), Some(tag.as_str()),
        "Expected tag '{}', got {:?}", tag, world.element_tag);
}

#[when(regex = r"^I get attributes for selector '(.+)'$")]
pub async fn get_attributes(world: &mut DxWorld, selector: String) {
    let cmd = HtmlCommand::new();
    match cmd.get_attributes(STORYBOOK_URL, &selector).await {
        Ok(attrs) => {
            world.attributes = attrs;
            world.store_success(format!("Got {} attributes", world.attributes.len()));
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r"^I should see attribute '(.+)' with value '(.+)'$")]
pub async fn should_see_attribute(world: &mut DxWorld, attr: String, value: String) {
    let actual = world.attributes.get(&attr);
    assert_eq!(actual, Some(&value),
        "Expected attribute '{}' to be '{}', got {:?}", attr, value, actual);
}

#[then(regex = r"^I should see attribute '(.+)'$")]
pub async fn should_see_attribute_exists(world: &mut DxWorld, attr: String) {
    assert!(world.attributes.contains_key(&attr),
        "Expected attribute '{}' to exist, got {:?}", attr, world.attributes.keys());
}

#[when(regex = r"^I get DOM tree for selector '(.+)' with depth (\d+)$")]
pub async fn get_dom_tree(world: &mut DxWorld, selector: String, depth: usize) {
    let cmd = HtmlCommand::new();
    match cmd.get_tree(STORYBOOK_URL, &selector, depth).await {
        Ok(tree) => {
            world.dom_tree = Some(tree.node_repr());
            world.store_success("Got DOM tree".to_string());
        }
        Err(e) => {
            world.store_error(e.to_string());
        }
    }
}

#[then(regex = r"^the DOM tree should contain '(.+)'$")]
pub async fn dom_tree_should_contain(world: &mut DxWorld, text: String) {
    let tree = world.dom_tree.as_ref().expect("No DOM tree captured");
    assert!(tree.contains(&text),
        "Expected DOM tree to contain '{}', got: {}", text, tree);
}
