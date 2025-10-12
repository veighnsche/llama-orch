# TEAM-DX-003 HANDOFF

**From:** TEAM-DX-002  
**To:** TEAM-DX-003  
**Date:** 2025-10-12  
**Status:** Phase 2 Complete, BDD Scaffold Required  
**Priority:** P1 - Wire BDD integration testing

---

## Mission: Wire BDD Scaffold for Integration Testing

**Goal:** Create a BDD test harness for the `dx` CLI tool using Storybook (port 6006) as the testing ground for integration tests.

**Why:** Currently we have 78 unit tests but only 8 ignored integration tests. We need a proper BDD scaffold to test real-world scenarios against a running Storybook server.

---

## What This Tool Is For

**The DX tool solves a critical problem for remote/SSH engineers:**

Engineers working **without browser access** (SSH, remote environments, CI/CD) need a simple CLI to:
- Verify CSS classes exist (Tailwind generation)
- Extract computed styles for elements
- Query DOM structure
- Inspect HTML attributes
- Visualize component trees

**Without this tool:** Engineers must manually `curl` HTML, parse it by hand, and guess where components are defined.

**With this tool:** Engineers get instant, actionable information about what's rendered and where to make changes.

---

## Current State

### What TEAM-DX-002 Delivered

**Phase 2 Complete:**
- ✅ 78 unit tests passing (CSS + HTML commands)
- ✅ All modules have comprehensive test coverage
- ✅ 8 integration tests marked as `#[ignore]` (require running server)
- ✅ Binary builds successfully (~5.8 MB)
- ✅ JSON output format implemented

**Integration Tests (Currently Ignored):**
```rust
// In src/commands/css.rs
#[tokio::test]
#[ignore]
async fn test_get_selector_styles() { ... }

#[tokio::test]
#[ignore]
async fn test_check_class_exists() { ... }

#[tokio::test]
#[ignore]
async fn test_list_classes() { ... }

// In src/commands/html.rs
#[tokio::test]
#[ignore]
async fn test_query_selector() { ... }

#[tokio::test]
#[ignore]
async fn test_get_attributes() { ... }

#[tokio::test]
#[ignore]
async fn test_get_tree() { ... }

// In src/fetcher/client.rs
#[tokio::test]
#[ignore]
async fn test_fetch_page_success() { ... }

#[tokio::test]
#[ignore]
async fn test_fetch_stylesheet_success() { ... }
```

---

## CRITICAL NEW FEATURE REQUIRED

### Story File Locator Command

**User Story:**
> "As a frontend engineer without browser access, when I want to add more stories to a Storybook page, I need the CLI to tell me which file defines that story so I know exactly where to make changes."

**Example Workflow:**

```bash
# Engineer sees this Storybook URL and wants to add more stories
dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-0"

# Output:
✓ Story file located
  URL: http://localhost:6006/story/stories-atoms-button-button-story-vue
  File: /home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Button/Button.story.vue
  Component: Button
  
  To add more stories, edit:
    - Story file: stories/atoms/Button/Button.story.vue
    - Component: stories/atoms/Button/Button.vue (if needed)
```

**Implementation Requirements:**

1. **New command:** `dx story-file <URL>`
2. **Parse Storybook URL** to extract story path:
   - URL: `http://localhost:6006/story/stories-atoms-button-button-story-vue`
   - Story path: `stories/atoms/Button/Button.story.vue`
3. **Resolve to filesystem path:**
   - Base: `frontend/libs/storybook/`
   - Full path: `/home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Button/Button.story.vue`
4. **Detect related files:**
   - Story file: `Button.story.vue`
   - Component file: `Button.vue` (same directory)
5. **Output format:**
   - Text: Human-readable with file paths
   - JSON: `{"story_file": "...", "component_file": "...", "directory": "..."}`

**BDD Scenarios Required:**

```gherkin
Feature: Story File Locator
  As a frontend engineer without browser access
  I want to find which file defines a Storybook story
  So I can quickly make changes

  Background:
    Given Storybook is running on port 6006

  Scenario: Locate Button story file
    When I run story-file with URL "http://localhost:6006/story/stories-atoms-button-button-story-vue"
    Then I should see story file path "stories/atoms/Button/Button.story.vue"
    And I should see component file path "stories/atoms/Button/Button.vue"
    And the files should exist on disk

  Scenario: Locate story with variant ID
    When I run story-file with URL "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-0"
    Then I should see story file path "stories/atoms/Button/Button.story.vue"
    And the variant ID should be ignored

  Scenario: Invalid story URL
    When I run story-file with URL "http://localhost:6006/invalid"
    Then I should see an error "Could not parse story path from URL"

  Scenario: Story file not found
    When I run story-file with URL "http://localhost:6006/story/nonexistent-story"
    Then I should see an error "Story file not found on disk"
```

**This is Priority 0 - Must be implemented and tested in BDD before other scenarios.**

---

## Your Mission: BDD Scaffold

### Target Server

**Use Storybook (Histoire) on port 6006:**
- Location: `frontend/libs/storybook`
- Start command: `pnpm story:dev`
- URL: `http://localhost:6006`
- Why: Storybook has predictable, stable HTML/CSS for testing

### BDD Structure to Create

Based on `.docs/testing/BDD_WIRING.md` and existing BDD harnesses in the monorepo:

```
frontend/.dx-tool/
├── Cargo.toml                    # Add BDD dependencies
├── bdd/
│   ├── Cargo.toml               # BDD harness crate
│   ├── src/
│   │   ├── main.rs              # Cucumber runner entrypoint
│   │   └── steps/
│   │       ├── mod.rs           # Re-export step modules
│   │       ├── world.rs         # BddWorld struct
│   │       ├── css_steps.rs     # CSS command steps
│   │       └── html_steps.rs    # HTML command steps
│   └── tests/
│       └── features/
│           ├── css/
│           │   ├── class_exists.feature
│           │   ├── selector_styles.feature
│           │   └── list_classes.feature
│           └── html/
│               ├── query_selector.feature
│               ├── attributes.feature
│               └── dom_tree.feature
```

---

## Implementation Guide

### Step 1: Create BDD Harness Crate

**File:** `frontend/.dx-tool/bdd/Cargo.toml`

```toml
[package]
name = "dx-bdd"
version = "0.1.0"
edition = "2021"
publish = false
license = "GPL-3.0-or-later"

[features]
default = ["bdd-cucumber"]
bdd-cucumber = []

[dependencies]
dx = { path = ".." }
cucumber = { version = "0.20", features = ["macros"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
anyhow = "1.0"
async-trait = "0.1"

[[bin]]
name = "bdd-runner"
path = "src/main.rs"
```

### Step 2: Create Main Entrypoint

**File:** `frontend/.dx-tool/bdd/src/main.rs`

```rust
// Created by: TEAM-DX-003
// BDD runner for dx CLI tool

mod steps;

use cucumber::World as _;
use steps::world::DxWorld;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features_env = std::env::var("DX_BDD_FEATURE_PATH").ok();
    let features = if let Some(p) = features_env {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() {
            pb
        } else {
            root.join(pb)
        }
    } else {
        root.join("tests/features")
    };

    DxWorld::cucumber()
        .fail_on_skipped()
        .run_and_exit(features)
        .await;
}
```

### Step 3: Create World

**File:** `frontend/.dx-tool/bdd/src/steps/world.rs`

```rust
// Created by: TEAM-DX-003
// BDD World for dx CLI integration tests

use cucumber::World;
use std::collections::HashMap;

/// Storybook server URL (port 6006)
pub const STORYBOOK_URL: &str = "http://localhost:6006";

#[derive(Debug, Default, World)]
pub struct DxWorld {
    /// Last command result
    pub last_result: Option<Result<String, String>>,
    
    /// Last CSS class check result
    pub class_exists: Option<bool>,
    
    /// Last extracted styles
    pub styles: HashMap<String, String>,
    
    /// Last extracted classes
    pub classes: Vec<String>,
    
    /// Last HTML element info
    pub element_tag: Option<String>,
    pub element_count: Option<usize>,
    
    /// Last attributes
    pub attributes: HashMap<String, String>,
    
    /// Last DOM tree
    pub dom_tree: Option<String>,
    
    /// Error message if command failed
    pub error_message: Option<String>,
}

impl DxWorld {
    /// Store success result
    pub fn store_success(&mut self, output: String) {
        self.last_result = Some(Ok(output));
        self.error_message = None;
    }
    
    /// Store error result
    pub fn store_error(&mut self, error: String) {
        self.last_result = Some(Err(error.clone()));
        self.error_message = Some(error);
    }
    
    /// Clear state for new scenario
    pub fn clear(&mut self) {
        self.last_result = None;
        self.class_exists = None;
        self.styles.clear();
        self.classes.clear();
        self.element_tag = None;
        self.element_count = None;
        self.attributes.clear();
        self.dom_tree = None;
        self.error_message = None;
    }
}
```

### Step 4: Create Step Modules

**File:** `frontend/.dx-tool/bdd/src/steps/mod.rs`

```rust
// Created by: TEAM-DX-003

pub mod world;
pub mod css_steps;
pub mod html_steps;
```

**File:** `frontend/.dx-tool/bdd/src/steps/css_steps.rs`

```rust
// Created by: TEAM-DX-003
// CSS command BDD steps

use cucumber::{given, when, then};
use crate::steps::world::{DxWorld, STORYBOOK_URL};
use dx::commands::CssCommand;

#[given(regex = r"^Storybook is running on port 6006$")]
pub async fn storybook_running(_world: &mut DxWorld) {
    // Assumption: Storybook must be running
    // Could add a health check here
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
```

**File:** `frontend/.dx-tool/bdd/src/steps/html_steps.rs`

```rust
// Created by: TEAM-DX-003
// HTML command BDD steps

use cucumber::{given, when, then};
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
```

### Step 5: Implement Story File Locator (PRIORITY 0)

**File:** `frontend/.dx-tool/src/commands/story.rs`

```rust
// Created by: TEAM-DX-003
// Story file locator command

use crate::error::{DxError, Result};
use colored::*;
use std::path::{Path, PathBuf};

/// Story command handler
pub struct StoryCommand {
    storybook_base: PathBuf,
}

impl StoryCommand {
    pub fn new() -> Self {
        // Default to frontend/libs/storybook from repo root
        let storybook_base = PathBuf::from("frontend/libs/storybook");
        Self { storybook_base }
    }
    
    /// Parse Storybook URL and locate the story file
    pub fn locate_story_file(&self, url: &str) -> Result<StoryFileInfo> {
        // Parse URL: http://localhost:6006/story/stories-atoms-button-button-story-vue
        let story_path = self.parse_story_path(url)?;
        
        // Convert to filesystem path
        let story_file = self.resolve_story_file(&story_path)?;
        let component_file = self.find_component_file(&story_file);
        
        Ok(StoryFileInfo {
            url: url.to_string(),
            story_path,
            story_file,
            component_file,
        })
    }
    
    fn parse_story_path(&self, url: &str) -> Result<String> {
        // Extract story path from URL
        // stories-atoms-button-button-story-vue -> stories/atoms/Button/Button.story.vue
        let url = url.split('?').next().unwrap_or(url); // Remove query params
        
        if let Some(story_part) = url.strip_prefix("http://localhost:6006/story/") {
            // Convert: stories-atoms-button-button-story-vue
            // To: stories/atoms/Button/Button.story.vue
            let parts: Vec<&str> = story_part.split('-').collect();
            
            if parts.len() < 4 {
                return Err(DxError::Parse("Invalid story URL format".to_string()));
            }
            
            // Build path: stories/atoms/Button/Button.story.vue
            let mut path_parts = vec![];
            let mut i = 0;
            
            while i < parts.len() {
                if i == parts.len() - 2 && parts[i] == "story" && parts[i + 1] == "vue" {
                    break; // Found .story.vue suffix
                }
                path_parts.push(parts[i]);
                i += 1;
            }
            
            // Last part is component name (capitalized)
            if let Some(component) = path_parts.last() {
                let component_cap = capitalize(component);
                path_parts.pop();
                
                let dir_path = path_parts.join("/");
                Ok(format!("{}/{}/{}.story.vue", dir_path, component_cap, component_cap))
            } else {
                Err(DxError::Parse("Could not extract component name".to_string()))
            }
        } else {
            Err(DxError::Parse("URL must start with http://localhost:6006/story/".to_string()))
        }
    }
    
    fn resolve_story_file(&self, story_path: &str) -> Result<PathBuf> {
        let full_path = self.storybook_base.join(story_path);
        
        if full_path.exists() {
            Ok(full_path)
        } else {
            Err(DxError::Parse(format!("Story file not found: {}", full_path.display())))
        }
    }
    
    fn find_component_file(&self, story_file: &Path) -> Option<PathBuf> {
        // Button.story.vue -> Button.vue
        let parent = story_file.parent()?;
        let stem = story_file.file_stem()?.to_str()?;
        let component_name = stem.strip_suffix(".story")?;
        
        let component_file = parent.join(format!("{}.vue", component_name));
        if component_file.exists() {
            Some(component_file)
        } else {
            None
        }
    }
    
    /// Print story file info
    pub fn print_story_info(&self, info: &StoryFileInfo) {
        println!("{} Story file located", "✓".green());
        println!("  URL: {}", info.url.cyan());
        println!("  File: {}", info.story_file.display().to_string().yellow());
        
        if let Some(component) = &info.component_file {
            println!("  Component: {}", component.display().to_string().yellow());
        }
        
        println!();
        println!("  To add more stories, edit:");
        println!("    - Story file: {}", info.story_path.cyan());
        if let Some(component) = &info.component_file {
            if let Some(name) = component.file_name() {
                println!("    - Component: {} (if needed)", name.to_string_lossy().cyan());
            }
        }
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[derive(Debug)]
pub struct StoryFileInfo {
    pub url: String,
    pub story_path: String,
    pub story_file: PathBuf,
    pub component_file: Option<PathBuf>,
}
```

**Update `src/commands/mod.rs`:**

```rust
pub mod css;
pub mod html;
pub mod story; // TEAM-DX-003: Added story file locator

pub use css::CssCommand;
pub use html::HtmlCommand;
pub use story::StoryCommand;
```

**Update `src/main.rs` to add story-file command:**

```rust
#[derive(Subcommand)]
enum Commands {
    /// CSS verification commands
    Css { ... },
    
    /// HTML structure queries
    Html { ... },
    
    /// Story file locator
    #[command(name = "story-file")]
    StoryFile {
        /// Storybook URL to locate
        url: String,
    },
}

// In main():
Commands::StoryFile { url } => {
    let cmd = StoryCommand::new();
    match cmd.locate_story_file(&url) {
        Ok(info) => {
            if use_json {
                println!("{{\"story_file\": \"{}\", \"component_file\": \"{:?}\"}}",
                    info.story_file.display(), info.component_file);
            } else {
                cmd.print_story_info(&info);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            ExitCode::from(1)
        }
    }
}
```

### Step 6: Create Feature Files

**File:** `frontend/.dx-tool/bdd/tests/features/story/story_locator.feature`

```gherkin
# Created by: TEAM-DX-003
# PRIORITY 0: Story file locator

Feature: Story File Locator
  As a frontend engineer without browser access
  I want to find which file defines a Storybook story
  So I can quickly make changes

  Background:
    Given Storybook is running on port 6006

  Scenario: Locate Button story file
    When I run story-file with URL "http://localhost:6006/story/stories-atoms-button-button-story-vue"
    Then I should see story file path "stories/atoms/Button/Button.story.vue"
    And I should see component file path "stories/atoms/Button/Button.vue"
    And the files should exist on disk

  Scenario: Locate story with variant ID
    When I run story-file with URL "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-0"
    Then I should see story file path "stories/atoms/Button/Button.story.vue"
    And the variant ID should be ignored

  Scenario: Invalid story URL
    When I run story-file with URL "http://localhost:6006/invalid"
    Then I should see an error "Could not parse story path from URL"

  Scenario: Story file not found
    When I run story-file with URL "http://localhost:6006/story/nonexistent-story"
    Then I should see an error "Story file not found on disk"
```

**File:** `frontend/.dx-tool/bdd/src/steps/story_steps.rs`

```rust
// Created by: TEAM-DX-003
// Story file locator BDD steps

use cucumber::{given, when, then};
use crate::steps::world::DxWorld;
use dx::commands::StoryCommand;
use std::path::PathBuf;

#[when(regex = r"^I run story-file with URL \"(.+)\"$")]
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

#[then(regex = r"^I should see story file path \"(.+)\"$")]
pub async fn should_see_story_path(world: &mut DxWorld, expected: String) {
    let story_path = world.story_path.as_ref().expect("No story path captured");
    assert_eq!(story_path, &expected,
        "Expected story path '{}', got '{}'", expected, story_path);
}

#[then(regex = r"^I should see component file path \"(.+)\"$")]
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

#[then(regex = r"^I should see an error \"(.+)\"$")]
pub async fn should_see_error(world: &mut DxWorld, expected_msg: String) {
    let error = world.error_message.as_ref().expect("No error captured");
    assert!(error.contains(&expected_msg),
        "Expected error to contain '{}', got '{}'", expected_msg, error);
}
```

**Update `bdd/src/steps/world.rs` to add story fields:**

```rust
#[derive(Debug, Default, World)]
pub struct DxWorld {
    // ... existing fields ...
    
    /// Story file locator fields
    pub story_file: Option<PathBuf>,
    pub component_file: Option<PathBuf>,
    pub story_path: Option<String>,
}
```

**File:** `frontend/.dx-tool/bdd/tests/features/css/class_exists.feature`

```gherkin
# Created by: TEAM-DX-003
# Test CSS class existence checking

Feature: CSS Class Existence
  As a frontend developer
  I want to check if CSS classes exist in Storybook
  So that I can verify Tailwind is generating my classes

  Background:
    Given Storybook is running on port 6006

  Scenario: Check for existing Tailwind class
    When I check if class 'flex' exists
    Then the class should exist

  Scenario: Check for non-existent class
    When I check if class 'nonexistent-class-xyz' exists
    Then the class should not exist

  Scenario: Check for Storybook-specific class
    When I check if class 'text-foreground' exists
    Then the class should exist
```

**File:** `frontend/.dx-tool/bdd/tests/features/css/selector_styles.feature`

```gherkin
# Created by: TEAM-DX-003
# Test CSS selector style extraction

Feature: CSS Selector Styles
  As a frontend developer
  I want to extract computed styles for selectors
  So that I can verify styling is applied correctly

  Background:
    Given Storybook is running on port 6006

  Scenario: Get styles for button
    When I get styles for selector 'button'
    Then I should see style 'cursor' with value 'pointer'

  Scenario: Get styles for specific component
    When I get styles for selector '.theme-toggle'
    Then I should see style 'position' with value 'relative'
```

**File:** `frontend/.dx-tool/bdd/tests/features/html/query_selector.feature`

```gherkin
# Created by: TEAM-DX-003
# Test HTML selector queries

Feature: HTML Selector Queries
  As a frontend developer
  I want to query DOM structure
  So that I can verify components are rendering

  Background:
    Given Storybook is running on port 6006

  Scenario: Query button elements
    When I query selector 'button'
    Then I should find at least 1 element
    And the element tag should be 'button'

  Scenario: Query navigation
    When I query selector 'nav'
    Then I should find 1 element
    And the element tag should be 'nav'
```

---

## Running the BDD Tests

### Prerequisites

1. **Start Storybook:**
   ```bash
   cd frontend/libs/storybook
   pnpm story:dev
   # Wait for: "Local: http://localhost:6006/"
   ```

2. **Run BDD tests:**
   ```bash
   cd frontend/.dx-tool/bdd
   cargo run --bin bdd-runner
   ```

3. **Run specific feature:**
   ```bash
   DX_BDD_FEATURE_PATH=tests/features/css/class_exists.feature \
     cargo run --bin bdd-runner
   ```

---

## Acceptance Criteria

### Must Have (P0) - PRIORITY ORDER

**1. Story File Locator (IMPLEMENT FIRST)**
- [ ] `dx story-file <URL>` command implemented in `src/commands/story.rs`
- [ ] URL parser converts Storybook URLs to filesystem paths
- [ ] File existence validation
- [ ] Component file detection (Button.story.vue → Button.vue)
- [ ] BDD feature: `story/story_locator.feature` with 4 scenarios
- [ ] BDD steps: `story_steps.rs` with all step definitions
- [ ] All story locator scenarios pass
- [ ] Text and JSON output formats

**2. BDD Harness Infrastructure**
- [ ] BDD harness crate created at `frontend/.dx-tool/bdd/`
- [ ] `DxWorld` struct with state management (including story fields)
- [ ] Cucumber runner with `fail_on_skipped()`
- [ ] `DX_BDD_FEATURE_PATH` environment variable support

**3. CSS & HTML Integration Tests (AFTER Story Locator)**
- [ ] CSS steps implemented (class_exists, selector_styles, list_classes)
- [ ] HTML steps implemented (query_selector, attributes, dom_tree)
- [ ] At least 3 additional feature files
- [ ] All scenarios pass against running Storybook

**4. Documentation**
- [ ] README with setup and run instructions
- [ ] TEAM-DX-003 signatures on all new files
- [ ] Update main handoff with completion status

### Should Have (P1)

- [ ] Error handling scenarios (server not running, invalid selectors)
- [ ] JSON output verification scenarios
- [ ] Parallel test execution support
- [ ] CI integration guide

### Nice to Have (P2)

- [ ] Auto-start Storybook if not running
- [ ] Screenshot capture on failure
- [ ] Performance benchmarks (sub-2s per scenario)

---

## Reference Examples

**Study these existing BDD harnesses:**
- `test-harness/bdd/` - Main BDD harness (most comprehensive)
- `bin/shared-crates/audit-logging/bdd/` - Simple BDD example
- `bin/shared-crates/input-validation/bdd/` - Another simple example

**Key patterns to follow:**
1. World struct with `#[derive(Debug, Default, World)]`
2. Step modules with `#[given]`, `#[when]`, `#[then]` macros
3. Async step functions: `pub async fn step_name(world: &mut DxWorld, ...)`
4. Feature files in `tests/features/` directory
5. Main entrypoint with `DxWorld::cucumber().fail_on_skipped().run_and_exit()`

---

## Integration with Existing Tests

**Convert these ignored tests to BDD scenarios:**

```rust
// src/commands/css.rs - Line 295-298
#[tokio::test]
#[ignore]
async fn test_get_selector_styles() { ... }
→ Feature: css/selector_styles.feature

// src/commands/html.rs - Line 263-267
#[tokio::test]
#[ignore]
async fn test_query_selector() { ... }
→ Feature: html/query_selector.feature
```

**Keep unit tests as-is** - BDD is for integration testing only.

---

## Troubleshooting Guide

### "Storybook not responding"
- Ensure Storybook is running: `pnpm story:dev` in `frontend/libs/storybook`
- Check port 6006 is not blocked: `curl http://localhost:6006`
- Wait 10-15 seconds after starting Storybook

### "Undefined steps"
- Ensure step module is exported in `steps/mod.rs`
- Check regex patterns match feature file text exactly
- Verify `cucumber = { version = "0.20", features = ["macros"] }` in Cargo.toml

### "World not found"
- Import World in step files: `use crate::steps::world::DxWorld;`
- Ensure World derives `cucumber::World`
- Check module path is correct

### "Tests fail in CI"
- Add Storybook startup to CI workflow
- Use health check before running tests
- Consider using a mock server for CI

---

## Documentation to Create

1. **`frontend/.dx-tool/bdd/README.md`** - Setup and usage guide
2. **`frontend/.dx-tool/bdd/SCENARIOS.md`** - List of all test scenarios
3. **Update `frontend/.dx-tool/TEAM_DX_002_HANDOFF.md`** - Mark integration tests as "moved to BDD"

---

## Estimated Effort

**REVISED with Story File Locator:**

- **Story file locator command:** 3-4 hours (PRIORITY 0)
- **Story file locator BDD tests:** 2-3 hours (PRIORITY 0)
- **BDD scaffold setup:** 2-3 hours
- **CSS steps + features:** 2-3 hours
- **HTML steps + features:** 2-3 hours
- **Documentation + testing:** 2 hours
- **Total:** ~15-18 hours

**Implementation Order:**
1. Story file locator (5-7 hours) - **DO THIS FIRST**
2. BDD infrastructure (2-3 hours)
3. CSS/HTML integration tests (4-6 hours)
4. Documentation (2 hours)

---

## Success Metrics

**Story File Locator (CRITICAL):**
- ✅ `dx story-file <URL>` command works
- ✅ Correctly maps Storybook URLs to filesystem paths
- ✅ Finds both story file and component file
- ✅ All 4 story locator scenarios pass
- ✅ Helps engineers find files to edit in <2 seconds

**BDD Infrastructure:**
- ✅ BDD harness compiles and runs
- ✅ All integration scenarios pass against Storybook
- ✅ Tests run in <30 seconds total
- ✅ Clear error messages on failure
- ✅ Documentation complete
- ✅ TEAM-DX-003 signatures everywhere

---

## Next Steps After BDD

**Phase 3 priorities (for TEAM-DX-004):**
- Accessibility audit commands
- CSS variable resolution
- Responsive breakpoint testing
- Performance optimization (parallel stylesheet fetching)

---

**TEAM-DX-002 OUT. BDD scaffold is your mission. Use Storybook port 6006. Follow the patterns in existing BDD harnesses. Make it green.**
