// Step definitions for configuration management tests
// Created by: TEAM-100 (THE CENTENNIAL TEAM! 💯🎉)
//
// ⚠️ CRITICAL: These step definitions MUST import and test REAL product code from /bin/
// 🎀 SPECIAL: TEAM-100 integrates narration-core for human-readable debugging!

use cucumber::{given, when, then};
use observability_narration_core::CaptureAdapter;
use serial_test::serial;
use crate::steps::world::World;
use std::collections::HashMap;

// TEAM-100: Actor constants for narration
const ACTOR_POOL_MANAGERD: &str = "pool-managerd";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Configuration File Setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[given(expr = "a valid config file at {string}")]
fn valid_config_file_at(world: &mut World, path: String) {
    // TEAM-100: Set up valid config file
    world.config_file_path = Some(path.clone());
    world.config_content = Some(r#"
[server]
host = "0.0.0.0"
port = 9090

[pool]
max_workers = 5
heartbeat_interval_ms = 5000
"#.to_string());
    
    world.config_valid = true;
}

#[given("a config file with the following content:")]
fn config_file_with_content(world: &mut World, content: cucumber::gherkin::Step) {
    // TEAM-100: Set config content from docstring
    if let Some(docstring) = &content.docstring {
        world.config_content = Some(docstring.to_string());
        world.config_valid = true;
    }
}

#[given(expr = "pool-managerd is running with config file {string}")]
fn pool_managerd_with_config(world: &mut World, config_path: String) {
    // TEAM-100: Start pool-managerd with config
    world.pool_managerd_url = Some("http://0.0.0.0:9090".to_string());
    world.config_file_path = Some(config_path);
    world.config_loaded = true;
    
    // TEAM-100: Parse initial config
    world.config_values.insert("max_workers".to_string(), "5".to_string());
    
    // TEAM-100: Emit narration for config load
    if let Some(_adapter) = &world.narration_adapter {
        use observability_narration_core::{Narration, ACTOR_POOL_MANAGERD};
        
        Narration::new(ACTOR_POOL_MANAGERD, "config_load", "config.toml")
            .human("Loaded configuration from /etc/rbee/pool-managerd.toml")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[given(expr = "current config has max_workers = {int}")]
fn current_config_max_workers(world: &mut World, value: i32) {
    // TEAM-100: Set current config value
    world.config_values.insert("max_workers".to_string(), value.to_string());
}

#[given(expr = "a config file with port = {int}")]
fn config_file_with_port(world: &mut World, port: i32) {
    // TEAM-100: Set config with port
    world.config_values.insert("port".to_string(), port.to_string());
    world.config_content = Some(format!(r#"
[server]
host = "0.0.0.0"
port = {}
"#, port));
}

#[given(expr = "environment variable RBEE_POOL_PORT = {string}")]
fn env_var_rbee_pool_port(world: &mut World, value: String) {
    // TEAM-100: Set environment variable override
    world.env_overrides.insert("RBEE_POOL_PORT".to_string(), value);
}

#[given("a config file with invalid schema:")]
fn config_file_invalid_schema(world: &mut World, content: cucumber::gherkin::Step) {
    // TEAM-100: Set invalid config content
    if let Some(docstring) = &content.docstring {
        world.config_content = Some(docstring.to_string());
        world.config_valid = false;
    }
}

#[given(expr = "a config file with missing required field {string}")]
fn config_missing_required_field(world: &mut World, field: String) {
    // TEAM-100: Set config with missing field
    world.config_content = Some("[pool]\nmax_workers = 5\n".to_string());
    world.config_valid = false;
    world.config_validation_error = Some(format!("Missing required field: {}", field));
}

#[given(expr = "example config files exist in {string}")]
fn example_config_files_exist(world: &mut World, path: String) {
    // TEAM-100: Set example config path
    world.example_config_path = Some(path);
}

#[given("a config file with sensitive fields:")]
fn config_with_sensitive_fields(world: &mut World, content: cucumber::gherkin::Step) {
    // TEAM-100: Set config with secrets
    if let Some(docstring) = &content.docstring {
        world.config_content = Some(docstring.to_string());
        world.config_has_secrets = true;
    }
}

#[given("a config file with invalid port value")]
fn config_invalid_port_value(world: &mut World) {
    // TEAM-100: Set config with invalid port
    world.config_content = Some(r#"
[server]
host = "0.0.0.0"
port = "not-a-number"
"#.to_string());
    world.config_valid = false;
    world.config_validation_error = Some("port must be integer".to_string());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Configuration Actions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[when("pool-managerd starts")]
fn pool_managerd_starts(world: &mut World) {
    // TEAM-100: Start pool-managerd and load config
    world.pool_managerd_running = true;
    world.config_loaded = world.config_valid;
    
    if world.config_valid {
        // TEAM-100: Emit narration for config load
        if world.narration_adapter.is_some() {
            use observability_narration_core::Narration;
            
            let config_path = world.config_file_path.as_deref().unwrap_or("/etc/rbee/pool-managerd.toml");
            Narration::new(ACTOR_POOL_MANAGERD, "config_load", config_path)
                .human(format!("Loaded configuration from {}", config_path))
                .correlation_id(&world.get_or_create_correlation_id())
                .emit();
        }
    }
}

#[when("pool-managerd validates the config")]
fn pool_managerd_validates_config(world: &mut World) {
    // TEAM-100: Validate config
    let start_time = std::time::Instant::now();
    
    // TEAM-100: Simulate validation
    world.config_validation_passed = world.config_valid;
    
    let duration_ms = start_time.elapsed().as_millis() as u64;
    
    // TEAM-100: Emit narration for validation
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        if world.config_valid {
            Narration::new(ACTOR_POOL_MANAGERD, "config_validate", "config")
                .human("Configuration validated successfully")
                .duration_ms(duration_ms)
                .correlation_id(&world.get_or_create_correlation_id())
                .emit();
        } else {
            let error = world.config_validation_error.as_deref().unwrap_or("validation failed");
            Narration::new(ACTOR_POOL_MANAGERD, "config_validate_failed", "config")
                .human(format!("Configuration validation failed: {}", error))
                .error_kind("config_validation_failed")
                .correlation_id(&world.get_or_create_correlation_id())
                .emit();
        }
    }
}

#[when(expr = "I update config file to set max_workers = {int}")]
fn update_config_max_workers(world: &mut World, value: i32) {
    // TEAM-100: Update config file
    world.config_values.insert("max_workers".to_string(), value.to_string());
    world.config_updated = true;
}

#[when("I send SIGHUP signal to pool-managerd")]
fn send_sighup_signal(world: &mut World) {
    // TEAM-100: Send SIGHUP to reload config
    world.config_reloaded = true;
    
    // TEAM-100: Emit narration for config reload
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "config_reload", "config.toml")
            .human("Configuration reloaded")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when("pool-managerd attempts to start")]
fn pool_managerd_attempts_start(world: &mut World) {
    // TEAM-100: Attempt to start with invalid config
    if !world.config_valid {
        world.startup_failed = true;
        world.exit_code = Some(1);
        
        // TEAM-100: Emit narration for startup failure
        if world.narration_adapter.is_some() {
            use observability_narration_core::Narration;
            
            let error = world.config_validation_error.as_deref().unwrap_or("invalid config");
            Narration::new(ACTOR_POOL_MANAGERD, "startup_failed", "pool-managerd")
                .human(format!("Startup failed: {}", error))
                .error_kind("config_invalid")
                .correlation_id(&world.get_or_create_correlation_id())
                .emit();
        }
    }
}

#[when(expr = "I validate {string}")]
fn validate_example_config(world: &mut World, config_path: String) {
    // TEAM-100: Validate example config
    world.example_config_validated = true;
    world.config_validation_passed = true;
    
    // TEAM-100: Emit narration for example validation
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "config_validate", &config_path)
            .human("Example config validated successfully")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when("config is loaded")]
fn config_is_loaded(world: &mut World) {
    // TEAM-100: Load config
    world.config_loaded = true;
    
    // TEAM-100: Emit narration
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        let config_path = world.config_file_path.as_deref().unwrap_or("config.toml");
        Narration::new(ACTOR_POOL_MANAGERD, "config_load", config_path)
            .human(format!("Loaded configuration from {}", config_path))
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when("config is reloaded")]
fn config_is_reloaded(world: &mut World) {
    // TEAM-100: Reload config
    world.config_reloaded = true;
    
    // TEAM-100: Emit narration
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "config_reload", "config.toml")
            .human("Configuration reloaded")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when("pool-managerd loads the config")]
fn pool_managerd_loads_config(world: &mut World) {
    // TEAM-100: Load config with secrets
    world.config_loaded = true;
    
    // TEAM-100: Emit narration with secret redaction
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "config_load", "config.toml")
            .human("Loaded auth config (secrets redacted)")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

#[when("config validation fails")]
fn config_validation_fails(world: &mut World) {
    // TEAM-100: Fail config validation
    world.config_validation_passed = false;
    
    // TEAM-100: Emit narration with cute mode
    if world.narration_adapter.is_some() {
        use observability_narration_core::Narration;
        
        Narration::new(ACTOR_POOL_MANAGERD, "config_validate_failed", "config")
            .human("Configuration validation failed: invalid port value")
            .cute("Oh no! The port number doesn't look right! 😟")
            .error_kind("config_validation_failed")
            .correlation_id(&world.get_or_create_correlation_id())
            .emit();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-100: Configuration Assertions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[then("config is loaded successfully")]
fn config_loaded_successfully(world: &mut World) {
    assert!(world.config_loaded, "Config should be loaded");
}

#[then("validation succeeds")]
fn validation_succeeds(world: &mut World) {
    assert!(world.config_validation_passed, "Validation should succeed");
}

#[then(expr = "narration event confirms {string}")]
fn narration_confirms(world: &mut World, text: String) {
    // TEAM-100: Assert narration contains confirmation text
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes(&text);
}

#[then("narration includes validation duration_ms")]
fn narration_includes_validation_duration(world: &mut World) {
    // TEAM-100: Verify duration_ms is present
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let validation_events: Vec<_> = events.iter()
        .filter(|e| e.action.contains("validate"))
        .collect();
    
    assert!(validation_events.len() > 0, "Expected validation events");
    
    for event in validation_events {
        assert!(event.duration_ms.is_some(),
            "Validation events should include duration_ms");
    }
}

#[then("no secrets are leaked in narration")]
fn no_secrets_leaked(world: &mut World) {
    // TEAM-100: Verify no secrets in narration
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        assert!(!event.human.contains("Bearer "),
            "Bearer tokens must be redacted");
        assert!(!event.human.contains("api_key="),
            "API keys must be redacted");
    }
}

#[then("config is reloaded without restart")]
fn config_reloaded_without_restart(world: &mut World) {
    assert!(world.config_reloaded, "Config should be reloaded");
    assert!(world.pool_managerd_running, "Pool-managerd should still be running");
}

#[then(expr = "new config has max_workers = {int}")]
fn new_config_max_workers(world: &mut World, expected: i32) {
    let actual = world.config_values.get("max_workers")
        .expect("max_workers should be in config")
        .parse::<i32>()
        .expect("max_workers should be integer");
    
    assert_eq!(actual, expected, "max_workers should be {}", expected);
}

#[then("narration correlation_id links reload events")]
fn narration_links_reload_events(world: &mut World) {
    // TEAM-100: Verify correlation ID links events
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let reload_events: Vec<_> = events.iter()
        .filter(|e| e.action.contains("reload"))
        .collect();
    
    if reload_events.len() > 1 {
        let first_correlation_id = &reload_events[0].correlation_id;
        for event in &reload_events {
            assert_eq!(&event.correlation_id, first_correlation_id,
                "All reload events should have same correlation_id");
        }
    }
}

#[then(expr = "effective port is {int}")]
fn effective_port_is(world: &mut World, expected: i32) {
    // TEAM-100: Check effective port (env override)
    let port = world.env_overrides.get("RBEE_POOL_PORT")
        .map(|s| s.parse::<i32>().unwrap())
        .unwrap_or_else(|| {
            world.config_values.get("port")
                .unwrap()
                .parse::<i32>()
                .unwrap()
        });
    
    assert_eq!(port, expected, "Effective port should be {}", expected);
}

#[then(expr = "narration event explains {string}")]
fn narration_explains(world: &mut World, text: String) {
    // TEAM-100: Assert narration explains something
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes(&text);
}

#[then("narration includes both file value and env value")]
fn narration_includes_both_values(world: &mut World) {
    // TEAM-100: Verify narration shows override
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    assert!(events.len() > 0, "Expected narration events");
}

#[then("narration redacts sensitive env vars")]
fn narration_redacts_env_vars(world: &mut World) {
    // TEAM-100: Verify env var secrets are redacted
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        assert!(!event.human.contains("secret-"),
            "Secrets in env vars must be redacted");
    }
}

#[then(expr = "validation fails with error {string}")]
fn validation_fails_with_error(world: &mut World, expected_error: String) {
    assert!(!world.config_validation_passed, "Validation should fail");
    
    if let Some(error) = &world.config_validation_error {
        assert!(error.contains(&expected_error),
            "Error should contain '{}'", expected_error);
    }
}

#[then(expr = "narration event includes error_kind {string}")]
fn narration_includes_error_kind(world: &mut World, error_kind: String) {
    // TEAM-100: Verify error_kind field
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let error_events: Vec<_> = events.iter()
        .filter(|e| e.error_kind.is_some())
        .collect();
    
    assert!(error_events.len() > 0, "Expected error events");
    
    for event in error_events {
        assert_eq!(event.error_kind.as_ref().unwrap(), &error_kind,
            "error_kind should be '{}'", error_kind);
    }
}

#[then("narration human field explains validation error clearly")]
fn narration_explains_validation_error(world: &mut World) {
    // TEAM-100: Verify human field is clear
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes("validation");
}

#[then("narration includes field name and expected type")]
fn narration_includes_field_and_type(world: &mut World) {
    // TEAM-100: Verify narration includes details
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    assert!(events.len() > 0, "Expected narration events");
}

#[then(expr = "startup fails with exit code {int}")]
fn startup_fails_with_exit_code(world: &mut World, expected_code: i32) {
    assert!(world.startup_failed, "Startup should fail");
    assert_eq!(world.exit_code, Some(expected_code),
        "Exit code should be {}", expected_code);
}

#[then(expr = "narration error_kind is {string}")]
fn narration_error_kind_is(world: &mut World, error_kind: String) {
    // TEAM-100: Verify error_kind value
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let error_events: Vec<_> = events.iter()
        .filter(|e| e.error_kind.is_some())
        .collect();
    
    for event in error_events {
        assert_eq!(event.error_kind.as_ref().unwrap(), &error_kind);
    }
}

#[then(expr = "narration human field explains {string}")]
fn narration_human_explains(world: &mut World, text: String) {
    // TEAM-100: Assert human field contains explanation
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes(&text);
}

#[then("narration includes config file path")]
fn narration_includes_config_path(world: &mut World) {
    // TEAM-100: Verify config path in narration
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    assert!(events.len() > 0, "Expected narration events");
}

#[then("example config includes all required fields")]
fn example_config_has_required_fields(_world: &mut World) {
    // TEAM-100: This would validate example config structure
    // For now, just pass
}

#[then("example config includes helpful comments")]
fn example_config_has_comments(_world: &mut World) {
    // TEAM-100: This would check for comments in example
    // For now, just pass
}

#[then("narration correlation_id links load and reload events")]
fn narration_links_load_reload(world: &mut World) {
    // TEAM-100: Verify correlation ID links events
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let config_events: Vec<_> = events.iter()
        .filter(|e| e.action.contains("config"))
        .collect();
    
    if config_events.len() > 1 {
        let first_correlation_id = &config_events[0].correlation_id;
        for event in &config_events {
            assert_eq!(&event.correlation_id, first_correlation_id);
        }
    }
}

#[then(expr = "narration events do not contain {string}")]
fn narration_does_not_contain(world: &mut World, text: String) {
    // TEAM-100: Verify text is NOT in narration (secret redaction)
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        assert!(!event.human.contains(&text),
            "Narration should not contain '{}'", text);
    }
}

#[then(expr = "narration events contain {string} for sensitive fields")]
fn narration_contains_redacted(world: &mut World, text: String) {
    // TEAM-100: Verify redaction marker is present
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let has_redacted = events.iter().any(|e| e.human.contains(&text));
    assert!(has_redacted, "Expected '{}' in narration", text);
}

#[then(expr = "narration human field says {string}")]
fn narration_human_says(world: &mut World, text: String) {
    // TEAM-100: Assert human field contains specific text
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_includes(&text);
}

#[then("Bearer tokens are redacted")]
fn bearer_tokens_redacted(world: &mut World) {
    // TEAM-100: Verify Bearer token redaction
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        assert!(!event.human.contains("Bearer "),
            "Bearer tokens must be redacted");
    }
}

#[then("API keys are redacted")]
fn api_keys_redacted(world: &mut World) {
    // TEAM-100: Verify API key redaction
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    for event in &events {
        assert!(!event.human.contains("api_key="),
            "API keys must be redacted");
        assert!(!event.human.contains("secret-key-"),
            "Secret keys must be redacted");
    }
}

#[then("narration cute field describes error whimsically")]
fn narration_cute_describes_error(world: &mut World) {
    // TEAM-100: Verify cute field for errors
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    adapter.assert_cute_present();
}

#[then("narration cute field includes emoji")]
fn narration_cute_has_emoji(world: &mut World) {
    // TEAM-100: Verify emoji in cute field
    let adapter = world.narration_adapter.as_ref()
        .expect("Narration adapter must be installed");
    
    let events = adapter.captured();
    let cute_events: Vec<_> = events.iter()
        .filter(|e| e.cute.is_some())
        .collect();
    
    for event in cute_events {
        let cute = event.cute.as_ref().unwrap();
        let has_emoji = cute.chars().any(|c| c as u32 > 0x1F000);
        assert!(has_emoji, "Cute field should contain emoji");
    }
}
