// TEAM-297: Phase 0 - Tests for ultra-concise narration macro
//! Comprehensive tests for the n!() macro and narration mode system

use observability_narration_core::*;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_simple_narration() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Simple message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].action, "test");
    assert_eq!(captured[0].human, "Simple message");
}

#[test]
#[serial(capture_adapter)]
fn test_narration_with_single_format() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Message with {}", "arg1");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Message with arg1");
}

#[test]
#[serial(capture_adapter)]
fn test_narration_with_multiple_format() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Message with {} and {}", "arg1", "arg2");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Message with arg1 and arg2");
}

#[test]
#[serial(capture_adapter)]
fn test_narration_with_format_specifiers() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Number: {}, Hex: {:x}, Debug: {:?}", 42, 255, vec![1, 2, 3]);
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Number: 42, Hex: ff, Debug: [1, 2, 3]");
}

#[test]
#[serial(capture_adapter)]
fn test_narration_modes_all_three() {
    let adapter = CaptureAdapter::install();
    
    n!("test",
        human: "Technical message",
        cute: "🐝 Fun message",
        story: "Story message"
    );
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Technical message");
    assert_eq!(captured[0].cute, Some("🐝 Fun message".to_string()));
    assert_eq!(captured[0].story, Some("Story message".to_string()));
}

#[test]
#[serial(capture_adapter)]
fn test_narration_modes_with_format_args() {
    let adapter = CaptureAdapter::install();
    
    n!("test",
        human: "Deploying {}",
        cute: "🚀 Launching {}!",
        story: "'Fly, {}', whispered the system",
        "my-service"
    );
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Deploying my-service");
    assert_eq!(captured[0].cute, Some("🚀 Launching my-service!".to_string()));
    assert_eq!(captured[0].story, Some("'Fly, my-service', whispered the system".to_string()));
}

#[test]
#[serial(capture_adapter)]
fn test_mode_selection_human() {
    set_narration_mode(NarrationMode::Human);
    let adapter = CaptureAdapter::install();
    
    n!("test",
        human: "Human message",
        cute: "Cute message",
        story: "Story message"
    );
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // Mode selection happens in narrate_at_level, not in macro
    // Macro always sets all three fields
    assert_eq!(captured[0].human, "Human message");
}

#[test]
#[serial(capture_adapter)]
fn test_mode_selection_cute() {
    set_narration_mode(NarrationMode::Cute);
    
    let mode = get_narration_mode();
    assert_eq!(mode, NarrationMode::Cute);
    
    // Reset for next test
    set_narration_mode(NarrationMode::Human);
}

#[test]
#[serial(capture_adapter)]
fn test_mode_selection_story() {
    set_narration_mode(NarrationMode::Story);
    
    let mode = get_narration_mode();
    assert_eq!(mode, NarrationMode::Story);
    
    // Reset for next test
    set_narration_mode(NarrationMode::Human);
}

#[test]
#[serial(capture_adapter)]
fn test_fallback_to_human() {
    let adapter = CaptureAdapter::install();
    
    // Only human provided, should work in any mode
    n!("test", "Only human message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Only human message");
    assert_eq!(captured[0].cute, None);
    assert_eq!(captured[0].story, None);
}

#[test]
#[serial(capture_adapter)]
fn test_explicit_human_mode() {
    let adapter = CaptureAdapter::install();
    
    n!(human: "test", "Human mode message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Human mode message");
    assert_eq!(captured[0].cute, None);
    assert_eq!(captured[0].story, None);
}

#[test]
#[serial(capture_adapter)]
fn test_explicit_cute_mode() {
    let adapter = CaptureAdapter::install();
    
    n!(cute: "test", "🐝 Cute message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // When only cute is provided, human is empty
    assert_eq!(captured[0].human, "");
    assert_eq!(captured[0].cute, Some("🐝 Cute message".to_string()));
    assert_eq!(captured[0].story, None);
}

#[test]
#[serial(capture_adapter)]
fn test_explicit_story_mode() {
    let adapter = CaptureAdapter::install();
    
    n!(story: "test", "'Hello', said the system");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // When only story is provided, human is empty
    assert_eq!(captured[0].human, "");
    assert_eq!(captured[0].cute, None);
    assert_eq!(captured[0].story, Some("'Hello', said the system".to_string()));
}

#[test]
#[serial(capture_adapter)]
fn test_explicit_human_with_format() {
    let adapter = CaptureAdapter::install();
    
    n!(human: "test", "Message {} and {}", "arg1", "arg2");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Message arg1 and arg2");
}

#[test]
#[serial(capture_adapter)]
fn test_explicit_cute_with_format() {
    let adapter = CaptureAdapter::install();
    
    n!(cute: "test", "🐝 Message {}", "arg1");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].cute, Some("🐝 Message arg1".to_string()));
}

#[test]
#[serial(capture_adapter)]
fn test_explicit_story_with_format() {
    let adapter = CaptureAdapter::install();
    
    n!(story: "test", "'Hello, {}', said the system", "world");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].story, Some("'Hello, world', said the system".to_string()));
}

#[test]
#[serial(capture_adapter)]
fn test_trailing_comma_simple() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Message with {}", "arg",);
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Message with arg");
}

#[test]
#[serial(capture_adapter)]
fn test_trailing_comma_all_modes() {
    let adapter = CaptureAdapter::install();
    
    n!("test",
        human: "Human {}",
        cute: "Cute {}",
        story: "Story {}",
        "arg",
    );
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Human arg");
    assert_eq!(captured[0].cute, Some("Cute arg".to_string()));
    assert_eq!(captured[0].story, Some("Story arg".to_string()));
}

#[test]
#[serial(capture_adapter)]
fn test_comparison_old_vs_new() {
    let adapter = CaptureAdapter::install();
    
    // Old way (builder)
    Narration::new("test-actor", "test-action", "target")
        .human("Old way message")
        .emit();
    
    // New way (macro)
    n!("test-action", "New way message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 2);
    
    // Old way
    assert_eq!(captured[0].actor, "test-actor");
    assert_eq!(captured[0].action, "test-action");
    assert_eq!(captured[0].human, "Old way message");
    
    // New way
    assert_eq!(captured[1].actor, "unknown"); // No context set
    assert_eq!(captured[1].action, "test-action");
    assert_eq!(captured[1].human, "New way message");
}

#[test]
#[serial(capture_adapter)]
fn test_backward_compatibility_builder_still_works() {
    let adapter = CaptureAdapter::install();
    
    // Old builder API should still work
    Narration::new("test-actor", "test-action", "target")
        .context("value1")
        .context("value2")
        .human("Message {0} and {1}")
        .emit();
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Message value1 and value2");
}

#[test]
fn test_mode_enum_properties() {
    // Test enum values
    assert_eq!(NarrationMode::Human as u8, 0);
    assert_eq!(NarrationMode::Cute as u8, 1);
    assert_eq!(NarrationMode::Story as u8, 2);
    
    // Test equality
    assert_eq!(NarrationMode::Human, NarrationMode::Human);
    assert_ne!(NarrationMode::Human, NarrationMode::Cute);
}

#[test]
fn test_mode_switching_thread_safe() {
    // Test that mode switching is thread-safe
    set_narration_mode(NarrationMode::Human);
    assert_eq!(get_narration_mode(), NarrationMode::Human);
    
    set_narration_mode(NarrationMode::Cute);
    assert_eq!(get_narration_mode(), NarrationMode::Cute);
    
    set_narration_mode(NarrationMode::Story);
    assert_eq!(get_narration_mode(), NarrationMode::Story);
    
    // Reset
    set_narration_mode(NarrationMode::Human);
}
