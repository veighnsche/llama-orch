//! Comprehensive integration tests for narration macros.
//!
//! Tests all macro behaviors including template interpolation, async support,
//! error handling, and actor inference.

use observability_narration_macros::{narrate, trace_fn};

// ============================================================================
// #[narrate(...)] Macro Tests
// ============================================================================

#[narrate(action = "basic", human = "Basic narration test")]
fn narrate_basic() -> String {
    "basic".to_string()
}

#[narrate(action = "interpolation", human = "Job {job_id} dispatched to {worker_id}")]
fn narrate_with_vars(job_id: &str, worker_id: &str) -> String {
    format!("{}:{}", job_id, worker_id)
}

#[narrate(action = "cute", human = "Spawning {name}", cute = "A new friend {name} is born! 🎉")]
fn narrate_with_cute(name: &str) -> String {
    name.to_string()
}

#[narrate(action = "story", human = "Worker {id} registered", story = "'Hello!' said worker {id}")]
fn narrate_with_story(id: &str) -> String {
    id.to_string()
}

#[narrate(
    action = "all",
    human = "Processing {item}",
    cute = "Working on {item}! ✨",
    story = "'I'm processing {item}' said the system"
)]
fn narrate_all_templates(item: &str) -> String {
    item.to_string()
}

#[narrate(action = "async_op", human = "Async operation with {param}")]
async fn narrate_async(param: &str) -> String {
    format!("async:{}", param)
}

#[narrate(action = "result", human = "Fallible operation with {input}")]
fn narrate_result(input: &str) -> Result<String, String> {
    if input.is_empty() {
        Err("empty".to_string())
    } else {
        Ok(format!("ok:{}", input))
    }
}

#[narrate(action = "generic", human = "Generic operation")]
fn narrate_generic<T: ToString>(value: T) -> String {
    value.to_string()
}

#[narrate(action = "multiple", human = "Multiple params: {a}, {b}, {c}")]
fn narrate_multiple(a: &str, b: i32, c: bool) -> String {
    format!("{}:{}:{}", a, b, c)
}

#[narrate(action = "no_vars", human = "Static message")]
fn narrate_no_vars() -> String {
    "static".to_string()
}

// ============================================================================
// #[trace_fn] Macro Tests
// ============================================================================

#[trace_fn]
fn trace_basic() -> String {
    "traced".to_string()
}

#[trace_fn]
fn trace_with_params(a: i32, b: &str) -> String {
    format!("{}:{}", a, b)
}

#[trace_fn]
async fn trace_async(param: &str) -> String {
    format!("async:{}", param)
}

#[trace_fn]
fn trace_result(input: &str) -> Result<String, String> {
    if input.is_empty() {
        Err("empty".to_string())
    } else {
        Ok(format!("ok:{}", input))
    }
}

#[trace_fn]
fn trace_generic<T: ToString>(value: T) -> String {
    value.to_string()
}

#[trace_fn]
fn trace_mut_param(value: &mut i32) {
    *value += 1;
}

#[trace_fn]
fn trace_lifetime<'a>(input: &'a str) -> &'a str {
    input
}

// ============================================================================
// Test Cases
// ============================================================================

#[test]
fn test_narrate_basic() {
    assert_eq!(narrate_basic(), "basic");
}

#[test]
fn test_narrate_with_vars() {
    assert_eq!(narrate_with_vars("job1", "worker1"), "job1:worker1");
}

#[test]
fn test_narrate_with_cute() {
    assert_eq!(narrate_with_cute("test"), "test");
}

#[test]
fn test_narrate_with_story() {
    assert_eq!(narrate_with_story("w1"), "w1");
}

#[test]
fn test_narrate_all_templates() {
    assert_eq!(narrate_all_templates("item1"), "item1");
}

#[test]
fn test_narrate_async() {
    let result = tokio_test::block_on(narrate_async("test"));
    assert_eq!(result, "async:test");
}

#[test]
fn test_narrate_result() {
    assert_eq!(narrate_result("test").unwrap(), "ok:test");
    assert!(narrate_result("").is_err());
}

#[test]
fn test_narrate_generic() {
    assert_eq!(narrate_generic(42), "42");
    assert_eq!(narrate_generic("test"), "test");
}

#[test]
fn test_narrate_multiple() {
    assert_eq!(narrate_multiple("a", 1, true), "a:1:true");
}

#[test]
fn test_narrate_no_vars() {
    assert_eq!(narrate_no_vars(), "static");
}

#[test]
fn test_trace_basic() {
    assert_eq!(trace_basic(), "traced");
}

#[test]
fn test_trace_with_params() {
    assert_eq!(trace_with_params(42, "test"), "42:test");
}

#[test]
fn test_trace_async() {
    let result = tokio_test::block_on(trace_async("test"));
    assert_eq!(result, "async:test");
}

#[test]
fn test_trace_result() {
    assert_eq!(trace_result("test").unwrap(), "ok:test");
    assert!(trace_result("").is_err());
}

#[test]
fn test_trace_generic() {
    assert_eq!(trace_generic(42), "42");
    assert_eq!(trace_generic("test"), "test");
}

#[test]
fn test_trace_mut_param() {
    let mut x = 10;
    trace_mut_param(&mut x);
    assert_eq!(x, 11);
}

#[test]
fn test_trace_lifetime() {
    assert_eq!(trace_lifetime("test"), "test");
}

// ============================================================================
// Template Variable Tests
// ============================================================================

#[narrate(action = "single", human = "Value: {val}")]
fn template_single_var(val: &str) -> String {
    val.to_string()
}

#[narrate(action = "multi", human = "{a} and {b} and {c}")]
fn template_multi_var(a: &str, b: &str, c: &str) -> String {
    format!("{}:{}:{}", a, b, c)
}

#[narrate(action = "repeat", human = "{x}, {x}, {x}")]
fn template_repeat_var(x: &str) -> String {
    x.to_string()
}

#[narrate(action = "underscore", human = "{var_name} and {another_var}")]
fn template_underscore(var_name: &str, another_var: &str) -> String {
    format!("{}:{}", var_name, another_var)
}

#[narrate(action = "emoji", human = "🎉 {event} 🎊")]
fn template_emoji(event: &str) -> String {
    event.to_string()
}

#[test]
fn test_template_single_var() {
    assert_eq!(template_single_var("test"), "test");
}

#[test]
fn test_template_multi_var() {
    assert_eq!(template_multi_var("a", "b", "c"), "a:b:c");
}

#[test]
fn test_template_repeat_var() {
    assert_eq!(template_repeat_var("x"), "x");
}

#[test]
fn test_template_underscore() {
    assert_eq!(template_underscore("a", "b"), "a:b");
}

#[test]
fn test_template_emoji() {
    assert_eq!(template_emoji("success"), "success");
}

// ============================================================================
// Visibility and Attribute Preservation Tests
// ============================================================================

mod visibility_tests {
    use super::*;

    #[narrate(action = "private", human = "Private function")]
    fn private_narrate() -> String {
        "private".to_string()
    }

    #[narrate(action = "public", human = "Public function")]
    pub fn public_narrate() -> String {
        "public".to_string()
    }

    #[trace_fn]
    fn private_trace() -> String {
        "private".to_string()
    }

    #[trace_fn]
    pub fn public_trace() -> String {
        "public".to_string()
    }

    #[test]
    fn test_private_functions() {
        assert_eq!(private_narrate(), "private");
        assert_eq!(private_trace(), "private");
    }

    #[test]
    fn test_public_functions() {
        assert_eq!(public_narrate(), "public");
        assert_eq!(public_trace(), "public");
    }
}

#[test]
fn test_public_visibility_from_outside() {
    assert_eq!(visibility_tests::public_narrate(), "public");
    assert_eq!(visibility_tests::public_trace(), "public");
}

// Test attribute preservation
#[narrate(action = "attrs", human = "With attributes")]
#[allow(dead_code)]
fn with_attributes() -> String {
    "attrs".to_string()
}

#[trace_fn]
#[allow(dead_code)]
fn trace_with_attributes() -> String {
    "attrs".to_string()
}

#[test]
fn test_attribute_preservation() {
    assert_eq!(with_attributes(), "attrs");
    assert_eq!(trace_with_attributes(), "attrs");
}

// ============================================================================
// Complex Type Tests
// ============================================================================

#[narrate(action = "option", human = "Optional operation")]
fn narrate_option(input: &str) -> Option<String> {
    if input.is_empty() {
        None
    } else {
        Some(input.to_string())
    }
}

#[narrate(action = "complex", human = "Complex return")]
fn narrate_complex(input: &str) -> Result<Option<String>, String> {
    if input == "err" {
        Err("error".to_string())
    } else if input.is_empty() {
        Ok(None)
    } else {
        Ok(Some(input.to_string()))
    }
}

#[test]
fn test_narrate_option() {
    assert_eq!(narrate_option("test"), Some("test".to_string()));
    assert_eq!(narrate_option(""), None);
}

#[test]
fn test_narrate_complex() {
    assert_eq!(narrate_complex("test").unwrap(), Some("test".to_string()));
    assert_eq!(narrate_complex("").unwrap(), None);
    assert!(narrate_complex("err").is_err());
}

// ============================================================================
// Where Clause Tests
// ============================================================================

#[narrate(action = "where", human = "With where clause")]
fn narrate_where<T>(value: T) -> String
where
    T: ToString,
{
    value.to_string()
}

#[trace_fn]
fn trace_where<T>(value: T) -> String
where
    T: ToString,
{
    value.to_string()
}

#[test]
fn test_narrate_where() {
    assert_eq!(narrate_where(42), "42");
}

#[test]
fn test_trace_where() {
    assert_eq!(trace_where(42), "42");
}
