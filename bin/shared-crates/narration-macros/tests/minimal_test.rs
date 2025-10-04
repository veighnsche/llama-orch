//! Minimal test to debug macro behavior

use observability_narration_macros::narrate;

#[narrate(
    action = "test",
    human = "Testing"
)]
fn test_function() -> String {
    "result".to_string()
}

#[test]
fn test_it_works() {
    let result = test_function();
    assert_eq!(result, "result");
}
