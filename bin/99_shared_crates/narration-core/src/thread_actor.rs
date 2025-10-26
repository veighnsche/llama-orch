// TEAM-309: Thread-local target storage
// TEAM-311: Fixed to use stack for nested function calls
//! Thread-local target (function name) for proc macro injection
//!
//! This allows #[narrate_fn] to set a target (function name) that all n!() calls
//! inside the function will automatically use.
//!
//! TEAM-311: Uses a stack to support nested #[narrate_fn] calls

use std::cell::RefCell;

thread_local! {
    static THREAD_TARGET_STACK: RefCell<Vec<String>> = RefCell::new(Vec::new());
}

/// Set thread-local target (called by narrate_fn macro)
/// 
/// TEAM-311: Pushes onto stack instead of replacing
#[doc(hidden)]
pub fn set_target(target: &str) {
    THREAD_TARGET_STACK.with(|stack| {
        stack.borrow_mut().push(target.to_string());
    });
}

/// Clear thread-local target (called by narrate_fn macro on exit)
///
/// TEAM-311: Pops from stack instead of clearing completely
#[doc(hidden)]
pub fn clear_target() {
    THREAD_TARGET_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

/// Get thread-local target if set
///
/// TEAM-311: Returns the top of the stack (most recent function)
pub(crate) fn get_target() -> Option<String> {
    THREAD_TARGET_STACK.with(|stack| {
        stack.borrow().last().cloned()
    })
}
