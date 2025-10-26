// TEAM-309: Thread-local target storage
//! Thread-local target (function name) for proc macro injection
//!
//! This allows #[narrate_fn] to set a target (function name) that all n!() calls
//! inside the function will automatically use.

use std::cell::RefCell;

thread_local! {
    static THREAD_TARGET: RefCell<Option<String>> = RefCell::new(None);
}

/// Set thread-local target (called by narrate_fn macro)
#[doc(hidden)]
pub fn set_target(target: &str) {
    THREAD_TARGET.with(|t| {
        *t.borrow_mut() = Some(target.to_string());
    });
}

/// Clear thread-local target (called by narrate_fn macro on exit)
#[doc(hidden)]
pub fn clear_target() {
    THREAD_TARGET.with(|t| {
        *t.borrow_mut() = None;
    });
}

/// Get thread-local target if set
pub(crate) fn get_target() -> Option<String> {
    THREAD_TARGET.with(|t| t.borrow().clone())
}
