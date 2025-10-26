// TEAM-309: Thread-local actor storage
//! Thread-local actor for proc macro injection
//!
//! This allows #[with_actor("name")] to set an actor that all n!() calls
//! inside the function will automatically use.

use std::cell::RefCell;

thread_local! {
    static THREAD_ACTOR: RefCell<Option<&'static str>> = RefCell::new(None);
}

/// Set thread-local actor (called by with_actor macro)
#[doc(hidden)]
pub fn set_actor(actor: &'static str) {
    THREAD_ACTOR.with(|a| {
        *a.borrow_mut() = Some(actor);
    });
}

/// Clear thread-local actor (called by with_actor macro on exit)
#[doc(hidden)]
pub fn clear_actor() {
    THREAD_ACTOR.with(|a| {
        *a.borrow_mut() = None;
    });
}

/// Get thread-local actor if set
pub(crate) fn get_actor() -> Option<&'static str> {
    THREAD_ACTOR.with(|a| *a.borrow())
}
