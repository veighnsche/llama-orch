//! Error-model notes: this crate returns Result and should not panic on normal IO.
//! We rely on existing steps performing sequences of operations without panics.

// No dedicated steps needed; scenarios execute happy-path operations.
