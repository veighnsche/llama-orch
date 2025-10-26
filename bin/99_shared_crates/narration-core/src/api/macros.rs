// TEAM-300: Modular reorganization - Macro definitions
//! Macro definitions for narration system
//!
//! This module contains the macro definitions. The actual macro_rules! are in lib.rs
//! because they need to be at the crate root for #[macro_export] to work properly.
//!
//! This file documents the macro APIs for reference.

// Note: The actual macros are defined in lib.rs:
// - narrate!() - Legacy macro with provenance
// - narration_macro!() - Module-level macro creator
// - n!() - Ultra-concise narration macro (Phase 0)
// - narrate_concise!() - Alias for n!()
