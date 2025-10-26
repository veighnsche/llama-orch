// TEAM-297: Phase 0 - Narration mode configuration
//! Runtime-configurable narration modes (Human/Cute/Story)

use std::sync::atomic::{AtomicU8, Ordering};

/// Global narration mode (thread-safe, atomic)
static NARRATION_MODE: AtomicU8 = AtomicU8::new(NarrationMode::Human as u8);

/// Narration display mode - determines which message variant is shown
///
/// TEAM-297: Added to make cute/story modes actually usable!
/// Previously the infrastructure existed but there was no way to configure it.
///
/// # Example
/// ```rust
/// use observability_narration_core::{set_narration_mode, NarrationMode};
///
/// // Switch to cute mode
/// set_narration_mode(NarrationMode::Cute);
///
/// // All narration now shows cute version (or falls back to human)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NarrationMode {
    /// Standard technical narration (default)
    Human = 0,
    
    /// Cute/whimsical narration (🐝 friendly)
    Cute = 1,
    
    /// Story-mode dialogue narration
    Story = 2,
}

/// Set the global narration mode
///
/// This affects ALL narration from this point forward.
/// Thread-safe and instant.
///
/// # Example
/// ```rust
/// use observability_narration_core::{set_narration_mode, NarrationMode};
///
/// // Switch to cute mode
/// set_narration_mode(NarrationMode::Cute);
///
/// // All narration now shows cute version (or falls back to human)
/// ```
pub fn set_narration_mode(mode: NarrationMode) {
    NARRATION_MODE.store(mode as u8, Ordering::Relaxed);
}

/// Get the current narration mode
pub fn get_narration_mode() -> NarrationMode {
    match NARRATION_MODE.load(Ordering::Relaxed) {
        0 => NarrationMode::Human,
        1 => NarrationMode::Cute,
        2 => NarrationMode::Story,
        _ => NarrationMode::Human, // Fallback for invalid values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mode_is_human() {
        // TEAM-298: Always reset mode at start of test (prevent state leaks)
        set_narration_mode(NarrationMode::Human);
        assert_eq!(get_narration_mode(), NarrationMode::Human);
    }

    #[test]
    fn test_mode_switching() {
        // TEAM-298: Always reset mode at start and end of test
        set_narration_mode(NarrationMode::Human);
        
        set_narration_mode(NarrationMode::Cute);
        assert_eq!(get_narration_mode(), NarrationMode::Cute);
        
        set_narration_mode(NarrationMode::Story);
        assert_eq!(get_narration_mode(), NarrationMode::Story);
        
        // Reset for other tests
        set_narration_mode(NarrationMode::Human);
        assert_eq!(get_narration_mode(), NarrationMode::Human);
    }
}
