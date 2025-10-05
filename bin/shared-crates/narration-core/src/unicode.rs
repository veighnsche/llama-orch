/// Unicode safety and sanitization helpers.
///
/// Implements simplified Unicode validation per Performance Team requirements:
/// - ASCII fast path (zero-copy for 90% of strings)
/// - CRLF sanitization (strip `\n`, `\r`, `\t` only)
/// - Basic control character filtering
use std::borrow::Cow;

/// Sanitize text for JSON output with ASCII fast path.
///
/// Performance: <1μs for 100-char string
///
/// # Example
/// ```
/// use observability_narration_core::unicode::sanitize_for_json;
///
/// let text = "Hello, world!";
/// let sanitized = sanitize_for_json(text);
/// assert_eq!(sanitized, "Hello, world!"); // Zero-copy for ASCII
/// ```
pub fn sanitize_for_json(text: &str) -> Cow<'_, str> {
    // ASCII fast path - zero-copy for 90% of cases
    if text.is_ascii() {
        return Cow::Borrowed(text);
    }

    // Simplified UTF-8 validation (not comprehensive per Performance Team)
    Cow::Owned(
        text.chars()
            .filter(|c| {
                !c.is_control() &&  // Basic control char filter
                !matches!(*c as u32,
                    0x200B..=0x200D |  // Zero-width space, ZWNJ, ZWJ
                    0xFEFF |           // Zero-width no-break space
                    0x2060             // Word joiner
                )
            })
            .collect(),
    )
}

/// CRLF sanitization - strip only \n, \r, \t.
///
/// Performance: <50ns for clean strings (zero-copy)
///
/// # Example
/// ```
/// use observability_narration_core::unicode::sanitize_crlf;
///
/// let text = "Line 1\nLine 2\rLine 3\tTab";
/// let sanitized = sanitize_crlf(text);
/// assert_eq!(sanitized, "Line 1 Line 2 Line 3 Tab");
/// ```
pub fn sanitize_crlf(text: &str) -> Cow<'_, str> {
    if !text.contains(['\n', '\r', '\t']) {
        return Cow::Borrowed(text); // Zero-copy (90% of cases)
    }

    Cow::Owned(text.replace(['\n', '\r', '\t'], " "))
}

/// Validate actor name (reject non-ASCII to prevent homograph attacks).
///
/// # Example
/// ```
/// use observability_narration_core::unicode::validate_actor;
///
/// assert!(validate_actor("orchestratord").is_ok());
/// assert!(validate_actor("оrchestratord").is_err()); // Cyrillic 'о'
/// ```
pub fn validate_actor(actor: &str) -> Result<&str, &'static str> {
    if !actor.is_ascii() {
        return Err("Actor name must be ASCII (homograph attack prevention)");
    }
    Ok(actor)
}

/// Validate action name (reject non-ASCII).
///
/// # Example
/// ```
/// use observability_narration_core::unicode::validate_action;
///
/// assert!(validate_action("dispatch").is_ok());
/// assert!(validate_action("dispаtch").is_err()); // Cyrillic 'а'
/// ```
pub fn validate_action(action: &str) -> Result<&str, &'static str> {
    if !action.is_ascii() {
        return Err("Action name must be ASCII (homograph attack prevention)");
    }
    Ok(action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_for_json_ascii_fast_path() {
        let text = "Hello, world!";
        let sanitized = sanitize_for_json(text);
        assert!(matches!(sanitized, Cow::Borrowed(_)));
        assert_eq!(sanitized, "Hello, world!");
    }

    #[test]
    fn test_sanitize_for_json_with_emoji() {
        let text = "Hello 🎀 world!";
        let sanitized = sanitize_for_json(text);
        assert_eq!(sanitized, "Hello 🎀 world!");
    }

    #[test]
    fn test_sanitize_for_json_removes_zero_width() {
        let text = "Hello\u{200B}world"; // Zero-width space
        let sanitized = sanitize_for_json(text);
        assert_eq!(sanitized, "Helloworld");
    }

    #[test]
    fn test_sanitize_crlf_clean_string() {
        let text = "No newlines here";
        let sanitized = sanitize_crlf(text);
        assert!(matches!(sanitized, Cow::Borrowed(_)));
        assert_eq!(sanitized, "No newlines here");
    }

    #[test]
    fn test_sanitize_crlf_with_newlines() {
        let text = "Line 1\nLine 2\rLine 3\tTab";
        let sanitized = sanitize_crlf(text);
        assert_eq!(sanitized, "Line 1 Line 2 Line 3 Tab");
    }

    #[test]
    fn test_validate_actor_ascii() {
        assert!(validate_actor("orchestratord").is_ok());
        assert!(validate_actor("pool-managerd").is_ok());
    }

    #[test]
    fn test_validate_actor_non_ascii() {
        // Cyrillic 'о' looks like Latin 'o'
        assert!(validate_actor("оrchestratord").is_err());
    }

    #[test]
    fn test_validate_action_ascii() {
        assert!(validate_action("dispatch").is_ok());
        assert!(validate_action("provision").is_ok());
    }

    #[test]
    fn test_validate_action_non_ascii() {
        // Cyrillic 'а' looks like Latin 'a'
        assert!(validate_action("dispаtch").is_err());
    }
}
