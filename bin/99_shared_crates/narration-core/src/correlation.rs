/// Correlation ID helpers for request tracking across services.
///
/// Provides utilities for generating, validating, and propagating correlation IDs.
use uuid::Uuid;

/// Generate a new correlation ID (UUID v4).
///
/// # Example
/// ```
/// use observability_narration_core::correlation::generate_correlation_id;
///
/// let correlation_id = generate_correlation_id();
/// assert_eq!(correlation_id.len(), 36);
/// ```
pub fn generate_correlation_id() -> String {
    Uuid::new_v4().to_string()
}

/// Validate a correlation ID format (UUID v4).
///
/// Performs byte-level validation for performance (<100ns).
///
/// # Example
/// ```
/// use observability_narration_core::correlation::validate_correlation_id;
///
/// assert!(validate_correlation_id("550e8400-e29b-41d4-a716-446655440000").is_some());
/// assert!(validate_correlation_id("invalid").is_none());
/// ```
pub fn validate_correlation_id(id: &str) -> Option<&str> {
    if id.len() != 36 {
        return None;
    }

    let bytes = id.as_bytes();

    // Check hyphens at positions 8, 13, 18, 23
    if bytes[8] != b'-' || bytes[13] != b'-' || bytes[18] != b'-' || bytes[23] != b'-' {
        return None;
    }

    // Check all other positions are hex digits
    for (i, &b) in bytes.iter().enumerate() {
        if i == 8 || i == 13 || i == 18 || i == 23 {
            continue;
        }
        if !b.is_ascii_hexdigit() {
            return None;
        }
    }

    Some(id) // Return borrowed (zero-copy)
}

/// Extract correlation ID from HTTP header value.
///
/// Validates and returns the correlation ID if present and valid.
///
/// # Example
/// ```
/// use observability_narration_core::correlation::from_header;
///
/// let correlation_id = from_header("550e8400-e29b-41d4-a716-446655440000");
/// assert!(correlation_id.is_some());
/// ```
pub fn from_header(header_value: &str) -> Option<String> {
    validate_correlation_id(header_value.trim()).map(|s| s.to_string())
}

/// Propagate correlation ID to downstream service.
///
/// Returns the correlation ID for inclusion in HTTP headers.
///
/// # Example
/// ```
/// use observability_narration_core::correlation::propagate;
///
/// let correlation_id = "550e8400-e29b-41d4-a716-446655440000";
/// let header_value = propagate(correlation_id);
/// assert_eq!(header_value, correlation_id);
/// ```
pub fn propagate(correlation_id: &str) -> &str {
    correlation_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_correlation_id() {
        let id = generate_correlation_id();
        assert_eq!(id.len(), 36);
        assert!(validate_correlation_id(&id).is_some());
    }

    #[test]
    fn test_validate_correlation_id() {
        // Valid UUID v4
        assert!(validate_correlation_id("550e8400-e29b-41d4-a716-446655440000").is_some());

        // Invalid: wrong length
        assert!(validate_correlation_id("550e8400").is_none());

        // Invalid: missing hyphens
        assert!(validate_correlation_id("550e8400e29b41d4a716446655440000").is_none());

        // Invalid: non-hex characters
        assert!(validate_correlation_id("550e8400-e29b-41d4-a716-44665544000g").is_none());
    }

    #[test]
    fn test_from_header() {
        // Valid with whitespace
        assert!(from_header("  550e8400-e29b-41d4-a716-446655440000  ").is_some());

        // Invalid
        assert!(from_header("invalid").is_none());
    }

    #[test]
    fn test_propagate() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        assert_eq!(propagate(id), id);
    }
}
