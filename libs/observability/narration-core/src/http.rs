// http.rs — HTTP header propagation for Cloud Profile
// Implements CLOUD_PROFILE_NARRATION_REQUIREMENTS.md Section 3

/// Standard HTTP header names for correlation and tracing.
pub mod headers {
    pub const CORRELATION_ID: &str = "X-Correlation-Id";
    pub const TRACE_ID: &str = "X-Trace-Id";
    pub const SPAN_ID: &str = "X-Span-Id";
    pub const PARENT_SPAN_ID: &str = "X-Parent-Span-Id";
}

/// Extract correlation and trace context from HTTP headers.
///
/// Compatible with axum's `HeaderMap`.
///
/// # Example (axum handler)
/// ```rust,ignore
/// use axum::http::HeaderMap;
/// use observability_narration_core::http::extract_context_from_headers;
///
/// async fn my_handler(headers: HeaderMap) {
///     let (correlation_id, trace_id, span_id, parent_span_id) =
///         extract_context_from_headers(&headers);
///     
///     // Use in narration
///     narrate(NarrationFields {
///         correlation_id,
///         trace_id,
///         span_id,
///         parent_span_id,
///         ..Default::default()
///     });
/// }
/// ```
pub fn extract_context_from_headers<H>(
    headers: &H,
) -> (Option<String>, Option<String>, Option<String>, Option<String>)
where
    H: HeaderLike,
{
    let correlation_id = headers.get_str(headers::CORRELATION_ID);
    let trace_id = headers.get_str(headers::TRACE_ID);
    let span_id = headers.get_str(headers::SPAN_ID);
    let parent_span_id = headers.get_str(headers::PARENT_SPAN_ID);

    (correlation_id, trace_id, span_id, parent_span_id)
}

/// Inject correlation and trace context into HTTP headers.
///
/// Compatible with reqwest's `HeaderMap` and axum's `HeaderMap`.
///
/// # Example (reqwest client)
/// ```rust,ignore
/// use observability_narration_core::http::inject_context_into_headers;
///
/// let mut headers = reqwest::header::HeaderMap::new();
/// inject_context_into_headers(
///     &mut headers,
///     Some("req-xyz"),
///     Some("trace-123"),
///     Some("span-456"),
///     None,
/// );
///
/// let response = client
///     .get("http://pool-managerd:9200/v2/pools/default/status")
///     .headers(headers)
///     .send()
///     .await?;
/// ```
pub fn inject_context_into_headers<H>(
    headers: &mut H,
    correlation_id: Option<&str>,
    trace_id: Option<&str>,
    span_id: Option<&str>,
    parent_span_id: Option<&str>,
) where
    H: HeaderLike,
{
    if let Some(id) = correlation_id {
        headers.insert_str(headers::CORRELATION_ID, id);
    }
    if let Some(id) = trace_id {
        headers.insert_str(headers::TRACE_ID, id);
    }
    if let Some(id) = span_id {
        headers.insert_str(headers::SPAN_ID, id);
    }
    if let Some(id) = parent_span_id {
        headers.insert_str(headers::PARENT_SPAN_ID, id);
    }
}

/// Trait for header map abstraction (works with axum and reqwest).
///
/// This trait provides a common interface for extracting and injecting
/// correlation/trace context from HTTP headers, regardless of the HTTP
/// library being used (axum, reqwest, hyper, etc.).
///
/// # Safety
///
/// Implementations should handle invalid header values gracefully:
/// - `get_str()` returns `None` if header is missing or invalid UTF-8
/// - `insert_str()` should validate or sanitize header values
///
/// # Example Implementation
///
/// ```rust,ignore
/// impl HeaderLike for axum::http::HeaderMap {
///     fn get_str(&self, name: &str) -> Option<String> {
///         self.get(name)?.to_str().ok().map(String::from)
///     }
///     
///     fn insert_str(&mut self, name: &str, value: &str) {
///         if let Ok(header_value) = axum::http::HeaderValue::from_str(value) {
///             self.insert(name, header_value);
///         }
///     }
/// }
/// ```
pub trait HeaderLike {
    /// Get a header value as a String.
    /// Returns `None` if the header is missing or contains invalid UTF-8.
    fn get_str(&self, name: &str) -> Option<String>;
    
    /// Insert a header value.
    /// Implementations should validate or sanitize the value.
    fn insert_str(&mut self, name: &str, value: &str);
}

// Implementation for standard HashMap (for testing)
impl HeaderLike for std::collections::HashMap<String, String> {
    fn get_str(&self, name: &str) -> Option<String> {
        self.get(name).cloned()
    }

    fn insert_str(&mut self, name: &str, value: &str) {
        self.insert(name.to_string(), value.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_extract_context_from_headers() {
        let mut headers = HashMap::new();
        headers.insert(headers::CORRELATION_ID.to_string(), "req-xyz".to_string());
        headers.insert(headers::TRACE_ID.to_string(), "trace-123".to_string());
        headers.insert(headers::SPAN_ID.to_string(), "span-456".to_string());

        let (correlation_id, trace_id, span_id, parent_span_id) =
            extract_context_from_headers(&headers);

        assert_eq!(correlation_id, Some("req-xyz".to_string()));
        assert_eq!(trace_id, Some("trace-123".to_string()));
        assert_eq!(span_id, Some("span-456".to_string()));
        assert_eq!(parent_span_id, None);
    }

    #[test]
    fn test_inject_context_into_headers() {
        let mut headers = HashMap::new();

        inject_context_into_headers(
            &mut headers,
            Some("req-xyz"),
            Some("trace-123"),
            Some("span-456"),
            Some("parent-789"),
        );

        assert_eq!(headers.get(headers::CORRELATION_ID), Some(&"req-xyz".to_string()));
        assert_eq!(headers.get(headers::TRACE_ID), Some(&"trace-123".to_string()));
        assert_eq!(headers.get(headers::SPAN_ID), Some(&"span-456".to_string()));
        assert_eq!(headers.get(headers::PARENT_SPAN_ID), Some(&"parent-789".to_string()));
    }

    #[test]
    fn test_inject_partial_context() {
        let mut headers = HashMap::new();

        inject_context_into_headers(&mut headers, Some("req-xyz"), None, None, None);

        assert_eq!(headers.get(headers::CORRELATION_ID), Some(&"req-xyz".to_string()));
        assert_eq!(headers.get(headers::TRACE_ID), None);
    }

    #[test]
    fn test_roundtrip() {
        let mut headers = HashMap::new();

        inject_context_into_headers(
            &mut headers,
            Some("req-xyz"),
            Some("trace-123"),
            Some("span-456"),
            None,
        );

        let (correlation_id, trace_id, span_id, parent_span_id) =
            extract_context_from_headers(&headers);

        assert_eq!(correlation_id, Some("req-xyz".to_string()));
        assert_eq!(trace_id, Some("trace-123".to_string()));
        assert_eq!(span_id, Some("span-456".to_string()));
        assert_eq!(parent_span_id, None);
    }
}
