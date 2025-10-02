# Error Classification Guide

**Purpose**: Classify `LoadError` variants by retriability and severity to help consumers handle errors appropriately.

---

## Error Categories

### 🔴 Fatal (Non-Retriable)

**Definition**: Errors that indicate a fundamental problem that cannot be fixed by retrying.

**Consumer Action**: Log error, alert operator, reject request.

| Error Variant | Reason | Example |
|---------------|--------|---------|
| `HashMismatch` | File integrity compromised | Computed hash doesn't match expected |
| `InvalidFormat` | Malformed GGUF file | Wrong magic number, invalid version |
| `PathValidationFailed` | Security violation | Path traversal attempt |
| `TensorCountExceeded` | Resource limit violation | File has > 10,000 tensors |
| `StringTooLong` | Resource limit violation | String > 10MB |
| `BufferOverflow` | Malformed file structure | Read beyond buffer bounds |
| `InvalidDataType` | Invalid GGUF data type enum | Unknown type value |

**Rationale**: These errors indicate either:
- **Security violations** (path traversal, buffer overflow)
- **Integrity failures** (hash mismatch)
- **Format violations** (invalid GGUF structure)

Retrying will not fix these issues. The file or request is fundamentally invalid.

---

### 🟡 Transient (Retriable)

**Definition**: Errors that may succeed if retried (e.g., after fixing external conditions).

**Consumer Action**: Retry with backoff, check system resources.

| Error Variant | Reason | Retry Strategy |
|---------------|--------|----------------|
| `Io(kind: NotFound)` | File doesn't exist yet | Retry after file appears |
| `Io(kind: PermissionDenied)` | Temporary permission issue | Retry after fixing permissions |
| `TooLarge` | File exceeds size limit | Increase limit or use smaller file |

**Rationale**: These errors may be temporary:
- File might not be staged yet (NotFound)
- Permissions might be fixed by operator
- Size limit might be adjustable

---

### 🟠 Configuration (Operator Action Required)

**Definition**: Errors that require operator intervention but are not security violations.

**Consumer Action**: Alert operator, provide actionable guidance.

| Error Variant | Operator Action | Example |
|---------------|-----------------|---------|
| `TooLarge` | Increase `max_size` limit or provide smaller file | 50GB file, 10GB limit |
| `Io(kind: NotFound)` | Stage file to expected path | Missing model file |
| `Io(kind: PermissionDenied)` | Fix file permissions | Unreadable file |

---

## Error Severity Levels

### Critical (Immediate Action)

- `HashMismatch` — **Integrity violation, possible tampering**
- `PathValidationFailed` — **Security violation, possible attack**
- `BufferOverflow` — **Malformed file, possible exploit attempt**

**Action**: Log to security audit trail, alert security team.

### High (Operational Issue)

- `TensorCountExceeded` — Resource exhaustion attempt
- `StringTooLong` — Resource exhaustion attempt
- `InvalidFormat` — Malformed file

**Action**: Log error, reject request, notify operator.

### Medium (Configuration Issue)

- `TooLarge` — Size limit configuration
- `Io(PermissionDenied)` — Permission configuration

**Action**: Log error, provide guidance to operator.

### Low (Transient Issue)

- `Io(NotFound)` — File not yet available

**Action**: Log warning, retry with backoff.

---

## Usage Examples

### Example 1: Worker Load Handler

```rust
use model_loader::{LoadError, ModelLoader, LoadRequest};

fn handle_load(request: LoadRequest) -> Result<Vec<u8>, String> {
    let loader = ModelLoader::new();
    
    match loader.load_and_validate(request) {
        Ok(bytes) => Ok(bytes),
        
        // Fatal: Security violations
        Err(LoadError::PathValidationFailed(msg)) => {
            tracing::error!(
                security_event = "path_traversal_attempt",
                error = %msg,
                "SECURITY: Path validation failed"
            );
            Err("Security violation: invalid path".to_string())
        }
        
        Err(LoadError::HashMismatch { expected, actual }) => {
            tracing::error!(
                security_event = "integrity_violation",
                expected = %expected,
                actual = %actual,
                "SECURITY: Hash mismatch detected"
            );
            Err("Integrity violation: hash mismatch".to_string())
        }
        
        // Fatal: Format violations
        Err(LoadError::InvalidFormat(msg)) => {
            tracing::warn!(
                error = %msg,
                "Invalid GGUF format"
            );
            Err(format!("Invalid model format: {}", msg))
        }
        
        // Fatal: Resource limits
        Err(LoadError::TensorCountExceeded { count, max }) => {
            tracing::warn!(
                count = count,
                max = max,
                "Tensor count limit exceeded"
            );
            Err(format!("Model too complex: {} tensors (max {})", count, max))
        }
        
        // Transient: File not found
        Err(LoadError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!("Model file not found, may retry");
            Err("Model file not available".to_string())
        }
        
        // Other I/O errors
        Err(LoadError::Io(e)) => {
            tracing::error!(error = %e, "I/O error loading model");
            Err(format!("I/O error: {}", e))
        }
        
        // Configuration: Size limit
        Err(LoadError::TooLarge { actual, max }) => {
            tracing::warn!(
                actual = actual,
                max = max,
                "File size exceeds limit"
            );
            Err(format!("Model too large: {} bytes (max {})", actual, max))
        }
        
        // Other errors
        Err(e) => {
            tracing::error!(error = %e, "Unexpected error");
            Err(format!("Load failed: {}", e))
        }
    }
}
```

### Example 2: Retry Logic

```rust
use std::time::Duration;
use tokio::time::sleep;

async fn load_with_retry(
    loader: &ModelLoader,
    request: LoadRequest<'_>,
    max_retries: u32,
) -> Result<Vec<u8>, LoadError> {
    let mut attempt = 0;
    
    loop {
        match loader.load_and_validate(request.clone()) {
            Ok(bytes) => return Ok(bytes),
            
            // Retriable errors
            Err(LoadError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                attempt += 1;
                if attempt >= max_retries {
                    return Err(LoadError::Io(e));
                }
                
                let backoff = Duration::from_secs(2u64.pow(attempt));
                tracing::info!(
                    attempt = attempt,
                    backoff_secs = backoff.as_secs(),
                    "File not found, retrying"
                );
                sleep(backoff).await;
            }
            
            // Non-retriable errors
            Err(e) => return Err(e),
        }
    }
}
```

---

## Error Message Guidelines

### DO: Provide Context

```rust
// ✅ Good: Includes offset, expected, actual
LoadError::BufferOverflow {
    offset: 1024,
    length: 8,
    available: 1028,
}
// Error message: "buffer overflow: tried to read 8 bytes at offset 1024, but only 1028 bytes available"
```

### DON'T: Expose Sensitive Data

```rust
// ❌ Bad: Exposes file path
LoadError::PathValidationFailed(
    format!("Path /etc/passwd outside allowed root")
)

// ✅ Good: Generic message
LoadError::PathValidationFailed(
    "Path outside allowed directory".to_string()
)
```

### DO: Be Actionable

```rust
// ✅ Good: Tells user what to do
LoadError::TooLarge {
    actual: 50_000_000_000,
    max: 10_000_000_000,
}
// Suggests: increase max_size or use smaller file
```

---

## Testing Error Handling

```rust
#[test]
fn test_error_classification() {
    // Fatal errors should not be retried
    let fatal_errors = vec![
        LoadError::HashMismatch {
            expected: "abc".to_string(),
            actual: "def".to_string(),
        },
        LoadError::InvalidFormat("bad magic".to_string()),
        LoadError::PathValidationFailed("traversal".to_string()),
    ];
    
    for error in fatal_errors {
        assert!(!is_retriable(&error), "Fatal error marked as retriable: {:?}", error);
    }
    
    // Transient errors can be retried
    let transient_errors = vec![
        LoadError::Io(std::io::Error::from(std::io::ErrorKind::NotFound)),
    ];
    
    for error in transient_errors {
        assert!(is_retriable(&error), "Transient error marked as fatal: {:?}", error);
    }
}

fn is_retriable(error: &LoadError) -> bool {
    matches!(
        error,
        LoadError::Io(e) if e.kind() == std::io::ErrorKind::NotFound
    )
}
```

---

## References

- `.specs/00_model-loader.md` §9 — Error Handling specification
- `.specs/20_security.md` §3.7 — Error message security
- `src/error.rs` — Error type definitions
