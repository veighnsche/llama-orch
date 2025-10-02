# Model Loader — Dependency Specification

**Status**: Draft  
**Last Updated**: 2025-10-02

---

## 0. Overview

This document specifies all dependencies for the `model-loader` crate, including rationale, security considerations, and version requirements.

**Dependency Philosophy**:
- Minimize dependencies (reduce attack surface)
- Prefer workspace-managed versions (consistency)
- Use professionally audited crates for cryptography
- No dependencies on network or async runtime (stateless utility)

---

## 1. Production Dependencies

### 1.1 Core Infrastructure

#### `thiserror` (workspace = true)

**Purpose**: Error type definitions with `#[derive(Error)]`

**Used for**:
```rust
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    
    #[error("model too large: {0} > {1}")]
    TooLarge(usize, usize),
    // ...
}
```

**Why this crate**:
- Industry standard for Rust error handling
- Provides `Display` and `Error` trait implementations
- Better than `anyhow` for library crates (structured errors)
- Zero runtime overhead

**Alternatives considered**:
- ❌ `anyhow` — Too opaque for API consumers
- ❌ Manual `impl Error` — Too much boilerplate

**Security considerations**: None (compile-time only)

**Status**: ✅ Already added

---

#### `tracing` (workspace = true)

**Purpose**: Structured logging for validation events

**Used for**:
```rust
tracing::info!(
    model_path = ?request.model_path,
    expected_hash = ?request.expected_hash,
    "Model load started"
);

tracing::error!(
    error = %e,
    model_path = ?request.model_path,
    "Model load failed"
);

tracing::warn!(
    event = "security_incident",
    incident_type = "path_traversal",
    attempted_path = ?request.model_path,
    "Path traversal attempt detected"
);
```

**Why this crate**:
- Industry standard for structured logging in Rust
- Zero-cost abstractions (compile-time filtering)
- Integration with `tracing-subscriber` for JSON output
- Supports correlation IDs and distributed tracing

**Alternatives considered**:
- ❌ `log` — Less structured, no context propagation
- ❌ `slog` — More complex, less ecosystem support

**Security considerations**:
- Never log file contents or sensitive data
- Sanitize paths before logging (use `?` debug format)
- Log security incidents for audit trail

**Status**: ✅ Already added

---

### 1.2 Cryptography

#### `sha2` (workspace = true)

**Purpose**: SHA-256 hash computation for integrity verification

**Used for**:
```rust
use sha2::{Sha256, Digest};

fn compute_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
```

**Why this crate**:
- RustCrypto project (professionally audited)
- FIPS 140-2 approved algorithm
- Pure Rust implementation (no C dependencies)
- Constant-time operations (side-channel resistant)

**Alternatives considered**:
- ❌ `ring` — Includes unnecessary algorithms, larger binary
- ❌ `openssl` — C dependency, harder to audit

**Security considerations**:
- Use SHA-256 (not SHA-1, which is broken)
- Hash entire file before GGUF parsing (fail fast)
- Log computed hash for audit trail

**Audit status**: RustCrypto audited by NCC Group (2020)

**Status**: ✅ Already added

---

### 1.3 Security (Critical)

#### `input-validation` (path = "../../shared-crates/input-validation")

**Purpose**: Centralized input validation and sanitization

**Used for**:
```rust
use input_validation::{validate_path, validate_hex_string};

// Path validation (prevents CWE-22: Path Traversal)
let canonical_path = validate_path(request.model_path, &allowed_root)
    .map_err(|e| LoadError::PathValidationFailed(e.to_string()))?;

// Hash format validation
validate_hex_string(expected_hash, 64)
    .map_err(|e| LoadError::InvalidFormat(e.to_string()))?;
```

**Why this crate**:
- Centralized security boundary (single source of truth)
- Prevents path traversal (CWE-22)
- Prevents injection attacks (CWE-20)
- Consistent validation across all services

**Functions used**:
- `validate_path()` — Path canonicalization + containment check
- `validate_hex_string()` — Hex format validation (64 chars for SHA-256)
- `validate_identifier()` — Shard ID validation (future)

**Security requirements**:
- **PATH-001 to PATH-008**: Path security requirements
- **HASH-007**: Hash format validation
- Implements SECURITY_AUDIT_EXISTING_CODEBASE.md Vulnerability #9

**Status**: ⬜ **MUST ADD** (P0 Critical)

**Blocking**: Cannot implement path security without this

---

### 1.4 Serialization (Optional)

#### `serde` (workspace = true, features = ["derive"], optional = true)

**Purpose**: Serialize/deserialize GGUF metadata

**Used for** (post-M0):
```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv: HashMap<String, MetadataValue>,
}

pub fn extract_metadata(&self, bytes: &[u8]) -> Result<GgufMetadata>
```

**Why this crate**:
- Industry standard for Rust serialization
- Zero-copy deserialization support
- Extensive ecosystem support

**When to enable**:
- Post-M0 when implementing `extract_metadata()` API
- Needed for capability-matcher integration

**Feature flag**: `metadata-extraction`

**Status**: ⬜ Add when implementing metadata extraction

---

#### `bytes` (workspace = true, optional = true)

**Purpose**: Zero-copy buffer handling for large models

**Used for** (post-M0):
```rust
use bytes::{Buf, Bytes};

// Zero-copy slicing for large buffers
pub fn validate_bytes_streaming(&self, bytes: Bytes) -> Result<()> {
    let mut cursor = bytes;
    
    // Read without copying
    let magic = cursor.get_u32_le();
    // ...
}
```

**Why this crate**:
- Zero-copy buffer operations
- Efficient for large files (> 10GB)
- Reference-counted slices (no allocation)

**When to enable**:
- Post-M0 when implementing streaming validation
- Performance optimization for large models

**Feature flag**: `streaming`

**Status**: ⬜ Add for performance optimization (P2)

---

### 1.5 Signature Verification (Optional)

#### `ed25519-dalek` (version = "2.0", optional = true)

**Purpose**: Cryptographic signature verification for model authenticity

**Used for** (post-M0):
```rust
use ed25519_dalek::{PublicKey, Signature, Verifier};

pub struct LoadRequest<'a> {
    pub model_path: &'a Path,
    pub expected_hash: Option<&'a str>,
    pub signature: Option<&'a [u8]>,        // ← New
    pub public_key: Option<&'a PublicKey>,  // ← New
    pub max_size: usize,
}

fn verify_signature(&self, bytes: &[u8], sig: &[u8], pubkey: &PublicKey) -> Result<()> {
    let signature = Signature::from_bytes(sig)
        .map_err(|_| LoadError::SignatureVerificationFailed)?;
    
    pubkey.verify(bytes, &signature)
        .map_err(|_| LoadError::SignatureVerificationFailed)?;
    
    Ok(())
}
```

**Why this crate**:
- Ed25519 is modern, fast, and secure
- Smaller signatures than RSA (64 bytes)
- Constant-time operations (side-channel resistant)
- Pure Rust implementation

**Alternatives considered**:
- ❌ RSA — Larger signatures, slower verification
- ❌ ECDSA — More complex, potential timing attacks

**When to enable**:
- Post-M0 when implementing signature verification
- Needed for model provenance and supply chain security

**Feature flag**: `signature-verification`

**Security considerations**:
- Signature verification MUST occur before hash verification
- Public keys MUST be distributed securely (not in model file)
- Consider certificate chains for model publishers

**Status**: ⬜ Add for signature verification (P3)

---

### 1.6 Async I/O (Optional)

#### `tokio` (workspace = true, features = ["fs", "io-util"], optional = true)

**Purpose**: Async file I/O for non-blocking model loading

**Used for** (post-M0):
```rust
use tokio::fs;
use tokio::io::AsyncReadExt;

pub async fn load_and_validate_async(&self, request: LoadRequest<'_>) -> Result<Vec<u8>> {
    // Non-blocking file read
    let mut file = fs::File::open(request.model_path).await?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).await?;
    
    // Validate
    self.validate_bytes(&bytes, request.expected_hash)?;
    
    Ok(bytes)
}
```

**Why this crate**:
- Industry standard async runtime for Rust
- Non-blocking I/O prevents worker blocking
- Efficient for concurrent model loads

**When to enable**:
- Post-M0 when worker-api uses async handlers
- Performance optimization for concurrent loads

**Feature flag**: `async`

**Status**: ⬜ Add for async support (P3)

---

### 1.7 Advanced Parsing (Optional)

#### `regex` (workspace = true, optional = true)

**Purpose**: Advanced GGUF metadata parsing

**Used for** (post-M0):
```rust
use regex::Regex;

fn extract_model_architecture(&self, metadata: &GgufMetadata) -> Result<String> {
    let arch_regex = Regex::new(r"^general\.architecture$")?;
    
    for (key, value) in &metadata.metadata_kv {
        if arch_regex.is_match(key) {
            return Ok(value.as_string()?);
        }
    }
    
    Err(LoadError::InvalidFormat("Missing architecture".into()))
}
```

**Why this crate**:
- Powerful pattern matching for metadata
- Compiled regex (fast)
- Unicode support

**When to enable**:
- Post-M0 when implementing advanced metadata extraction
- Needed for complex model capability detection

**Feature flag**: `metadata-extraction`

**Security considerations**:
- Regex complexity can cause ReDoS (Regular Expression Denial of Service)
- Use simple patterns only
- Set timeout limits for regex matching

**Status**: ⬜ Add for metadata parsing (P3)

---

## 2. Development Dependencies

### 2.1 Property Testing (Critical)

#### `proptest` (workspace = true)

**Purpose**: Property-based testing for parser robustness

**Used for**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn parser_never_panics(bytes: Vec<u8>) {
        let loader = ModelLoader::new();
        let _ = loader.validate_bytes(&bytes, None);
        // Should never panic on any input
    }
    
    #[test]
    fn valid_gguf_always_accepted(
        tensor_count in 1usize..100,
        metadata_pairs in 0usize..10
    ) {
        let gguf = create_valid_gguf(tensor_count, metadata_pairs);
        let loader = ModelLoader::new();
        assert!(loader.validate_bytes(&gguf, None).is_ok());
    }
    
    #[test]
    fn oversized_tensor_count_rejected(
        count in 10_001usize..100_000
    ) {
        let gguf = create_gguf_with_tensor_count(count);
        let loader = ModelLoader::new();
        assert!(loader.validate_bytes(&gguf, None).is_err());
    }
}
```

**Why this crate**:
- Generates random test inputs (fuzz-like testing)
- Finds edge cases that manual tests miss
- Shrinks failing inputs to minimal examples
- Industry standard for Rust property testing

**Security requirements**:
- **Testing requirement**: Property tests for parser robustness
- **GGUF-011**: Parser must fail fast on invalid input
- Implements SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #19

**Test coverage required**:
- Parser never panics on any input
- Valid GGUF files always accepted
- Invalid GGUF files always rejected
- Resource limits enforced (tensor count, string length)

**Status**: ⬜ **MUST ADD** (P0 Critical)

**Blocking**: Cannot verify parser security without property tests

---

### 2.2 Test Utilities

#### `tempfile` (version = "3.8")

**Purpose**: Create temporary files and directories for testing

**Used for**:
```rust
use tempfile::TempDir;

#[test]
fn test_load_from_file() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test-model.gguf");
    
    // Write test GGUF file
    std::fs::write(&model_path, create_valid_gguf()).unwrap();
    
    // Test loading
    let loader = ModelLoader::new();
    let result = loader.load_and_validate(LoadRequest {
        model_path: &model_path,
        expected_hash: None,
        max_size: 1000,
    });
    
    assert!(result.is_ok());
}

#[test]
fn test_rejects_symlink_escape() {
    let temp_dir = TempDir::new().unwrap();
    let symlink_path = temp_dir.path().join("evil.gguf");
    
    // Create symlink outside allowed directory
    std::os::unix::fs::symlink("/etc/passwd", &symlink_path).unwrap();
    
    let loader = ModelLoader::new();
    let result = loader.load_and_validate(LoadRequest {
        model_path: &symlink_path,
        expected_hash: None,
        max_size: 1000,
    });
    
    assert!(matches!(result, Err(LoadError::PathValidationFailed(_))));
}
```

**Why this crate**:
- Automatic cleanup (RAII)
- Cross-platform (Windows, Unix)
- Secure temp file creation
- Industry standard for Rust testing

**Test coverage required**:
- File loading tests
- Path validation tests
- Symlink attack tests
- Permission tests

**Status**: ⬜ **MUST ADD** (P0 Critical)

**Blocking**: Cannot test path security without temp files

---

#### `insta` (workspace = true, features = ["yaml"])

**Purpose**: Snapshot testing for error messages and validation output

**Used for**:
```rust
use insta::assert_yaml_snapshot;

#[test]
fn test_error_message_format() {
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&[0x00, 0x00, 0x00, 0x00], None);
    
    // Snapshot the error message
    assert_yaml_snapshot!(result.unwrap_err());
}

#[test]
fn test_gguf_metadata_extraction() {
    let gguf = create_test_gguf_with_metadata();
    let loader = ModelLoader::new();
    let metadata = loader.extract_metadata(&gguf).unwrap();
    
    // Snapshot the metadata structure
    assert_yaml_snapshot!(metadata);
}
```

**Why this crate**:
- Regression testing for error messages
- Verify error format doesn't change unexpectedly
- Easy to review changes (git diff)
- Industry standard for Rust snapshot testing

**When to use**:
- Error message format verification
- Metadata extraction output verification
- Regression testing

**Status**: ⬜ Add for snapshot testing (P2)

---

## 3. Feature Flags

### 3.1 Feature Matrix

```toml
[features]
default = []
signature-verification = ["ed25519-dalek"]
async = ["tokio"]
metadata-extraction = ["serde", "regex"]
streaming = ["bytes"]
full = ["signature-verification", "async", "metadata-extraction", "streaming"]
```

### 3.2 Feature Descriptions

**`default`** (empty):
- Minimal dependencies for M0
- Synchronous file loading only
- Hash verification only (no signatures)
- No metadata extraction

**`signature-verification`**:
- Adds `ed25519-dalek` dependency
- Enables `verify_signature()` method
- Verifies model authenticity (not just integrity)
- Use case: Production deployments with signed models

**`async`**:
- Adds `tokio` dependency (fs + io-util features)
- Enables `load_and_validate_async()` method
- Non-blocking file I/O
- Use case: Async worker-api handlers

**`metadata-extraction`**:
- Adds `serde` and `regex` dependencies
- Enables `extract_metadata()` method
- Parses GGUF metadata key-value pairs
- Use case: Capability matching, model introspection

**`streaming`**:
- Adds `bytes` dependency
- Enables `validate_bytes_streaming()` method
- Zero-copy buffer operations
- Use case: Large models (> 10GB), memory optimization

**`full`**:
- Enables all features
- Use case: Development, testing, advanced deployments

---

## 4. Dependency Security Analysis

### 4.1 Supply Chain Security

**Dependency audit**:
```bash
# Check for known vulnerabilities
cargo audit

# Check for outdated dependencies
cargo outdated

# Check dependency tree
cargo tree -p model-loader
```

**Trusted sources**:
- ✅ `thiserror` — dtolnay (Rust core team)
- ✅ `tracing` — tokio-rs (Rust foundation)
- ✅ `sha2` — RustCrypto (audited by NCC Group)
- ✅ `proptest` — proptest-rs (widely used)
- ✅ `tempfile` — Stebalien (Rust ecosystem)
- ✅ `ed25519-dalek` — dalek-cryptography (audited)

**Audit status**:
- `sha2`: Audited by NCC Group (2020)
- `ed25519-dalek`: Audited by Quarkslab (2020)
- Others: Community-maintained, widely used

### 4.2 Dependency Minimization

**Why minimal dependencies matter**:
- Smaller attack surface
- Faster compilation
- Easier security audits
- Fewer supply chain risks

**Current dependency count**:
- Production (required): 3 (`thiserror`, `tracing`, `sha2`)
- Production (critical): +1 (`input-validation`)
- Production (optional): 5 (features)
- Development: 3 (`proptest`, `tempfile`, `insta`)

**Total**: 4 required, 8 optional

**Comparison**:
- vram-residency: 6 required (TIER 1)
- model-loader: 4 required (TIER 1)
- ✅ Minimal for security-critical crate

### 4.3 Version Pinning Strategy

**Workspace dependencies** (preferred):
- Use `workspace = true` for consistency
- Versions managed in root `Cargo.toml`
- All services use same versions
- Easier to audit and update

**Direct dependencies** (when needed):
- Pin exact versions for security-critical crates
- Use `=` for exact version (not `^` or `~`)
- Document why specific version is required

**Example**:
```toml
# Workspace dependency (preferred)
sha2.workspace = true

# Direct dependency (if needed)
ed25519-dalek = "=2.0.0"  # Exact version for security audit
```

---

## 5. Dependency Integration

### 5.1 input-validation Integration

**Required functions**:
```rust
use input_validation::{validate_path, validate_hex_string};

impl ModelLoader {
    pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
        // Path validation
        let allowed_root = PathBuf::from("/var/lib/llorch/models");
        let canonical_path = validate_path(request.model_path, &allowed_root)
            .map_err(|e| LoadError::PathValidationFailed(e.to_string()))?;
        
        // Read file
        let bytes = std::fs::read(&canonical_path)?;
        
        // Hash validation
        if let Some(expected_hash) = request.expected_hash {
            validate_hex_string(expected_hash, 64)
                .map_err(|e| LoadError::InvalidFormat(e.to_string()))?;
            
            self.verify_hash(&bytes, expected_hash)?;
        }
        
        // GGUF validation
        self.validate_gguf(&bytes)?;
        
        Ok(bytes)
    }
}
```

**Error mapping**:
```rust
// Map ValidationError to LoadError
match validate_path(path, &root) {
    Ok(canonical) => canonical,
    Err(ValidationError::PathTraversal) => {
        return Err(LoadError::PathValidationFailed("Path traversal detected".into()));
    },
    Err(ValidationError::PathOutsideRoot { path }) => {
        return Err(LoadError::PathValidationFailed(
            format!("Path outside allowed directory: {}", path)
        ));
    },
    Err(e) => {
        return Err(LoadError::PathValidationFailed(e.to_string()));
    },
}
```

### 5.2 proptest Integration

**Test organization**:
```rust
// tests/property_tests.rs
use proptest::prelude::*;
use model_loader::{ModelLoader, LoadError};

// Strategy: Generate valid GGUF files
fn valid_gguf_strategy() -> impl Strategy<Value = Vec<u8>> {
    (1usize..100, 0usize..10).prop_map(|(tensor_count, metadata_pairs)| {
        create_valid_gguf(tensor_count, metadata_pairs)
    })
}

// Strategy: Generate invalid GGUF files
fn invalid_gguf_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        // Invalid magic
        Just(vec![0x00, 0x00, 0x00, 0x00]),
        // Oversized tensor count
        (10_001usize..100_000).prop_map(create_gguf_with_tensor_count),
        // Truncated file
        (0usize..100).prop_map(|len| vec![0x47, 0x47, 0x55, 0x46].into_iter().take(len).collect()),
    ]
}

proptest! {
    #[test]
    fn valid_gguf_accepted(gguf in valid_gguf_strategy()) {
        let loader = ModelLoader::new();
        assert!(loader.validate_bytes(&gguf, None).is_ok());
    }
    
    #[test]
    fn invalid_gguf_rejected(gguf in invalid_gguf_strategy()) {
        let loader = ModelLoader::new();
        assert!(loader.validate_bytes(&gguf, None).is_err());
    }
}
```

---

## 6. Dependency Alternatives Considered

### 6.1 Why NOT These Dependencies

**`anyhow`** ❌:
- **Reason**: Too opaque for library crates
- **Alternative**: `thiserror` (structured errors)
- **Why better**: API consumers need specific error types

**`log`** ❌:
- **Reason**: Less structured than `tracing`
- **Alternative**: `tracing` (structured logging)
- **Why better**: Context propagation, correlation IDs

**`ring`** ❌:
- **Reason**: Includes unnecessary algorithms, larger binary
- **Alternative**: `sha2` (minimal, pure Rust)
- **Why better**: Smaller binary, easier to audit

**`openssl`** ❌:
- **Reason**: C dependency, harder to audit
- **Alternative**: `sha2`, `ed25519-dalek` (pure Rust)
- **Why better**: No C dependencies, memory safety

**`serde_yaml`** ❌:
- **Reason**: Not needed (GGUF is binary, not YAML)
- **Alternative**: None needed
- **Why better**: Fewer dependencies

**`reqwest`** ❌:
- **Reason**: No network operations in this crate
- **Alternative**: None needed
- **Why better**: Stateless utility, no I/O

**`axum`** ❌:
- **Reason**: No HTTP server in this crate
- **Alternative**: None needed
- **Why better**: worker-api handles HTTP

**`audit-logging`** ❌:
- **Reason**: Use `tracing` for now
- **Alternative**: `tracing` (sufficient for M0)
- **Why better**: worker-api handles audit events

---

## 7. Implementation Roadmap

### 7.1 Phase 1: M0 Essentials (Immediate)

**Add to Cargo.toml**:
```toml
[dependencies]
# Already present
thiserror.workspace = true
tracing.workspace = true
sha2.workspace = true

# MUST ADD (P0 Critical)
input-validation = { path = "../../shared-crates/input-validation" }

[dev-dependencies]
# MUST ADD (P0 Critical)
proptest.workspace = true
tempfile = "3.8"
```

**Blocking work**:
- ✅ `thiserror` — Already added
- ✅ `tracing` — Already added
- ✅ `sha2` — Already added
- ⬜ `input-validation` — **CRITICAL** (path security)
- ⬜ `proptest` — **CRITICAL** (security testing)
- ⬜ `tempfile` — **CRITICAL** (path testing)

**Timeline**: Immediate (blocking for M0)

---

### 7.2 Phase 2: Production Hardening (Next)

**Add optional dependencies**:
```toml
[features]
metadata-extraction = ["serde", "regex"]
streaming = ["bytes"]

[dependencies]
serde = { workspace = true, optional = true }
bytes = { workspace = true, optional = true }
regex = { workspace = true, optional = true }

[dev-dependencies]
insta = { workspace = true, features = ["yaml"] }
```

**Timeline**: Post-M0 (1-2 weeks)

---

### 7.3 Phase 3: Advanced Features (Post-M0)

**Add signature verification**:
```toml
[features]
signature-verification = ["ed25519-dalek"]
async = ["tokio"]

[dependencies]
ed25519-dalek = { version = "2.0", optional = true }
tokio = { workspace = true, features = ["fs", "io-util"], optional = true }
```

**Timeline**: Post-M0 (4-6 weeks)

---

## 8. Dependency Verification

### 8.1 Audit Commands

**Check for vulnerabilities**:
```bash
# Install cargo-audit
cargo install cargo-audit

# Run audit
cargo audit -p model-loader
```

**Check for outdated dependencies**:
```bash
# Install cargo-outdated
cargo install cargo-outdated

# Check outdated
cargo outdated -p model-loader
```

**Inspect dependency tree**:
```bash
# Show dependency tree
cargo tree -p model-loader

# Show duplicate dependencies
cargo tree -p model-loader --duplicates

# Show features
cargo tree -p model-loader --features full
```

### 8.2 License Compliance

**Allowed licenses**:
- MIT
- Apache-2.0
- BSD-3-Clause
- ISC

**Check licenses**:
```bash
# Install cargo-license
cargo install cargo-license

# Check licenses
cargo license -p model-loader
```

**Expected licenses**:
- `thiserror`: MIT OR Apache-2.0
- `tracing`: MIT
- `sha2`: MIT OR Apache-2.0
- `proptest`: MIT OR Apache-2.0
- `tempfile`: MIT OR Apache-2.0
- `ed25519-dalek`: BSD-3-Clause

---

## 9. Refinement Opportunities

### 9.1 Streaming Hash Computation

**Current**: Load entire file, then compute hash  
**Future**: Stream file while computing hash

**Benefits**:
- Constant memory usage
- Faster failure on large files
- Better performance

**Dependencies needed**:
- `tokio` (async I/O)
- `bytes` (zero-copy buffers)

### 9.2 Multi-Format Support

**Current**: GGUF only  
**Future**: SafeTensors, PyTorch .bin

**Dependencies needed**:
- `safetensors` crate (for SafeTensors format)
- `zip` crate (for PyTorch .bin archives)

**Security considerations**:
- Each format needs separate parser
- Each parser needs separate fuzzing
- Shared validation framework

### 9.3 Parallel Hash Computation

**Current**: Single-threaded hash computation  
**Future**: Multi-threaded for large files

**Dependencies needed**:
- `rayon` (data parallelism)

**Benefits**:
- Faster hash computation for large models
- Better CPU utilization

### 9.4 Compression Support

**Current**: Uncompressed GGUF only  
**Future**: Compressed models (gzip, zstd)

**Dependencies needed**:
- `flate2` (gzip)
- `zstd` (zstd compression)

**Security considerations**:
- Decompression bombs (zip bombs)
- Memory exhaustion
- Validate decompressed size

---

## 10. References

**Dependency Documentation**:
- `thiserror`: https://docs.rs/thiserror
- `tracing`: https://docs.rs/tracing
- `sha2`: https://docs.rs/sha2
- `proptest`: https://docs.rs/proptest
- `tempfile`: https://docs.rs/tempfile
- `ed25519-dalek`: https://docs.rs/ed25519-dalek

**Security Audits**:
- RustCrypto (sha2): https://research.nccgroup.com/2020/02/26/public-report-rustcrypto-aes-gcm-and-chacha20poly1305-implementation-review/
- dalek-cryptography: https://blog.quarkslab.com/resources/2020-04-03-audit-dalek-libraries/20-04-594-REP.pdf

**Specifications**:
- `00_model-loader.md` — Functional specification
- `10_expectations.md` — Consumer expectations
- `20_security.md` — Security specification

---

**End of Dependency Specification**
