# Model Loader — Security Specification

**Status**: Draft  
**Security Tier**: TIER 1 (Critical)  
**Last Updated**: 2025-10-02

---

## 0. Security Classification

### 0.1 Criticality Assessment

**Tier**: TIER 1 — Security-Critical

**Rationale**:
- First validation boundary for untrusted model bytes (security perimeter)
- Parses untrusted binary format (GGUF) from network or disk
- Controls what gets loaded into VRAM (integrity gatekeeper)
- Prevents buffer overflows and integer overflows (memory safety)
- Enforces path traversal protection (filesystem security)
- Validates cryptographic hashes (integrity verification)

**Impact of compromise**:
- **Remote Code Execution** via malicious GGUF files (buffer overflow)
- **Model poisoning** via hash bypass or signature forgery
- **Path traversal** enabling arbitrary file read
- **Denial of Service** via resource exhaustion (OOM, infinite loops)
- **Integer overflow** in tensor dimension calculations
- **Data exfiltration** via symlink attacks

---

## 1. Threat Model

### 1.1 Adversary Capabilities

**External Attacker** (network access):
- Can send malicious model files to Commit endpoint
- Can craft malformed GGUF headers to trigger parser bugs
- Can attempt path traversal via model_path parameter
- Can send oversized files to exhaust memory
- Cannot directly access worker filesystem (unless path traversal succeeds)

**Compromised pool-managerd**:
- Can send poisoned model bytes to worker
- Can forge expected_hash to bypass integrity checks
- Can attempt to load arbitrary files via path parameter
- Can send models with malicious GGUF payloads
- Can attempt to exhaust worker resources

**Malicious Model File**:
- Can contain crafted GGUF headers (buffer overflow triggers)
- Can specify excessive tensor counts (OOM attack)
- Can contain oversized strings (allocation DoS)
- Can have invalid tensor dimensions (integer overflow)
- Can contain backdoored weights (model poisoning)
- Can trigger parser edge cases (infinite loops, panics)

**Filesystem Attacker**:
- Can create symlinks to sensitive files
- Can use `..` sequences to escape allowed directory
- Can use null bytes to truncate paths
- Can use absolute paths to bypass root check

### 1.2 Assets to Protect

**Primary Assets**:
1. **Worker process integrity** — Prevent RCE via parser exploits
2. **Filesystem isolation** — Prevent arbitrary file read
3. **Model integrity** — Prevent poisoned models from loading
4. **Worker availability** — Prevent DoS via resource exhaustion

**Secondary Assets**:
5. **VRAM allocation** — Prevent loading invalid models into VRAM
6. **Error messages** — Prevent information leakage
7. **Validation state** — Ensure consistent security checks

---

## 2. Security Requirements (RFC-2119)

### 2.1 GGUF Parser Safety

**GGUF-001**: GGUF parser MUST enforce maximum tensor count (default 10,000).

**GGUF-002**: GGUF parser MUST enforce maximum file size (default 100GB).

**GGUF-003**: GGUF parser MUST enforce maximum string length (default 64KB).

**GGUF-004**: GGUF parser MUST enforce maximum metadata pairs (default 1,000).

**GGUF-005**: All GGUF reads MUST be bounds-checked before accessing bytes.

**GGUF-006**: String length fields MUST be validated before allocation.

**GGUF-007**: Tensor dimension multiplication MUST use checked arithmetic.

**GGUF-008**: Data type enums MUST be validated against known values.

**GGUF-009**: GGUF version field MUST be validated (2 or 3 only).

**GGUF-010**: Magic number MUST be validated (`0x46554747` = "GGUF").

**GGUF-011**: Parser MUST fail fast on first invalid field (no partial parsing).

**GGUF-012**: Parser MUST NOT allocate based on untrusted size fields.

### 2.2 Path Security

**PATH-001**: All filesystem paths MUST be validated via `input-validation` crate.

**PATH-002**: Paths MUST be canonicalized (resolve `..` and symlinks).

**PATH-003**: Canonicalized paths MUST be checked against allowed root directory.

**PATH-004**: Paths MUST NOT contain null bytes (`\0`).

**PATH-005**: Paths MUST NOT contain path traversal sequences (`../`, `..\\`).

**PATH-006**: Symlinks MUST be resolved and validated against allowed root.

**PATH-007**: Absolute paths outside allowed root MUST be rejected.

**PATH-008**: Path validation failures MUST NOT expose sensitive path components.

### 2.3 Hash Verification

**HASH-001**: SHA-256 MUST be used for digest computation (FIPS 140-2 approved).

**HASH-002**: Hash format MUST be validated (64 hex characters) before comparison.

**HASH-003**: Hash computation MUST occur before GGUF parsing (fail fast).

**HASH-004**: Hash mismatch MUST reject model load immediately.

**HASH-005**: Hash comparison MUST NOT be timing-safe (not a secret).

**HASH-006**: Computed hash MUST be logged for audit trail (even if not verified).

**HASH-007**: Hash validation MUST use `input-validation::validate_hex_string()`.

### 2.4 Resource Limits

**LIMIT-001**: File size MUST be checked before reading entire file.

**LIMIT-002**: Maximum file size MUST be configurable (default 100GB).

**LIMIT-003**: Tensor count MUST be validated against MAX_TENSORS.

**LIMIT-004**: String allocations MUST be validated against MAX_STRING_LEN.

**LIMIT-005**: Metadata pair count MUST be validated against MAX_METADATA_PAIRS.

**LIMIT-006**: Memory allocation MUST fail gracefully with actionable errors.

**LIMIT-007**: Parser MUST NOT enter infinite loops on malformed input.

### 2.5 Error Handling

**ERROR-001**: All operations MUST return `Result<T, LoadError>` (no panics).

**ERROR-002**: Error messages MUST NOT expose file contents or sensitive data.

**ERROR-003**: Error messages MUST include actionable diagnostics (offset, expected vs actual).

**ERROR-004**: Errors MUST distinguish retriable (Io) from fatal (HashMismatch, InvalidFormat).

**ERROR-005**: Path validation errors MUST NOT expose canonicalized paths.

**ERROR-006**: Parser errors MUST include offset and field name for debugging.

### 2.6 Memory Safety

**MEM-001**: All array access MUST be bounds-checked (no indexing without validation).

**MEM-002**: All arithmetic MUST use checked operations or saturating arithmetic.

**MEM-003**: Buffer reads MUST validate `offset + length <= buffer.len()`.

**MEM-004**: String allocations MUST validate length before `Vec::with_capacity()`.

**MEM-005**: Tensor dimension multiplication MUST use `checked_mul()`.

**MEM-006**: File size casts MUST validate against `usize::MAX`.

---

## 3. Vulnerability Analysis

### 3.1 GGUF Parser Buffer Overflow (CRITICAL)

**Vulnerability**: Parsing untrusted GGUF format without bounds checking.

**Attack Vector**:
```rust
// VULNERABLE CODE
fn read_string(&self, offset: usize) -> String {
    let len = u32::from_le_bytes([
        self.bytes[offset],     // ← No bounds check
        self.bytes[offset + 1], // ← Could overflow
        self.bytes[offset + 2],
        self.bytes[offset + 3],
    ]);
    
    let start = offset + 4;
    let end = start + len as usize; // ← Integer overflow possible
    
    String::from_utf8(self.bytes[start..end].to_vec()).unwrap() // ← Buffer overflow
}
```

**Malicious GGUF**:
```
Offset 0x00: Magic: 0x46554747 ("GGUF")
Offset 0x04: Version: 3
Offset 0x08: Tensor count: 1
Offset 0x10: Metadata KV count: 1
Offset 0x18: String length: 0xFFFFFFFF  ← Triggers overflow
```

**Impact**:
- Buffer overflow → read out of bounds
- Integer overflow → incorrect buffer allocation
- Remote code execution via heap corruption
- Denial of service via panic

**Mitigation** (GGUF-005, MEM-003):
```rust
fn read_u32(&self, offset: usize) -> Result<u32> {
    // Check bounds before reading
    let end = offset.checked_add(4)
        .ok_or(LoadError::BufferOverflow { 
            offset, 
            length: 4, 
            available: self.bytes.len() 
        })?;
    
    if end > self.bytes.len() {
        return Err(LoadError::BufferOverflow { 
            offset, 
            length: 4, 
            available: self.bytes.len() 
        });
    }
    
    // Safe to read now
    Ok(u32::from_le_bytes([
        self.bytes[offset],
        self.bytes[offset + 1],
        self.bytes[offset + 2],
        self.bytes[offset + 3],
    ]))
}

fn read_string(&self, offset: usize) -> Result<String> {
    let len = self.read_u32(offset)? as usize;
    
    // Validate string length
    if len > MAX_STRING_LEN {
        return Err(LoadError::StringTooLong { 
            length: len, 
            max: MAX_STRING_LEN 
        });
    }
    
    let start = offset.checked_add(4)
        .ok_or(LoadError::BufferOverflow { offset, length: 4, available: self.bytes.len() })?;
    
    let end = start.checked_add(len)
        .ok_or(LoadError::BufferOverflow { offset: start, length: len, available: self.bytes.len() })?;
    
    if end > self.bytes.len() {
        return Err(LoadError::BufferOverflow { 
            offset: start, 
            length: len, 
            available: self.bytes.len() 
        });
    }
    
    // Safe to read now
    String::from_utf8(self.bytes[start..end].to_vec())
        .map_err(|e| LoadError::InvalidFormat(format!("Invalid UTF-8: {}", e)))
}
```

**Status**: ⬜ Not yet implemented (M0 priority)

**References**: 
- SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #19
- CWE-119: Buffer Overflow
- CWE-190: Integer Overflow

---

### 3.2 Tensor Count Resource Exhaustion (HIGH)

**Vulnerability**: Allocating memory based on untrusted tensor_count field.

**Attack Vector**:
```rust
// VULNERABLE CODE
fn parse_gguf(&self) -> Result<Model> {
    let tensor_count = self.read_u64(8)? as usize; // ← Untrusted
    
    let mut tensors = Vec::with_capacity(tensor_count); // ← OOM attack
    
    for i in 0..tensor_count {
        tensors.push(self.read_tensor()?); // ← Could loop forever
    }
}
```

**Malicious GGUF**:
```
Tensor count: 0xFFFFFFFFFFFFFFFF (usize::MAX)
→ Attempts to allocate ~18 exabytes
→ OOM or integer overflow
```

**Impact**:
- Out-of-memory crash (DoS)
- Integer overflow in allocation size
- Infinite loop consuming CPU
- Worker becomes unresponsive

**Mitigation** (GGUF-001, LIMIT-003):
```rust
const MAX_TENSORS: usize = 10_000;

fn parse_gguf(&self) -> Result<Model> {
    let tensor_count = self.read_u64(8)?;
    
    // Validate before casting
    if tensor_count > MAX_TENSORS as u64 {
        return Err(LoadError::TensorCountExceeded { 
            count: tensor_count as usize, 
            max: MAX_TENSORS 
        });
    }
    
    let tensor_count = tensor_count as usize;
    
    // Safe to allocate now (bounded)
    let mut tensors = Vec::with_capacity(tensor_count);
    
    for i in 0..tensor_count {
        tensors.push(self.read_tensor()?);
    }
    
    Ok(Model { tensors })
}
```

**Status**: ⬜ Not yet implemented (M0 priority)

**References**:
- SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #19
- CWE-770: Allocation of Resources Without Limits
- CWE-400: Uncontrolled Resource Consumption

---

### 3.3 Path Traversal (HIGH)

**Vulnerability**: Loading arbitrary files via path parameter.

**Attack Vector**:
```rust
// VULNERABLE CODE
pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
    // No path validation
    let bytes = std::fs::read(request.model_path)?; // ← Arbitrary file read
    // ...
}
```

**Malicious Request**:
```rust
LoadRequest {
    model_path: &PathBuf::from("../../../../etc/passwd"),
    expected_hash: None,
    max_size: 100_000_000_000,
}
```

**Impact**:
- Read arbitrary files on worker filesystem
- Exfiltrate sensitive data (/etc/shadow, private keys)
- Information disclosure (system configuration)
- Bypass model staging controls

**Mitigation** (PATH-001 to PATH-008):
```rust
use input_validation::validate_path;

pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
    let allowed_root = PathBuf::from("/var/lib/llorch/models");
    
    // Validate path (canonicalize + containment check)
    let canonical_path = validate_path(request.model_path, &allowed_root)
        .map_err(|e| LoadError::PathValidationFailed(e.to_string()))?;
    
    // Now safe to read (guaranteed within allowed_root)
    let bytes = std::fs::read(&canonical_path)?;
    
    // Continue validation...
}
```

**Additional Protections**:
```rust
// Reject null bytes
if path_str.contains('\0') {
    return Err(LoadError::PathValidationFailed("Null byte in path".into()));
}

// Reject path traversal sequences
if path_str.contains("../") || path_str.contains("..\\") {
    return Err(LoadError::PathValidationFailed("Path traversal detected".into()));
}

// Canonicalize and check containment
let canonical = path.canonicalize()
    .map_err(|e| LoadError::Io(e))?;

if !canonical.starts_with(&allowed_root) {
    return Err(LoadError::PathValidationFailed(
        format!("Path outside allowed directory: {:?}", canonical)
    ));
}
```

**Status**: ⬜ Not yet implemented (M0 priority)

**References**:
- SECURITY_AUDIT_EXISTING_CODEBASE.md Vulnerability #9
- CWE-22: Path Traversal
- CWE-59: Improper Link Resolution

---

### 3.4 Hash Bypass (HIGH)

**Vulnerability**: Hash verification can be skipped or forged.

**Attack Vector 1** — Optional hash:
```rust
// VULNERABLE CODE
pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
    let bytes = std::fs::read(request.model_path)?;
    
    // Hash verification is optional
    if let Some(expected_hash) = request.expected_hash {
        self.verify_hash(&bytes, expected_hash)?;
    }
    
    // Attacker sets expected_hash = None → no verification
    Ok(bytes)
}
```

**Attack Vector 2** — Hash format not validated:
```rust
// VULNERABLE CODE
fn verify_hash(&self, bytes: &[u8], expected_hash: &str) -> Result<()> {
    let actual_hash = compute_sha256(bytes);
    
    // No format validation → attacker sends "abc" instead of 64 hex chars
    if actual_hash != expected_hash {
        return Err(LoadError::HashMismatch { expected, actual });
    }
    
    Ok(())
}
```

**Impact**:
- Load poisoned models without detection
- Bypass integrity verification
- Model poisoning attack succeeds
- Compromised pool-managerd can inject malicious models

**Mitigation** (HASH-001 to HASH-007):
```rust
use input_validation::validate_hex_string;
use sha2::{Sha256, Digest};

fn verify_hash(&self, bytes: &[u8], expected_hash: &str) -> Result<()> {
    // Validate hash format (64 hex chars)
    validate_hex_string(expected_hash, 64)
        .map_err(|e| LoadError::InvalidFormat(format!("Invalid hash format: {}", e)))?;
    
    // Compute actual hash
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let actual_hash = format!("{:x}", hasher.finalize());
    
    // Compare
    if actual_hash != expected_hash {
        return Err(LoadError::HashMismatch {
            expected: expected_hash.to_string(),
            actual: actual_hash,
        });
    }
    
    // Log for audit trail
    tracing::info!(
        hash = %actual_hash,
        "Model hash verified"
    );
    
    Ok(())
}
```

**Policy Decision**: Should hash verification be mandatory?
- **Option A**: Make expected_hash required (reject if None)
- **Option B**: Keep optional but log warning if skipped
- **Option C**: Require for production, optional for dev/test

**Recommendation**: Option A for M0 (fail closed)

**Status**: ⬜ Not yet implemented (M0 priority)

**References**:
- SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #9
- CWE-345: Insufficient Verification of Data Authenticity

---

### 3.5 Integer Overflow in Tensor Dimensions (MEDIUM)

**Vulnerability**: Tensor dimension multiplication without overflow checks.

**Attack Vector**:
```rust
// VULNERABLE CODE
fn parse_tensor(&self, offset: usize) -> Result<Tensor> {
    let ndim = self.read_u32(offset)? as usize;
    let mut dims = Vec::new();
    
    for i in 0..ndim {
        dims.push(self.read_u64(offset + 4 + i * 8)?);
    }
    
    // Calculate total elements
    let total_elements = dims.iter().product::<u64>(); // ← Overflow possible
    
    let bytes_per_element = 4; // f32
    let total_bytes = total_elements * bytes_per_element; // ← Overflow
    
    // Allocate based on overflowed value
    let mut data = vec![0u8; total_bytes as usize]; // ← Wrong size
}
```

**Malicious GGUF**:
```
Tensor dimensions: [2^32, 2^32, 2^32]
→ Product: 2^96 (overflows u64)
→ Wraps to small value
→ Allocates tiny buffer
→ Writes overflow buffer
```

**Impact**:
- Integer overflow → incorrect buffer size
- Buffer overflow during tensor data read
- Heap corruption
- Potential RCE

**Mitigation** (GGUF-007, MEM-005):
```rust
fn parse_tensor(&self, offset: usize) -> Result<Tensor> {
    let ndim = self.read_u32(offset)? as usize;
    
    if ndim > 8 {
        return Err(LoadError::InvalidFormat(
            format!("Too many dimensions: {}", ndim)
        ));
    }
    
    let mut dims = Vec::new();
    for i in 0..ndim {
        dims.push(self.read_u64(offset + 4 + i * 8)?);
    }
    
    // Calculate total elements with overflow checking
    let mut total_elements: u64 = 1;
    for &dim in &dims {
        total_elements = total_elements.checked_mul(dim)
            .ok_or(LoadError::InvalidFormat(
                format!("Tensor dimensions overflow: {:?}", dims)
            ))?;
    }
    
    // Check against reasonable limit
    const MAX_TENSOR_ELEMENTS: u64 = 1_000_000_000; // 1B elements
    if total_elements > MAX_TENSOR_ELEMENTS {
        return Err(LoadError::InvalidFormat(
            format!("Tensor too large: {} elements", total_elements)
        ));
    }
    
    // Safe to proceed
    Ok(Tensor { dims, total_elements })
}
```

**Status**: ⬜ Not yet implemented (M0 priority)

**References**:
- CWE-190: Integer Overflow
- CWE-680: Integer Overflow to Buffer Overflow

---

### 3.6 Symlink Attack (MEDIUM)

**Vulnerability**: Following symlinks to read arbitrary files.

**Attack Vector**:
```bash
# Attacker creates symlink in allowed directory
cd /var/lib/llorch/models
ln -s /etc/shadow malicious-model.gguf

# Worker follows symlink
LoadRequest {
    model_path: "/var/lib/llorch/models/malicious-model.gguf",
    // Points to /etc/shadow
}
```

**Impact**:
- Read arbitrary files via symlink
- Exfiltrate sensitive data
- Bypass path containment checks

**Mitigation** (PATH-006):
```rust
use std::fs;

pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
    let allowed_root = PathBuf::from("/var/lib/llorch/models");
    
    // Canonicalize resolves symlinks
    let canonical_path = request.model_path.canonicalize()
        .map_err(|e| LoadError::Io(e))?;
    
    // Check if resolved path is still within allowed root
    if !canonical_path.starts_with(&allowed_root) {
        return Err(LoadError::PathValidationFailed(
            format!("Symlink points outside allowed directory: {:?}", canonical_path)
        ));
    }
    
    // Additional check: reject if it's a symlink to sensitive file
    let metadata = fs::symlink_metadata(request.model_path)?;
    if metadata.is_symlink() {
        tracing::warn!(
            path = ?request.model_path,
            target = ?canonical_path,
            "Loading model via symlink"
        );
    }
    
    // Safe to read
    let bytes = fs::read(&canonical_path)?;
    Ok(bytes)
}
```

**Status**: ⬜ Not yet implemented (M0 priority)

**References**:
- CWE-59: Improper Link Resolution
- CWE-61: UNIX Symbolic Link Following

---

### 3.7 Information Leakage via Error Messages (LOW)

**Vulnerability**: Error messages expose sensitive file paths or contents.

**Attack Vector**:
```rust
// VULNERABLE CODE
Err(LoadError::Io(std::io::Error::new(
    std::io::ErrorKind::NotFound,
    format!("File not found: /var/lib/llorch/models/secret-model-v2-internal.gguf")
    // ← Leaks internal model naming
)))

Err(LoadError::InvalidFormat(
    format!("Invalid magic at offset 0: expected 0x46554747, got 0x{:x}", magic)
    // ← Leaks file contents
))
```

**Impact**:
- Information disclosure (file paths, directory structure)
- Enumeration of available models
- Fingerprinting of model versions
- Reconnaissance for further attacks

**Mitigation** (ERROR-002, ERROR-005):
```rust
// GOOD: Generic error without sensitive details
Err(LoadError::Io(e)) // Just wrap the error, don't format path

// GOOD: Include debugging info without sensitive data
Err(LoadError::InvalidFormat(
    format!("Invalid GGUF magic number at offset 0")
    // No actual bytes exposed
))

// GOOD: Path validation error without exposing canonical path
Err(LoadError::PathValidationFailed(
    "Path outside allowed directory".to_string()
    // Don't include the actual path
))
```

**Status**: ⬜ Not yet implemented (M0 priority)

**References**:
- CWE-209: Information Exposure Through Error Messages
- OWASP: Information Leakage

---

## 4. Attack Surface Summary

### 4.1 Input Attack Surface

**Untrusted Inputs**:
1. **model_path** (filesystem path)
   - Path traversal sequences
   - Symlinks
   - Null bytes
   - Absolute paths

2. **model_bytes** (binary data)
   - Malformed GGUF headers
   - Oversized fields
   - Invalid UTF-8 strings
   - Crafted tensor dimensions

3. **expected_hash** (hex string)
   - Invalid format
   - Wrong length
   - Null bytes

4. **max_size** (integer)
   - Overflow values
   - Zero or negative

### 4.2 Parser Attack Surface

**GGUF Format Parsing**:
- Magic number (4 bytes)
- Version field (4 bytes)
- Tensor count (8 bytes) ← **HIGH RISK**
- Metadata KV count (8 bytes) ← **HIGH RISK**
- String lengths (variable) ← **CRITICAL**
- Tensor dimensions (variable) ← **HIGH RISK**
- Data type enums (1 byte)
- Tensor data offsets (8 bytes)

**Risk Areas**:
- ⚠️ **CRITICAL**: String length fields (buffer overflow)
- ⚠️ **HIGH**: Tensor count (resource exhaustion)
- ⚠️ **HIGH**: Tensor dimensions (integer overflow)
- ⚠️ **MEDIUM**: Metadata pairs (allocation DoS)

### 4.3 Filesystem Attack Surface

**File Operations**:
- `std::fs::read()` — Arbitrary file read if path not validated
- `std::fs::metadata()` — File size check (can be bypassed via TOCTOU)
- `Path::canonicalize()` — Symlink resolution (must check result)

**Risk Areas**:
- ⚠️ **HIGH**: Path traversal via `../` sequences
- ⚠️ **MEDIUM**: Symlink attacks
- ⚠️ **LOW**: TOCTOU between size check and read

---

## 5. Security Controls

### 5.1 Input Validation

**Implemented via `input-validation` crate**:
```rust
use input_validation::{validate_path, validate_hex_string};

// Path validation
let canonical_path = validate_path(request.model_path, &allowed_root)?;

// Hash format validation
validate_hex_string(expected_hash, 64)?;
```

**Custom validation**:
```rust
// File size validation
if file_size > request.max_size {
    return Err(LoadError::TooLarge(file_size, request.max_size));
}

// GGUF magic number
if magic != 0x46554747 {
    return Err(LoadError::InvalidFormat(format!("Invalid magic: 0x{:x}", magic)));
}
```

### 5.2 Bounds Checking

**All reads must be bounds-checked**:
```rust
fn read_u32(&self, offset: usize) -> Result<u32> {
    let end = offset.checked_add(4)
        .ok_or(LoadError::BufferOverflow { offset, length: 4, available: self.bytes.len() })?;
    
    if end > self.bytes.len() {
        return Err(LoadError::BufferOverflow { offset, length: 4, available: self.bytes.len() });
    }
    
    Ok(u32::from_le_bytes([
        self.bytes[offset],
        self.bytes[offset + 1],
        self.bytes[offset + 2],
        self.bytes[offset + 3],
    ]))
}
```

### 5.3 Resource Limits

**Security limits enforced**:
```rust
const MAX_TENSORS: usize = 10_000;
const MAX_FILE_SIZE: usize = 100_000_000_000; // 100GB
const MAX_STRING_LEN: usize = 65536; // 64KB
const MAX_METADATA_PAIRS: usize = 1000;

// Validate before allocating
if tensor_count > MAX_TENSORS {
    return Err(LoadError::TensorCountExceeded { count: tensor_count, max: MAX_TENSORS });
}

if string_len > MAX_STRING_LEN {
    return Err(LoadError::StringTooLong { length: string_len, max: MAX_STRING_LEN });
}
```

### 5.4 Cryptographic Verification

**SHA-256 hash verification**:
```rust
use sha2::{Sha256, Digest};

fn verify_hash(&self, bytes: &[u8], expected_hash: &str) -> Result<()> {
    // Validate format
    validate_hex_string(expected_hash, 64)?;
    
    // Compute hash
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let actual_hash = format!("{:x}", hasher.finalize());
    
    // Compare
    if actual_hash != expected_hash {
        return Err(LoadError::HashMismatch { expected, actual });
    }
    
    Ok(())
}
```

---

## 6. Security Testing Requirements

### 6.1 Fuzz Testing

**Required fuzz targets**:
```rust
// fuzz/fuzz_targets/gguf_parser.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use model_loader::ModelLoader;

fuzz_target!(|data: &[u8]| {
    let loader = ModelLoader::new();
    let _ = loader.validate_bytes(data, None);
    // Should never panic, crash, or hang
});
```

**Fuzzing goals**:
- Find buffer overflows
- Find integer overflows
- Find infinite loops
- Find panics
- Find memory leaks

**Fuzzing duration**: Minimum 24 hours per target

### 6.2 Property Testing

**Required property tests**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn parser_never_panics(bytes: Vec<u8>) {
        let loader = ModelLoader::new();
        let _ = loader.validate_bytes(&bytes, None);
        // Should never panic
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

### 6.3 Security Test Cases

**Required test coverage**:

**Path Traversal Tests**:
```rust
#[test]
fn test_rejects_path_traversal() {
    let loader = ModelLoader::new();
    let result = loader.load_and_validate(LoadRequest {
        model_path: &PathBuf::from("../../../../etc/passwd"),
        expected_hash: None,
        max_size: 1000,
    });
    assert!(matches!(result, Err(LoadError::PathValidationFailed(_))));
}

#[test]
fn test_rejects_symlink_escape() {
    // Create symlink outside allowed directory
    let temp_dir = TempDir::new().unwrap();
    let symlink_path = temp_dir.path().join("evil.gguf");
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

**Buffer Overflow Tests**:
```rust
#[test]
fn test_rejects_oversized_string() {
    let mut gguf = create_valid_gguf_header();
    // Set string length to MAX + 1
    gguf.extend_from_slice(&(MAX_STRING_LEN as u32 + 1).to_le_bytes());
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&gguf, None);
    assert!(matches!(result, Err(LoadError::StringTooLong { .. })));
}

#[test]
fn test_rejects_buffer_overflow() {
    let mut gguf = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    gguf.extend_from_slice(&3u32.to_le_bytes()); // Version
    gguf.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
    gguf.extend_from_slice(&1u64.to_le_bytes()); // Metadata pairs
    // String length points beyond buffer
    gguf.extend_from_slice(&1000u32.to_le_bytes());
    // But buffer is only 100 bytes total
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&gguf, None);
    assert!(matches!(result, Err(LoadError::BufferOverflow { .. })));
}
```

**Resource Exhaustion Tests**:
```rust
#[test]
fn test_rejects_excessive_tensor_count() {
    let gguf = create_gguf_with_tensor_count(MAX_TENSORS + 1);
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(&gguf, None);
    assert!(matches!(result, Err(LoadError::TensorCountExceeded { .. })));
}

#[test]
fn test_rejects_oversized_file() {
    let loader = ModelLoader::new();
    let result = loader.load_and_validate(LoadRequest {
        model_path: &PathBuf::from("/dev/zero"),
        expected_hash: None,
        max_size: 1000,
    });
    assert!(matches!(result, Err(LoadError::TooLarge(_, _))));
}
```

**Hash Verification Tests**:
```rust
#[test]
fn test_rejects_hash_mismatch() {
    let model_bytes = b"test model";
    let wrong_hash = "0".repeat(64);
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(model_bytes, Some(&wrong_hash));
    assert!(matches!(result, Err(LoadError::HashMismatch { .. })));
}

#[test]
fn test_rejects_invalid_hash_format() {
    let model_bytes = b"test model";
    let invalid_hash = "not_hex";
    
    let loader = ModelLoader::new();
    let result = loader.validate_bytes(model_bytes, Some(invalid_hash));
    assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
}
```

---

## 7. Clippy Security Configuration

### 7.1 TIER 1 Lints (Enforced)

```rust
// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]           // No unwrap (must use ?)
#![deny(clippy::expect_used)]           // No expect (must use ?)
#![deny(clippy::panic)]                 // No panic (must return Result)
#![deny(clippy::indexing_slicing)]      // No [] indexing (use .get())
#![deny(clippy::integer_arithmetic)]    // Use checked arithmetic
#![deny(clippy::cast_ptr_alignment)]    // Safe pointer casts only
#![deny(clippy::mem_forget)]            // No mem::forget (resource leak)
#![deny(clippy::todo)]                  // No TODOs in production
#![deny(clippy::unimplemented)]         // No unimplemented!()
```

### 7.2 Additional Security Lints

```rust
#![warn(clippy::arithmetic_side_effects)]  // Warn on unchecked arithmetic
#![warn(clippy::cast_lossless)]            // Prefer lossless casts
#![warn(clippy::cast_possible_truncation)] // Warn on truncating casts
#![warn(clippy::cast_possible_wrap)]       // Warn on wrapping casts
#![warn(clippy::cast_precision_loss)]      // Warn on precision loss
#![warn(clippy::cast_sign_loss)]           // Warn on sign loss
#![warn(clippy::string_slice)]             // Warn on string slicing
#![warn(clippy::missing_errors_doc)]       // Document error cases
#![warn(clippy::missing_panics_doc)]       // Document panic cases
#![warn(clippy::missing_safety_doc)]       // Document unsafe code
```

---

## 8. Audit Requirements

### 8.1 Security Audit Events

**Required audit logging**:
```rust
// Model load started
tracing::info!(
    model_path = ?request.model_path,
    expected_hash = ?request.expected_hash,
    max_size = request.max_size,
    "Model load started"
);

// Hash verification
tracing::info!(
    hash = %computed_hash,
    verified = expected_hash.is_some(),
    "Model hash computed"
);

// Load failed
tracing::error!(
    error = %e,
    model_path = ?request.model_path,
    "Model load failed"
);

// Load succeeded
tracing::info!(
    model_path = ?request.model_path,
    file_size = bytes.len(),
    tensor_count = metadata.tensor_count,
    "Model load completed"
);
```

### 8.2 Security Incident Logging

**Critical events that require immediate attention**:
```rust
// Path traversal attempt
tracing::warn!(
    event = "security_incident",
    incident_type = "path_traversal",
    attempted_path = ?request.model_path,
    "Path traversal attempt detected"
);

// Buffer overflow attempt
tracing::error!(
    event = "security_incident",
    incident_type = "buffer_overflow",
    offset = offset,
    length = length,
    available = self.bytes.len(),
    "Buffer overflow attempt detected"
);

// Resource exhaustion attempt
tracing::warn!(
    event = "security_incident",
    incident_type = "resource_exhaustion",
    tensor_count = count,
    max_allowed = MAX_TENSORS,
    "Excessive tensor count detected"
);
```

---

## 9. Deployment Security

### 9.1 Filesystem Permissions

**Required permissions**:
```bash
# Model directory (read-only for worker)
/var/lib/llorch/models/
  Owner: root:root
  Mode: 0755 (drwxr-xr-x)

# Model files (read-only for worker)
/var/lib/llorch/models/*.gguf
  Owner: root:llorch
  Mode: 0640 (-rw-r-----)

# Worker process runs as non-root
User: llorch
Group: llorch
```

### 9.2 Process Isolation

**Recommended isolation**:
- Run worker-orcd as non-root user
- Use systemd sandboxing (PrivateTmp, ProtectHome, etc.)
- Consider containers for multi-tenancy
- Limit filesystem access via AppArmor/SELinux

---

## 10. Refinement Opportunities

### 10.1 Signature Verification

**Current**: Hash verification only (integrity)  
**Future**: Add cryptographic signature verification (authenticity)

**Implementation**:
```rust
pub struct LoadRequest<'a> {
    pub model_path: &'a Path,
    pub expected_hash: Option<&'a str>,
    pub signature: Option<&'a [u8]>,        // ← New
    pub public_key: Option<&'a PublicKey>,  // ← New
    pub max_size: usize,
}

fn verify_signature(&self, bytes: &[u8], sig: &[u8], pubkey: &PublicKey) -> Result<()> {
    use ed25519_dalek::Verifier;
    
    let signature = ed25519_dalek::Signature::from_bytes(sig)
        .map_err(|e| LoadError::SignatureVerificationFailed)?;
    
    pubkey.verify(bytes, &signature)
        .map_err(|e| LoadError::SignatureVerificationFailed)?;
    
    Ok(())
}
```

### 10.2 Streaming Validation

**Current**: Load entire file into memory  
**Future**: Stream validation for large files (> 10GB)

**Benefits**:
- Reduce memory footprint
- Faster failure on invalid files
- Support files larger than RAM

### 10.3 Multi-Format Support

**Current**: GGUF only  
**Future**: SafeTensors, PyTorch .bin, etc.

**Security considerations**:
- Each format needs separate parser
- Each parser needs separate fuzzing
- Shared validation framework

### 10.4 Incremental Hashing

**Current**: Hash entire file at once  
**Future**: Incremental hashing during streaming read

**Benefits**:
- Constant memory usage
- Early failure on hash mismatch
- Better performance for large files

---

## 11. References

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issue #9, #19
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #9

**Specifications**:
- `00_model-loader.md` — Functional specification
- `10_expectations.md` — Consumer expectations

**Standards**:
- CWE-119: Buffer Overflow
- CWE-190: Integer Overflow
- CWE-22: Path Traversal
- CWE-345: Insufficient Verification of Data Authenticity
- CWE-770: Allocation of Resources Without Limits
- OWASP: Input Validation Cheat Sheet

---

**End of Security Specification**
