# Audit Logging — Storage & Tamper Evidence Specification

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies storage requirements, tamper-evidence mechanisms, and integrity verification for audit logs. Audit logs MUST be immutable, append-only, and tamper-evident to meet compliance requirements (GDPR, SOC2, ISO 27001).

---

## 1. Storage Requirements

### 1.1 Immutability Principle

**Audit logs MUST be append-only**:
- ✅ New events can be appended
- ❌ Existing events CANNOT be modified
- ❌ Existing events CANNOT be deleted (except by retention policy)
- ✅ Tampering MUST be detectable

**Rationale**: Compliance regulations require immutable audit trails for forensic investigation and legal evidence.

---

### 1.2 Local Storage Mode

**Purpose**: Single-node deployments store audit logs locally.

**Storage Location**:
```
/var/lib/llorch/audit/{service-name}/
  ├─ 2025-10-01.audit           # Daily log file
  ├─ 2025-10-01.audit.sha256    # Checksum file
  ├─ 2025-10-02.audit
  ├─ 2025-10-02.audit.sha256
  └─ manifest.json              # File index
```

**File Format**: Newline-delimited JSON (NDJSON)
```
{"audit_id":"audit-001","timestamp":"2025-10-01T10:00:00Z",...}\n
{"audit_id":"audit-002","timestamp":"2025-10-01T10:00:01Z",...}\n
{"audit_id":"audit-003","timestamp":"2025-10-01T10:00:02Z",...}\n
```

**Rotation Policy**:
- Daily rotation at midnight UTC
- Rotate when file exceeds 100MB
- Keep rotated files for retention period

---

### 1.3 Platform Storage Mode

**Purpose**: Platform/marketplace deployments send audit logs to central service.

**Architecture**:
```
Provider's service
         ↓
    audit-logging crate
         ↓
    audit-client (HTTP)
         ↓
Platform Audit Service (audit.llama-orch-platform.com)
         ↓
    Immutable Storage (S3 Object Lock, TimescaleDB)
```

**Storage Backend**:
- **S3 with Object Lock** (WORM mode) for immutability
- **Glacier** for long-term archival (7+ years)
- **TimescaleDB** for queryable time-series data
- **Cross-region replication** for disaster recovery

**Encryption**:
- **At rest**: AES-256 (AWS KMS)
- **In transit**: TLS 1.3

---

### 1.4 Manifest File

**Purpose**: Index of all audit log files for integrity verification.

**Location**: `/var/lib/llorch/audit/{service-name}/manifest.json`

**Format**:
```json
{
  "service_id": "rbees-orcd",
  "created_at": "2025-10-01T00:00:00Z",
  "files": [
    {
      "filename": "2025-10-01.audit",
      "created_at": "2025-10-01T00:00:00Z",
      "closed_at": "2025-10-02T00:00:00Z",
      "event_count": 15234,
      "first_audit_id": "audit-2025-1001-000000-abc123",
      "last_audit_id": "audit-2025-1001-235959-xyz789",
      "sha256": "abc123def456...",
      "size_bytes": 52428800
    },
    {
      "filename": "2025-10-02.audit",
      "created_at": "2025-10-02T00:00:00Z",
      "closed_at": null,
      "event_count": 8421,
      "first_audit_id": "audit-2025-1002-000000-aaa111",
      "last_audit_id": "audit-2025-1002-143022-bbb222",
      "sha256": null,
      "size_bytes": 28672000
    }
  ]
}
```

**Update Policy**:
- Append new file entry when rotation occurs
- Update `event_count` and `last_audit_id` on flush
- Compute `sha256` when file is closed (rotated)
- Never modify entries for closed files

---

## 2. Tamper Evidence Mechanisms

### 2.1 Hash Chain (Blockchain-Style)

**Purpose**: Detect modification or deletion of events.

**Mechanism**: Each event includes hash of previous event.

**Event Envelope**:
```rust
pub struct AuditEventEnvelope {
    pub audit_id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub service_id: String,
    pub event: AuditEvent,
    pub prev_hash: String,  // SHA-256 of previous event
    pub hash: String,       // SHA-256 of this event
}
```

**Hash Computation**:
```rust
use sha2::{Sha256, Digest};

fn compute_event_hash(envelope: &AuditEventEnvelope) -> String {
    let mut hasher = Sha256::new();
    
    // Hash canonical representation (excluding hash field itself)
    hasher.update(envelope.audit_id.as_bytes());
    hasher.update(envelope.timestamp.to_rfc3339().as_bytes());
    hasher.update(envelope.event_type.as_bytes());
    hasher.update(envelope.service_id.as_bytes());
    hasher.update(serde_json::to_string(&envelope.event).unwrap().as_bytes());
    hasher.update(envelope.prev_hash.as_bytes());
    
    format!("{:x}", hasher.finalize())
}
```

**First Event**: `prev_hash` is `"0000000000000000000000000000000000000000000000000000000000000000"`

**Example Chain**:
```
Event 1: prev_hash=0000..., hash=abc123...
Event 2: prev_hash=abc123..., hash=def456...
Event 3: prev_hash=def456..., hash=ghi789...
```

**Tampering Detection**:
- If Event 2 is modified, its hash changes
- Event 3's `prev_hash` no longer matches Event 2's hash
- Chain is broken → tampering detected

---

### 2.2 File Checksums

**Purpose**: Detect modification of entire log files.

**Mechanism**: SHA-256 checksum of entire file.

**Checksum File**: `2025-10-01.audit.sha256`
```
abc123def456789012345678901234567890123456789012345678901234  2025-10-01.audit
```

**Computation**:
```bash
sha256sum 2025-10-01.audit > 2025-10-01.audit.sha256
```

**Verification**:
```bash
sha256sum -c 2025-10-01.audit.sha256
```

**When to Compute**:
- When file is rotated (closed)
- Before archival
- On demand for verification

---

### 2.3 Event Signatures (Platform Mode)

**Purpose**: Prove events originated from legitimate provider.

**Mechanism**: Provider signs each event with private key.

**Signature Algorithm**: HMAC-SHA256 or Ed25519

**HMAC-SHA256 Example**:
```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

fn sign_event(envelope: &AuditEventEnvelope, key: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(key).unwrap();
    
    // Sign canonical representation
    mac.update(envelope.audit_id.as_bytes());
    mac.update(envelope.timestamp.to_rfc3339().as_bytes());
    mac.update(envelope.event_type.as_bytes());
    mac.update(serde_json::to_string(&envelope.event).unwrap().as_bytes());
    
    let result = mac.finalize();
    hex::encode(result.into_bytes())
}

fn verify_signature(envelope: &AuditEventEnvelope, signature: &str, key: &[u8]) -> bool {
    let expected = sign_event(envelope, key);
    constant_time_eq(expected.as_bytes(), signature.as_bytes())
}
```

**Ed25519 Example** (more secure, asymmetric):
```rust
use ed25519_dalek::{Keypair, Signature, Signer, Verifier};

fn sign_event_ed25519(envelope: &AuditEventEnvelope, keypair: &Keypair) -> String {
    let message = format!(
        "{}|{}|{}",
        envelope.audit_id,
        envelope.timestamp.to_rfc3339(),
        serde_json::to_string(&envelope.event).unwrap()
    );
    
    let signature = keypair.sign(message.as_bytes());
    hex::encode(signature.to_bytes())
}

fn verify_signature_ed25519(
    envelope: &AuditEventEnvelope,
    signature: &str,
    public_key: &ed25519_dalek::PublicKey
) -> bool {
    let message = format!(
        "{}|{}|{}",
        envelope.audit_id,
        envelope.timestamp.to_rfc3339(),
        serde_json::to_string(&envelope.event).unwrap()
    );
    
    let sig_bytes = hex::decode(signature).unwrap();
    let signature = Signature::from_bytes(&sig_bytes).unwrap();
    
    public_key.verify(message.as_bytes(), &signature).is_ok()
}
```

**Platform Verification**:
1. Platform receives signed event from provider
2. Looks up provider's public key
3. Verifies signature matches event data
4. Rejects event if signature invalid

---

## 3. Integrity Verification

### 3.1 Hash Chain Verification

**Purpose**: Verify no events were modified or deleted.

**Algorithm**:
```rust
pub fn verify_hash_chain(events: &[AuditEventEnvelope]) -> Result<(), AuditError> {
    if events.is_empty() {
        return Ok(());
    }
    
    // First event must have zero prev_hash
    if events[0].prev_hash != "0000000000000000000000000000000000000000000000000000000000000000" {
        return Err(AuditError::InvalidChain("First event prev_hash must be zero".into()));
    }
    
    // Verify each event's hash
    for (i, event) in events.iter().enumerate() {
        let computed_hash = compute_event_hash(event);
        if computed_hash != event.hash {
            return Err(AuditError::InvalidChain(
                format!("Event {} hash mismatch", event.audit_id)
            ));
        }
        
        // Verify chain link
        if i > 0 {
            let prev_event = &events[i - 1];
            if event.prev_hash != prev_event.hash {
                return Err(AuditError::BrokenChain(
                    format!("Chain broken at event {}", event.audit_id)
                ));
            }
        }
    }
    
    Ok(())
}
```

**When to Verify**:
- On service startup (verify last N events)
- Periodically (every hour)
- On demand (admin query)
- Before archival

---

### 3.2 File Checksum Verification

**Purpose**: Verify log files haven't been modified.

**Algorithm**:
```rust
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::Read;

pub fn verify_file_checksum(file_path: &Path, expected_hash: &str) -> Result<(), AuditError> {
    let mut file = File::open(file_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    
    let computed_hash = format!("{:x}", hasher.finalize());
    
    if computed_hash != expected_hash {
        return Err(AuditError::ChecksumMismatch {
            file: file_path.display().to_string(),
            expected: expected_hash.to_string(),
            actual: computed_hash,
        });
    }
    
    Ok(())
}
```

---

### 3.3 Signature Verification (Platform Mode)

**Purpose**: Verify events originated from legitimate provider.

**Algorithm**:
```rust
pub fn verify_event_signature(
    envelope: &AuditEventEnvelope,
    provider_public_key: &ed25519_dalek::PublicKey
) -> Result<(), AuditError> {
    let signature = envelope.signature.as_ref()
        .ok_or(AuditError::MissingSignature)?;
    
    if !verify_signature_ed25519(envelope, signature, provider_public_key) {
        return Err(AuditError::InvalidSignature {
            audit_id: envelope.audit_id.clone(),
        });
    }
    
    Ok(())
}
```

---

## 4. Storage Implementation

### 4.1 Append-Only File Writer

**Purpose**: Write events to file with append-only guarantees.

**Implementation**:
```rust
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

pub struct AuditFileWriter {
    file_path: PathBuf,
    file: std::fs::File,
    event_count: usize,
    last_hash: String,
}

impl AuditFileWriter {
    pub fn new(file_path: PathBuf) -> Result<Self, AuditError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)  // Append-only
            .open(&file_path)?;
        
        Ok(Self {
            file_path,
            file,
            event_count: 0,
            last_hash: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
        })
    }
    
    pub fn write_event(&mut self, mut envelope: AuditEventEnvelope) -> Result<(), AuditError> {
        // Set prev_hash to last event's hash
        envelope.prev_hash = self.last_hash.clone();
        
        // Compute this event's hash
        envelope.hash = compute_event_hash(&envelope);
        
        // Serialize to JSON
        let json = serde_json::to_string(&envelope)?;
        
        // Write with newline
        writeln!(self.file, "{}", json)?;
        
        // Flush to disk (durability)
        self.file.sync_all()?;
        
        // Update state
        self.last_hash = envelope.hash.clone();
        self.event_count += 1;
        
        Ok(())
    }
    
    pub fn close(self) -> Result<(), AuditError> {
        // Final flush
        self.file.sync_all()?;
        
        // Compute file checksum
        let checksum = compute_file_checksum(&self.file_path)?;
        
        // Write checksum file
        let checksum_path = self.file_path.with_extension("audit.sha256");
        std::fs::write(&checksum_path, format!("{} {}\n", checksum, self.file_path.display()))?;
        
        Ok(())
    }
}
```

---

### 4.2 File Rotation

**Purpose**: Rotate log files daily or when size limit reached.

**Implementation**:
```rust
pub struct AuditLogger {
    base_dir: PathBuf,
    service_id: String,
    current_writer: Option<AuditFileWriter>,
    rotation_policy: RotationPolicy,
}

pub enum RotationPolicy {
    Daily,
    SizeLimit(usize),  // Bytes
    Both { daily: bool, size_limit: usize },
}

impl AuditLogger {
    pub fn should_rotate(&self) -> bool {
        match &self.rotation_policy {
            RotationPolicy::Daily => {
                // Check if date changed
                let current_date = Utc::now().format("%Y-%m-%d").to_string();
                let file_date = self.current_file_date();
                current_date != file_date
            }
            RotationPolicy::SizeLimit(limit) => {
                // Check file size
                self.current_file_size() >= *limit
            }
            RotationPolicy::Both { daily, size_limit } => {
                let date_changed = if *daily {
                    let current_date = Utc::now().format("%Y-%m-%d").to_string();
                    current_date != self.current_file_date()
                } else {
                    false
                };
                let size_exceeded = self.current_file_size() >= *size_limit;
                date_changed || size_exceeded
            }
        }
    }
    
    pub fn rotate(&mut self) -> Result<(), AuditError> {
        // Close current file
        if let Some(writer) = self.current_writer.take() {
            writer.close()?;
        }
        
        // Create new file
        let new_file_path = self.base_dir.join(format!(
            "{}.audit",
            Utc::now().format("%Y-%m-%d")
        ));
        
        self.current_writer = Some(AuditFileWriter::new(new_file_path)?);
        
        Ok(())
    }
}
```

---

## 5. Retention Policy

### 5.1 Retention Requirements

| Regulation | Minimum Retention | Recommended |
|------------|-------------------|-------------|
| **GDPR** | 1 year | 3 years |
| **SOC2** | 7 years | 7 years |
| **ISO 27001** | 3 years | 5 years |
| **HIPAA** | 6 years | 6 years |

**Default**: 7 years (SOC2 requirement)

---

### 5.2 Retention Implementation

**Policy**:
```rust
pub struct RetentionPolicy {
    pub min_retention_days: u32,  // 2555 days = 7 years
    pub archive_after_days: u32,  // 90 days
    pub delete_after_days: u32,   // 2555 days
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            min_retention_days: 2555,  // 7 years
            archive_after_days: 90,
            delete_after_days: 2555,
        }
    }
}
```

**Archival**:
```rust
pub fn archive_old_files(base_dir: &Path, policy: &RetentionPolicy) -> Result<(), AuditError> {
    let archive_threshold = Utc::now() - Duration::days(policy.archive_after_days as i64);
    
    for entry in std::fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension() == Some(std::ffi::OsStr::new("audit")) {
            let metadata = entry.metadata()?;
            let modified = metadata.modified()?;
            let modified_datetime: DateTime<Utc> = modified.into();
            
            if modified_datetime < archive_threshold {
                // Move to archive directory or compress
                archive_file(&path)?;
            }
        }
    }
    
    Ok(())
}
```

**Deletion** (after retention period):
```rust
pub fn delete_expired_files(base_dir: &Path, policy: &RetentionPolicy) -> Result<(), AuditError> {
    let delete_threshold = Utc::now() - Duration::days(policy.delete_after_days as i64);
    
    for entry in std::fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension() == Some(std::ffi::OsStr::new("audit")) {
            let metadata = entry.metadata()?;
            let modified = metadata.modified()?;
            let modified_datetime: DateTime<Utc> = modified.into();
            
            if modified_datetime < delete_threshold {
                // Verify retention period met
                if (Utc::now() - modified_datetime).num_days() >= policy.min_retention_days as i64 {
                    // Audit the deletion itself
                    audit_file_deletion(&path)?;
                    
                    // Delete file
                    std::fs::remove_file(&path)?;
                    std::fs::remove_file(path.with_extension("audit.sha256"))?;
                }
            }
        }
    }
    
    Ok(())
}
```

---

## 6. Error Handling

### 6.1 Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AuditError {
    #[error("Invalid hash chain: {0}")]
    InvalidChain(String),
    
    #[error("Broken hash chain at event {0}")]
    BrokenChain(String),
    
    #[error("Checksum mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    
    #[error("Missing signature for event {0}")]
    MissingSignature,
    
    #[error("Invalid signature for event {audit_id}")]
    InvalidSignature {
        audit_id: String,
    },
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
```

---

## 7. Refinement Opportunities

### 7.1 Immediate Improvements

1. **Implement atomic file writes** using temp file + rename pattern
2. **Add file locking** to prevent concurrent writes
3. **Implement buffered writes** for performance (flush every 1 second or 100 events)
4. **Add compression** for archived files (gzip, zstd)

### 7.2 Medium-Term Enhancements

5. **Implement incremental verification** (verify last N events on startup)
6. **Add Merkle tree** for efficient partial verification
7. **Implement write-ahead log** for durability
8. **Add backup/restore** functionality

### 7.3 Long-Term Vision

9. **Distributed hash chain** across multiple nodes
10. **Blockchain integration** for ultimate tamper evidence
11. **Zero-knowledge proofs** for privacy-preserving verification
12. **Hardware security module** (HSM) integration for key management

---

**End of Storage & Tamper Evidence Specification**
