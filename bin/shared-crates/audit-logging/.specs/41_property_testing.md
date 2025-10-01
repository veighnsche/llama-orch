# Audit Logging — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐⭐ HIGH  
**Applies to**: `bin/shared-crates/audit-logging/`

---

## 0. Why Property Testing for Audit Logging?

**Audit logs are the security safety net**. Property-based testing ensures:

- ✅ Hash chains are **never broken**
- ✅ HMAC integrity is **always maintained**
- ✅ Timestamps are **monotonically increasing**
- ✅ No events are **lost** under concurrent writes
- ✅ Checksums **always verify**
- ✅ Log rotation doesn't **corrupt** data

**Broken audit logs = no forensic evidence**

---

## 1. Critical Properties to Test

### 1.1 Hash Chain Integrity

**Property**: Hash chain is never broken, each event links to previous.

```rust
use proptest::prelude::*;

proptest! {
    /// Hash chain maintains integrity
    #[test]
    fn hash_chain_integrity(events in prop::collection::vec(any_audit_event(), 1..100)) {
        let mut logger = AuditLogger::new()?;
        let mut prev_hash = None;
        
        for event in events {
            let logged = logger.log_event(event)?;
            
            if let Some(expected_prev) = prev_hash {
                prop_assert_eq!(logged.previous_hash, expected_prev);
            }
            
            prev_hash = Some(logged.current_hash.clone());
        }
        
        // Verify entire chain
        let chain_valid = logger.verify_chain()?;
        prop_assert!(chain_valid);
    }
    
    /// Tampering breaks hash chain
    #[test]
    fn tampering_detected(events in prop::collection::vec(any_audit_event(), 2..100)) {
        let mut logger = AuditLogger::new()?;
        
        for event in events {
            logger.log_event(event)?;
        }
        
        // Tamper with middle event
        let logs = logger.get_all_events()?;
        if logs.len() > 1 {
            let tampered_index = logs.len() / 2;
            // Modify event data
            // ... tampering logic ...
            
            // Verification should fail
            let chain_valid = logger.verify_chain()?;
            prop_assert!(!chain_valid);
        }
    }
}

// Helper to generate random audit events
fn any_audit_event() -> impl Strategy<Value = AuditEvent> {
    (
        any::<u64>(), // timestamp
        "[a-zA-Z0-9_-]{1,50}", // actor
        "[a-zA-Z0-9_-]{1,50}", // action
        "\\PC{0,500}", // details
    ).prop_map(|(ts, actor, action, details)| {
        AuditEvent {
            timestamp: ts,
            actor: ActorInfo { id: actor },
            action: action,
            resource: ResourceInfo { details },
        }
    })
}
```

---

### 1.2 HMAC Verification

**Property**: HMAC always verifies for unmodified events.

```rust
proptest! {
    /// HMAC verification is deterministic
    #[test]
    fn hmac_deterministic(
        event_data in prop::collection::vec(any::<u8>(), 0..1000),
        key in prop::collection::vec(any::<u8>(), 32..32)
    ) {
        let hmac1 = compute_hmac(&event_data, &key)?;
        let hmac2 = compute_hmac(&event_data, &key)?;
        
        prop_assert_eq!(hmac1, hmac2);
    }
    
    /// Modified data fails HMAC verification
    #[test]
    fn hmac_detects_modification(
        event_data in prop::collection::vec(any::<u8>(), 1..1000),
        key in prop::collection::vec(any::<u8>(), 32..32),
        flip_bit_index in 0usize..1000
    ) {
        let original_hmac = compute_hmac(&event_data, &key)?;
        
        // Flip one bit
        let mut modified_data = event_data.clone();
        if flip_bit_index < modified_data.len() {
            modified_data[flip_bit_index] ^= 0x01;
            
            let modified_hmac = compute_hmac(&modified_data, &key)?;
            
            // HMACs should differ
            prop_assert_ne!(original_hmac, modified_hmac);
        }
    }
}
```

---

### 1.3 Timestamp Monotonicity

**Property**: Timestamps always increase (or stay same for concurrent events).

```rust
proptest! {
    /// Timestamps are monotonically increasing
    #[test]
    fn timestamps_monotonic(events in prop::collection::vec(any_audit_event(), 1..100)) {
        let mut logger = AuditLogger::new()?;
        let mut last_timestamp = 0u64;
        
        for event in events {
            let logged = logger.log_event(event)?;
            
            // Timestamp should not decrease
            prop_assert!(logged.timestamp >= last_timestamp);
            last_timestamp = logged.timestamp;
        }
    }
    
    /// Clock skew is handled
    #[test]
    fn clock_skew_handling(
        timestamps in prop::collection::vec(any::<u64>(), 1..100)
    ) {
        let mut logger = AuditLogger::new()?;
        
        for ts in timestamps {
            let event = AuditEvent::with_timestamp(ts);
            let logged = logger.log_event(event)?;
            
            // Logger should enforce monotonicity even with clock skew
            // (either reject or adjust timestamp)
            prop_assert!(logged.timestamp >= 0);
        }
    }
}
```

---

### 1.4 No Event Loss Under Concurrency

**Property**: All events are logged, even under concurrent writes.

```rust
use std::sync::Arc;
use std::thread;

proptest! {
    /// No events lost under concurrent writes
    #[test]
    fn concurrent_writes_safe(
        event_count in 10usize..100,
        thread_count in 2usize..10
    ) {
        let logger = Arc::new(AuditLogger::new()?);
        let mut handles = vec![];
        
        for thread_id in 0..thread_count {
            let logger_clone = Arc::clone(&logger);
            let events_per_thread = event_count / thread_count;
            
            let handle = thread::spawn(move || {
                for i in 0..events_per_thread {
                    let event = AuditEvent::new(
                        format!("thread-{}", thread_id),
                        format!("action-{}", i),
                    );
                    logger_clone.log_event(event).unwrap();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all events were logged
        let logged_count = logger.count_events()?;
        prop_assert_eq!(logged_count, event_count);
    }
}
```

---

### 1.5 Checksum Verification

**Property**: File checksums always match content.

```rust
proptest! {
    /// Checksums always verify
    #[test]
    fn checksum_verification(events in prop::collection::vec(any_audit_event(), 1..100)) {
        let temp_dir = tempfile::tempdir()?;
        let log_file = temp_dir.path().join("audit.log");
        
        let mut logger = AuditLogger::new_with_file(&log_file)?;
        
        for event in events {
            logger.log_event(event)?;
        }
        
        logger.flush()?;
        
        // Compute checksum
        let checksum = compute_file_checksum(&log_file)?;
        
        // Verify checksum
        let verified = verify_file_checksum(&log_file, &checksum)?;
        prop_assert!(verified);
    }
    
    /// Modified files fail checksum
    #[test]
    fn checksum_detects_tampering(events in prop::collection::vec(any_audit_event(), 1..100)) {
        let temp_dir = tempfile::tempdir()?;
        let log_file = temp_dir.path().join("audit.log");
        
        let mut logger = AuditLogger::new_with_file(&log_file)?;
        
        for event in events {
            logger.log_event(event)?;
        }
        
        logger.flush()?;
        let checksum = compute_file_checksum(&log_file)?;
        
        // Tamper with file
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new().append(true).open(&log_file)?;
        file.write_all(b"TAMPERED")?;
        
        // Verification should fail
        let verified = verify_file_checksum(&log_file, &checksum)?;
        prop_assert!(!verified);
    }
}
```

---

### 1.6 Log Rotation Preserves Data

**Property**: Log rotation never loses or corrupts events.

```rust
proptest! {
    /// Log rotation preserves all events
    #[test]
    fn rotation_preserves_events(events in prop::collection::vec(any_audit_event(), 100..1000)) {
        let temp_dir = tempfile::tempdir()?;
        let mut logger = AuditLogger::new_with_rotation(
            temp_dir.path(),
            1024, // Rotate after 1KB
        )?;
        
        let mut expected_events = vec![];
        
        for event in events {
            logger.log_event(event.clone())?;
            expected_events.push(event);
        }
        
        // Read all events from all rotated files
        let all_logged = logger.read_all_events()?;
        
        prop_assert_eq!(all_logged.len(), expected_events.len());
        
        // Verify hash chain across rotations
        let chain_valid = logger.verify_chain_all_files()?;
        prop_assert!(chain_valid);
    }
}
```

---

## 2. Specific Event Types to Test

### 2.1 Authentication Events

```rust
proptest! {
    /// Authentication events are always logged
    #[test]
    fn auth_events_logged(
        user_id in "[a-zA-Z0-9_-]{1,50}",
        success in any::<bool>()
    ) {
        let mut logger = AuditLogger::new()?;
        
        let event = AuditEvent::authentication(user_id, success);
        let logged = logger.log_event(event)?;
        
        prop_assert!(logged.event_type == EventType::Authentication);
        prop_assert_eq!(logged.success, success);
    }
}
```

---

### 2.2 Authorization Events

```rust
proptest! {
    /// Authorization decisions are logged
    #[test]
    fn authz_events_logged(
        user_id in "[a-zA-Z0-9_-]{1,50}",
        resource in "[a-zA-Z0-9_-]{1,50}",
        action in "[a-zA-Z0-9_-]{1,50}",
        granted in any::<bool>()
    ) {
        let mut logger = AuditLogger::new()?;
        
        let event = AuditEvent::authorization(user_id, resource, action, granted);
        let logged = logger.log_event(event)?;
        
        prop_assert!(logged.event_type == EventType::Authorization);
        prop_assert_eq!(logged.granted, granted);
    }
}
```

---

### 2.3 Data Access Events

```rust
proptest! {
    /// Data access is always audited
    #[test]
    fn data_access_logged(
        user_id in "[a-zA-Z0-9_-]{1,50}",
        resource_id in "[a-zA-Z0-9_-]{1,50}",
        operation in "(read|write|delete)"
    ) {
        let mut logger = AuditLogger::new()?;
        
        let event = AuditEvent::data_access(user_id, resource_id, operation);
        let logged = logger.log_event(event)?;
        
        prop_assert!(logged.event_type == EventType::DataAccess);
    }
}
```

---

## 3. Query Properties

### 3.1 Query Results are Consistent

```rust
proptest! {
    /// Queries return consistent results
    #[test]
    fn query_consistency(events in prop::collection::vec(any_audit_event(), 1..100)) {
        let mut logger = AuditLogger::new()?;
        
        for event in events {
            logger.log_event(event)?;
        }
        
        // Query twice
        let results1 = logger.query_events(QueryFilter::all())?;
        let results2 = logger.query_events(QueryFilter::all())?;
        
        prop_assert_eq!(results1.len(), results2.len());
        
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            prop_assert_eq!(r1.event_id, r2.event_id);
        }
    }
}
```

---

## 4. Performance Properties

### 4.1 Logging Performance

```rust
use std::time::{Duration, Instant};

proptest! {
    /// Logging completes in reasonable time
    #[test]
    fn logging_performance(event in any_audit_event()) {
        let mut logger = AuditLogger::new()?;
        
        let start = Instant::now();
        logger.log_event(event)?;
        let elapsed = start.elapsed();
        
        // Should complete in < 10ms
        prop_assert!(elapsed < Duration::from_millis(10));
    }
    
    /// Batch logging is efficient
    #[test]
    fn batch_logging_efficient(events in prop::collection::vec(any_audit_event(), 100..1000)) {
        let mut logger = AuditLogger::new()?;
        
        let start = Instant::now();
        for event in events.iter() {
            logger.log_event(event.clone())?;
        }
        let elapsed = start.elapsed();
        
        let events_per_sec = events.len() as f64 / elapsed.as_secs_f64();
        
        // Should handle > 1000 events/sec
        prop_assert!(events_per_sec > 1000.0);
    }
}
```

---

## 5. Error Handling Properties

### 5.1 Disk Full Handling

```rust
proptest! {
    /// Handles disk full gracefully
    #[test]
    fn disk_full_handling(events in prop::collection::vec(any_audit_event(), 1..100)) {
        // Create small temp filesystem (if possible on platform)
        // Log until disk full
        // Verify: no data corruption, clear error, recovery possible
    }
}
```

---

## 6. Implementation Guide

### 6.1 Add Proptest Dependency

```toml
# Cargo.toml
[dev-dependencies]
proptest.workspace = true
tempfile = "3.8"
```

### 6.2 Test Structure

```rust
//! Property-based tests for audit logging
//!
//! These tests verify:
//! - Hash chain integrity
//! - HMAC verification
//! - Timestamp monotonicity
//! - No event loss under concurrency
//! - Checksum verification

use proptest::prelude::*;
use audit_logging::*;

proptest! {
    // Core properties
}

#[cfg(test)]
mod hash_chain {
    use super::*;
    proptest! {
        // Hash chain tests
    }
}

#[cfg(test)]
mod concurrency {
    use super::*;
    proptest! {
        // Concurrent access tests
    }
}

#[cfg(test)]
mod integrity {
    use super::*;
    proptest! {
        // Integrity verification tests
    }
}
```

---

## 7. Running Tests

```bash
# Run all tests
cargo test -p audit-logging

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p audit-logging --test property_tests

# Test with thread sanitizer
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p audit-logging

# Stress test
PROPTEST_CASES=100000 cargo test -p audit-logging -- --test-threads=1
```

---

## 8. Audit Trail Validation Checklist

Before production:
- [ ] Hash chain never breaks (verified by property tests)
- [ ] HMAC always verifies (verified by property tests)
- [ ] Timestamps monotonic (verified by property tests)
- [ ] No event loss under concurrency (verified by stress tests)
- [ ] Checksums always match (verified by property tests)
- [ ] Log rotation preserves data (verified by property tests)
- [ ] Tampering is detected (verified by property tests)
- [ ] Performance acceptable (> 1000 events/sec)

---

## 9. Common Pitfalls

### ❌ Don't Do This

```rust
// BAD: Testing with only sequential events
proptest! {
    #[test]
    fn bad_test(count in 1usize..10) {
        for i in 0..count {
            log_event(i);
        }
    }
}
```

### ✅ Do This Instead

```rust
// GOOD: Testing with concurrent events
proptest! {
    #[test]
    fn good_test(events in prop::collection::vec(any_audit_event(), 1..100)) {
        // Test with random events, including concurrent writes
        test_concurrent_logging(events);
    }
}
```

---

## 10. Refinement Opportunities

### 10.1 Advanced Testing

**Future work**:
- Test with filesystem failures
- Test with network storage
- Test with encryption at rest
- Test with log aggregation

### 10.2 Compliance Testing

**Future work**:
- Verify GDPR compliance (data retention)
- Verify SOC 2 requirements (audit trail)
- Verify HIPAA compliance (access logs)

---

## 11. References

- **Audit Logging Best Practices**: https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- **Hash Chains**: https://en.wikipedia.org/wiki/Hash_chain
- **HMAC**: https://tools.ietf.org/html/rfc2104

---

**Priority**: ⭐⭐⭐⭐ HIGH  
**Estimated Effort**: 2-3 days  
**Impact**: Ensures forensic evidence integrity  
**Status**: Recommended for immediate implementation
