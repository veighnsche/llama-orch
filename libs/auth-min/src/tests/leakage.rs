//! Token leakage detection tests
//!
//! These tests verify that token fingerprints don't leak sensitive information
//! and that the fingerprinting function has appropriate properties.

use crate::token_fp6;

#[test]
fn test_fingerprint_non_reversible() {
    // Verify we cannot recover the token from its fingerprint
    let token = "super-secret-token-abc123";
    let fp = token_fp6(token);
    
    // Fingerprint should be much shorter than token
    assert!(fp.len() < token.len());
    
    // Fingerprint should not contain token substring
    assert!(!fp.contains("super"));
    assert!(!fp.contains("secret"));
    assert!(!fp.contains("abc123"));
}

#[test]
fn test_fingerprint_collision_resistance() {
    // Generate fingerprints for many similar tokens
    let mut fingerprints = std::collections::HashSet::new();
    
    for i in 0..1000 {
        let token = format!("token-{}", i);
        let fp = token_fp6(&token);
        
        // Each fingerprint should be unique
        assert!(
            fingerprints.insert(fp.clone()),
            "Collision detected at token-{}: {}",
            i,
            fp
        );
    }
    
    // Should have 1000 unique fingerprints
    assert_eq!(fingerprints.len(), 1000);
}

#[test]
fn test_fingerprint_prefix_independence() {
    // Tokens with same prefix should have different fingerprints
    let tokens = vec![
        "prefix-a",
        "prefix-b",
        "prefix-c",
        "prefix-aa",
        "prefix-ab",
    ];
    
    let mut fingerprints = std::collections::HashSet::new();
    for token in &tokens {
        let fp = token_fp6(token);
        assert!(
            fingerprints.insert(fp.clone()),
            "Collision for token: {} (fp: {})",
            token,
            fp
        );
    }
}

#[test]
fn test_fingerprint_suffix_independence() {
    // Tokens with same suffix should have different fingerprints
    let tokens = vec![
        "a-suffix",
        "b-suffix",
        "c-suffix",
        "aa-suffix",
        "ab-suffix",
    ];
    
    let mut fingerprints = std::collections::HashSet::new();
    for token in &tokens {
        let fp = token_fp6(token);
        assert!(
            fingerprints.insert(fp.clone()),
            "Collision for token: {} (fp: {})",
            token,
            fp
        );
    }
}

#[test]
fn test_fingerprint_avalanche_effect() {
    // Small change in token should produce very different fingerprint
    let token1 = "secret-token-abc123";
    let token2 = "secret-token-abc124"; // Last char changed
    
    let fp1 = token_fp6(token1);
    let fp2 = token_fp6(token2);
    
    // Fingerprints should be completely different
    assert_ne!(fp1, fp2);
    
    // Count differing characters (should be high due to avalanche effect)
    let diff_count = fp1
        .chars()
        .zip(fp2.chars())
        .filter(|(a, b)| a != b)
        .count();
    
    // At least half the characters should differ (avalanche property)
    assert!(
        diff_count >= 3,
        "Insufficient avalanche effect: only {} chars differ (fp1: {}, fp2: {})",
        diff_count,
        fp1,
        fp2
    );
}

#[test]
fn test_fingerprint_no_common_patterns() {
    // Verify fingerprints don't reveal patterns in tokens
    let tokens = vec![
        "aaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbb",
        "cccccccccccccccc",
        "0000000000000000",
        "1111111111111111",
    ];
    
    let mut fingerprints = Vec::new();
    for token in &tokens {
        let fp = token_fp6(token);
        fingerprints.push(fp.clone());
        
        // Fingerprint should not be all same character
        let first_char = fp.chars().next().unwrap();
        assert!(
            !fp.chars().all(|c| c == first_char),
            "Fingerprint has pattern: {} for token: {}",
            fp,
            token
        );
    }
    
    // All fingerprints should be unique
    let unique: std::collections::HashSet<_> = fingerprints.iter().collect();
    assert_eq!(unique.len(), tokens.len());
}

#[test]
fn test_fingerprint_safe_for_logging() {
    // Verify fingerprints are safe to include in logs
    let sensitive_token = "super-secret-api-key-do-not-leak-12345";
    let fp = token_fp6(sensitive_token);
    
    // Fingerprint should be short
    assert_eq!(fp.len(), 6);
    
    // Fingerprint should be alphanumeric (safe for logs)
    assert!(fp.chars().all(|c| c.is_ascii_alphanumeric()));
    
    // Fingerprint should not contain sensitive parts
    assert!(!fp.contains("secret"));
    assert!(!fp.contains("api"));
    assert!(!fp.contains("key"));
    assert!(!fp.contains("12345"));
    
    // Simulated log message should be safe
    let log_msg = format!("Authentication successful: token:{}", fp);
    assert!(!log_msg.contains("super-secret"));
    assert!(!log_msg.contains("api-key"));
}
