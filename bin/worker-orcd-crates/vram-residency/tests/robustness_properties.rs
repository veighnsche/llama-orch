//! Property-Based Robustness Tests
//!
//! Uses proptest to verify properties hold across many random inputs.
//!
//! Note: These tests run 256 cases per property by default.
//! Expected duration: 5-15 seconds total for all property tests.
//!
//! To skip these tests: cargo test --features skip-long-tests

use vram_residency::{VramManager, compute_digest, compute_signature, verify_signature, SealedShard};
use proptest::prelude::*;

// Print progress for long-running property tests
fn print_property_test_start(name: &str) {
    println!("\n⏱️  Running property test: {}", name);
    println!("   Testing 256 random cases...");
    println!("   Expected duration: 1-2 seconds");
}

// Property: Sealing and verifying always succeeds for valid data
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_seal_verify_roundtrip(data in prop::collection::vec(any::<u8>(), 1..10000)) {
        let mut manager = VramManager::new();
        
        // Seal the data
        let seal_result = manager.seal_model(&data, 0);
        
        // If seal succeeds, verify should also succeed
        if let Ok(shard) = seal_result {
            let verify_result = manager.verify_sealed(&shard);
            prop_assert!(verify_result.is_ok(), "Verify should succeed after seal");
        }
    }
}

// Property: Different data produces different digests
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_different_data_different_digests(
        data1 in prop::collection::vec(any::<u8>(), 1..1000),
        data2 in prop::collection::vec(any::<u8>(), 1..1000)
    ) {
        // Skip if data is identical
        prop_assume!(data1 != data2);
        
        let digest1 = compute_digest(&data1);
        let digest2 = compute_digest(&data2);
        
        prop_assert_ne!(digest1, digest2, "Different data should produce different digests");
    }
}

// Property: Same data produces same digest (deterministic)
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_digest_deterministic(data in prop::collection::vec(any::<u8>(), 1..1000)) {
        let digest1 = compute_digest(&data);
        let digest2 = compute_digest(&data);
        
        prop_assert_eq!(digest1, digest2, "Digest should be deterministic");
    }
}

// Property: Digest is always 64 hex characters
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_digest_format(data in prop::collection::vec(any::<u8>(), 0..10000)) {
        let digest = compute_digest(&data);
        
        prop_assert_eq!(digest.len(), 64, "Digest should be 64 characters");
        prop_assert!(digest.chars().all(|c| c.is_ascii_hexdigit()), "Digest should be hex");
    }
}

// Property: Same data + same key produces same signature
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_signature_deterministic(
        shard_id in "[a-zA-Z0-9_-]{1,256}",
        vram_bytes in 1usize..1000000usize,
        key in prop::collection::vec(any::<u8>(), 32..64)
    ) {
        let digest = "a".repeat(64);
        let shard = SealedShard::new_for_test(
            shard_id,
            0,
            vram_bytes,
            digest,
            0x1000,
        );
        
        let sig1 = compute_signature(&shard, &key);
        let sig2 = compute_signature(&shard, &key);
        
        prop_assert_eq!(sig1.as_ref().ok(), sig2.as_ref().ok(), "Signature should be deterministic");
    }
}

// Property: Different keys produce different signatures
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_different_keys_different_signatures(
        key1 in prop::collection::vec(any::<u8>(), 32..64),
        key2 in prop::collection::vec(any::<u8>(), 32..64)
    ) {
        prop_assume!(key1 != key2);
        
        let shard = SealedShard::new_for_test(
            "test-shard".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        
        let sig1 = compute_signature(&shard, &key1).ok();
        let sig2 = compute_signature(&shard, &key2).ok();
        
        if let (Some(s1), Some(s2)) = (sig1, sig2) {
            prop_assert_ne!(s1, s2, "Different keys should produce different signatures");
        }
    }
}

// Property: Valid signature always verifies
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_valid_signature_verifies(
        shard_id in "[a-zA-Z0-9_-]{1,256}",
        vram_bytes in 1usize..1000000usize,
        key in prop::collection::vec(any::<u8>(), 32..64)
    ) {
        let digest = "a".repeat(64);
        let shard = SealedShard::new_for_test(
            shard_id,
            0,
            vram_bytes,
            digest,
            0x1000,
        );
        
        if let Ok(signature) = compute_signature(&shard, &key) {
            let verify_result = verify_signature(&shard, &signature, &key);
            prop_assert!(verify_result.is_ok(), "Valid signature should verify");
        }
    }
}

// Property: Invalid signature never verifies
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_invalid_signature_fails(
        shard_id in "[a-zA-Z0-9_-]{1,256}",
        vram_bytes in 1usize..1000000usize,
        key in prop::collection::vec(any::<u8>(), 32..64),
        invalid_sig in prop::collection::vec(any::<u8>(), 32..=32)
    ) {
        let digest = "a".repeat(64);
        let shard = SealedShard::new_for_test(
            shard_id,
            0,
            vram_bytes,
            digest,
            0x1000,
        );
        
        // Compute valid signature
        if let Ok(valid_sig) = compute_signature(&shard, &key) {
            // Skip if invalid_sig happens to match valid_sig
            prop_assume!(invalid_sig != valid_sig);
            
            let verify_result = verify_signature(&shard, &invalid_sig, &key);
            prop_assert!(verify_result.is_err(), "Invalid signature should fail verification");
        }
    }
}

// Property: Signature is always 32 bytes
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_signature_length(
        shard_id in "[a-zA-Z0-9_-]{1,256}",
        vram_bytes in 1usize..1000000usize,
        key in prop::collection::vec(any::<u8>(), 32..64)
    ) {
        let digest = "a".repeat(64);
        let shard = SealedShard::new_for_test(
            shard_id,
            0,
            vram_bytes,
            digest,
            0x1000,
        );
        
        if let Ok(signature) = compute_signature(&shard, &key) {
            prop_assert_eq!(signature.len(), 32, "Signature should be 32 bytes (HMAC-SHA256)");
        }
    }
}

// Property: Seal generates unique shard IDs
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_unique_shard_ids(
        data1 in prop::collection::vec(any::<u8>(), 1..1000),
        data2 in prop::collection::vec(any::<u8>(), 1..1000)
    ) {
        let mut manager = VramManager::new();
        
        if let (Ok(shard1), Ok(shard2)) = (
            manager.seal_model(&data1, 0),
            manager.seal_model(&data2, 0)
        ) {
            prop_assert_ne!(shard1.shard_id, shard2.shard_id, "Shard IDs should be unique");
        }
    }
}

// Property: Seal preserves data size
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_seal_preserves_size(data in prop::collection::vec(any::<u8>(), 1..10000)) {
        let mut manager = VramManager::new();
        let size = data.len();
        
        if let Ok(shard) = manager.seal_model(&data, 0) {
            prop_assert_eq!(shard.vram_bytes, size, "Seal should preserve data size");
        }
    }
}

// Property: Tampered shard_id fails verification
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_tampered_shard_id_fails(
        original_id in "[a-zA-Z0-9_-]{1,256}",
        tampered_id in "[a-zA-Z0-9_-]{1,256}",
        key in prop::collection::vec(any::<u8>(), 32..64)
    ) {
        prop_assume!(original_id != tampered_id);
        
        let digest = "a".repeat(64);
        let shard = SealedShard::new_for_test(
            original_id.clone(),
            0,
            1024,
            digest.clone(),
            0x1000,
        );
        
        if let Ok(signature) = compute_signature(&shard, &key) {
            // Tamper with shard_id
            let mut tampered_shard = SealedShard::new_for_test(
                tampered_id,
                0,
                1024,
                digest,
                0x1000,
            );
            tampered_shard.set_signature_for_test(signature.clone());
            
            let verify_result = verify_signature(&tampered_shard, &signature, &key);
            prop_assert!(verify_result.is_err(), "Tampered shard_id should fail verification");
        }
    }
}

// Property: Digest changes with single bit flip
proptest! {
    #[test]
    #[cfg_attr(feature = "skip-long-tests", ignore)]
    fn prop_digest_avalanche_effect(
        mut data in prop::collection::vec(any::<u8>(), 1..1000),
        byte_idx in 0usize..1000usize,
        bit_idx in 0u8..8u8
    ) {
        prop_assume!(byte_idx < data.len());
        
        let digest1 = compute_digest(&data);
        
        // Flip one bit
        data[byte_idx] ^= 1 << bit_idx;
        
        let digest2 = compute_digest(&data);
        
        prop_assert_ne!(digest1, digest2, "Single bit flip should change digest");
    }
}
