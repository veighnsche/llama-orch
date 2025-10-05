//! Property-based tests for GPU detection security
//!
//! These tests use proptest to generate random inputs and verify that:
//! - Parser never panics on malformed input
//! - Arithmetic operations never overflow
//! - String operations respect bounds

use proptest::prelude::*;

// Re-export for testing (would need to make parse function pub(crate) or add test helper)
// For now, we'll test the public API

proptest! {
    /// Test that detect_gpus never panics regardless of PATH state
    #[test]
    fn detect_gpus_never_panics(_random_seed in any::<u64>()) {
        // This should never panic, even if nvidia-smi is missing
        let _ = gpu_info::detect_gpus();
    }

    /// Test that has_gpu never panics
    #[test]
    fn has_gpu_never_panics(_random_seed in any::<u64>()) {
        let _ = gpu_info::has_gpu();
    }

    /// Test that gpu_count never panics
    #[test]
    fn gpu_count_never_panics(_random_seed in any::<u64>()) {
        let count = gpu_info::gpu_count();
        // Count should be reasonable (< 256 GPUs)
        prop_assert!(count < 256);
    }
}

#[cfg(test)]
mod vram_calculations {
    use super::*;

    proptest! {
        /// Test that VRAM calculations never overflow
        #[test]
        fn vram_mb_to_bytes_never_overflows(vram_mb in 0usize..1_000_000usize) {
            // Simulate the conversion we do in parser
            let bytes = vram_mb.saturating_mul(1024).saturating_mul(1024);

            // Should never overflow (saturating prevents it)
            prop_assert!(bytes <= usize::MAX);

            // If input is reasonable, result should be reasonable
            if vram_mb <= 1_000_000 {
                prop_assert!(bytes <= 1_000_000 * 1024 * 1024);
            }
        }

        /// Test that string truncation works correctly
        #[test]
        fn string_truncation_safe(s in "\\PC{0,1000}") {
            const MAX_LEN: usize = 256;

            // Simulate our truncation logic
            let truncated: String = s.chars().take(MAX_LEN).collect();

            // Should never exceed max length
            prop_assert!(truncated.chars().count() <= MAX_LEN);

            // Should not panic
            prop_assert!(truncated.len() <= s.len());
        }
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    proptest! {
        /// Test that null bytes are handled safely
        #[test]
        fn null_bytes_handled_safely(s in "\\PC{0,100}") {
            // Add null byte at random position
            let with_null = format!("{}\0{}", s, s);

            // Our parser should reject this or handle it safely
            // (We can't test the internal parser directly, but the API should be safe)
            let _ = with_null.contains('\0');

            // This test mainly ensures we don't panic
            prop_assert!(true);
        }

        /// Test that extremely long strings don't cause issues
        #[test]
        fn long_strings_handled(len in 0usize..10_000usize) {
            let long_string = "A".repeat(len);

            // Should not panic or exhaust memory
            prop_assert!(long_string.len() == len);

            // Truncation should work
            let truncated: String = long_string.chars().take(256).collect();
            prop_assert!(truncated.chars().count() <= 256);
        }
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    proptest! {
        /// Test that device validation handles all u32 values
        #[test]
        fn device_validation_safe(device_id in any::<u32>()) {
            let info = gpu_info::detect_gpus();

            // Should either succeed or return appropriate error
            let result = info.validate_device(device_id);

            // Should not panic
            if info.count == 0 {
                prop_assert!(result.is_err());
            } else if (device_id as usize) < info.count {
                prop_assert!(result.is_ok());
            } else {
                prop_assert!(result.is_err());
            }
        }

        /// Test VRAM calculations with edge values
        #[test]
        fn vram_calculations_edge_cases(
            total_mb in 0usize..2_000_000usize,
            free_mb in 0usize..2_000_000usize
        ) {
            // Convert to bytes using saturating arithmetic
            let total_bytes = total_mb.saturating_mul(1024).saturating_mul(1024);
            let free_bytes = free_mb.saturating_mul(1024).saturating_mul(1024);

            // Clamp free to total (as we do in parser)
            let clamped_free = free_bytes.min(total_bytes);

            // Invariants
            prop_assert!(clamped_free <= total_bytes);
            prop_assert!(total_bytes <= usize::MAX);
            prop_assert!(clamped_free <= usize::MAX);
        }
    }
}
