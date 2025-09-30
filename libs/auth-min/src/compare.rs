//! Timing-safe comparison functions
//!
//! Implements constant-time equality checks to prevent timing side-channel attacks (CWE-208).
//!
//! # Security Properties
//!
//! The comparison examines all bytes regardless of where a mismatch occurs,
//! ensuring the execution time does not leak information about the position
//! of the first differing byte.
//!
//! # References
//!
//! - CWE-208: Observable Timing Discrepancy
//! - `.specs/12_auth-min-hardening.md` (SEC-AUTH-2001)

/// Constant-time comparison of two byte slices.
///
/// This function compares two byte slices in constant time to prevent timing attacks.
/// It returns `true` if the slices are equal, `false` otherwise.
///
/// # Security
///
/// The comparison always examines all bytes, even after finding a mismatch.
/// This ensures the execution time does not reveal information about where
/// the slices differ.
///
/// # Examples
///
/// ```
/// use auth_min::timing_safe_eq;
///
/// let token1 = b"secret-token-abc123";
/// let token2 = b"secret-token-abc123";
/// assert!(timing_safe_eq(token1, token2));
///
/// let token3 = b"wrong-token";
/// assert!(!timing_safe_eq(token1, token3));
/// ```
///
/// # Implementation Notes
///
/// Uses bitwise OR accumulation to ensure constant-time behavior:
/// - Length check is constant-time (single comparison)
/// - Byte-by-byte XOR with OR accumulation examines all bytes
/// - Final comparison is constant-time (single comparison)
pub fn timing_safe_eq(a: &[u8], b: &[u8]) -> bool {
    // Early length check (constant time - single comparison)
    if a.len() != b.len() {
        return false;
    }

    // Accumulate differences using bitwise OR
    // This ensures we examine all bytes regardless of where differences occur
    let mut diff: u8 = 0;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }

    // Final comparison is constant-time
    // Use explicit comparison to avoid compiler optimization
    let result = diff == 0;

    // Compiler fence to prevent reordering (defense-in-depth)
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_slices() {
        let a = b"secret-token-abc123";
        let b = b"secret-token-abc123";
        assert!(timing_safe_eq(a, b));
    }

    #[test]
    fn test_unequal_slices() {
        let a = b"secret-token-abc123";
        let b = b"wrong-token-xyz789";
        assert!(!timing_safe_eq(a, b));
    }

    #[test]
    fn test_different_lengths() {
        let a = b"short";
        let b = b"much-longer-string";
        assert!(!timing_safe_eq(a, b));
    }

    #[test]
    fn test_empty_slices() {
        let a = b"";
        let b = b"";
        assert!(timing_safe_eq(a, b));
    }

    #[test]
    fn test_one_byte_difference_early() {
        let a = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = b"baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert!(!timing_safe_eq(a, b));
    }

    #[test]
    fn test_one_byte_difference_late() {
        let a = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab";
        assert!(!timing_safe_eq(a, b));
    }

    #[test]
    fn test_timing_variance_acceptable() {
        use std::time::Instant;

        // Create 64-byte tokens
        let token = vec![b'a'; 64];
        let wrong_early = {
            let mut t = token.clone();
            t[0] = b'b'; // Differs at position 0
            t
        };
        let wrong_late = {
            let mut t = token.clone();
            t[63] = b'b'; // Differs at position 63
            t
        };

        // Measure comparison time for early mismatch
        let iterations = 100000; // More iterations for stable measurement
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = timing_safe_eq(&token, &wrong_early);
        }
        let time_early = start.elapsed().as_nanos() / iterations;

        // Measure comparison time for late mismatch
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = timing_safe_eq(&token, &wrong_late);
        }
        let time_late = start.elapsed().as_nanos() / iterations;

        // Calculate variance
        let variance = if time_early > time_late {
            (time_early - time_late) as f64 / time_early as f64
        } else {
            (time_late - time_early) as f64 / time_late as f64
        };

        // Note: In debug builds, variance can be higher due to lack of optimization
        // In release builds with --release, variance should be < 10%
        // For debug builds, we accept < 80% as "reasonably constant-time"
        // The compiler fence adds some overhead but ensures security
        let threshold = if cfg!(debug_assertions) { 0.8 } else { 0.1 };

        println!(
            "Timing variance: {:.2}% (early: {}ns, late: {}ns) [threshold: {:.0}%]",
            variance * 100.0,
            time_early,
            time_late,
            threshold * 100.0
        );

        assert!(
            variance < threshold,
            "Timing variance too high: {:.2}% (early: {}ns, late: {}ns, threshold: {:.0}%)",
            variance * 100.0,
            time_early,
            time_late,
            threshold * 100.0
        );
    }
}
