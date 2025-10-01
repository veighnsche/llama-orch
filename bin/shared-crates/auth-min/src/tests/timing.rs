//! Timing attack resistance tests
//!
//! These tests verify that timing_safe_eq exhibits constant-time behavior
//! by measuring execution time variance for different input patterns.

use crate::timing_safe_eq;
use std::time::Instant;

/// Measure average execution time for a comparison operation
fn measure_comparison_time(a: &[u8], b: &[u8], iterations: usize) -> u128 {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = timing_safe_eq(a, b);
    }
    start.elapsed().as_nanos() / iterations as u128
}

#[test]
fn test_timing_variance_64_byte_tokens() {
    // Test with realistic token sizes (64 bytes)
    let iterations = 100_000;

    let correct = vec![b'a'; 64];
    let wrong_early = {
        let mut t = correct.clone();
        t[0] = b'b'; // Differs at position 0
        t
    };
    let wrong_middle = {
        let mut t = correct.clone();
        t[32] = b'b'; // Differs at position 32
        t
    };
    let wrong_late = {
        let mut t = correct.clone();
        t[63] = b'b'; // Differs at position 63
        t
    };

    let time_early = measure_comparison_time(&correct, &wrong_early, iterations);
    let time_middle = measure_comparison_time(&correct, &wrong_middle, iterations);
    let time_late = measure_comparison_time(&correct, &wrong_late, iterations);

    // Calculate max variance
    let times = [time_early, time_middle, time_late];
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();
    let variance = (max_time - min_time) as f64 / max_time as f64;

    println!(
        "Timing variance: {:.2}% (early: {}ns, middle: {}ns, late: {}ns)",
        variance * 100.0,
        time_early,
        time_middle,
        time_late
    );

    // Note: In debug builds, variance can be higher due to lack of optimization
    // In release builds with --release, variance should be < 10%
    // For debug builds, we accept < 80% as "reasonably constant-time"
    // The compiler fence adds some overhead but ensures security
    let threshold = if cfg!(debug_assertions) { 0.8 } else { 0.1 };

    assert!(
        variance < threshold,
        "Timing variance too high: {:.2}% (threshold: {:.0}%)",
        variance * 100.0,
        threshold * 100.0
    );
}

#[test]
fn test_timing_variance_32_byte_tokens() {
    // Test with shorter tokens (32 bytes)
    let iterations = 100_000;

    let correct = vec![b'x'; 32];
    let wrong_early = {
        let mut t = correct.clone();
        t[0] = b'y';
        t
    };
    let wrong_late = {
        let mut t = correct.clone();
        t[31] = b'y';
        t
    };

    let time_early = measure_comparison_time(&correct, &wrong_early, iterations);
    let time_late = measure_comparison_time(&correct, &wrong_late, iterations);

    let variance = if time_early > time_late {
        (time_early - time_late) as f64 / time_early as f64
    } else {
        (time_late - time_early) as f64 / time_late as f64
    };

    println!(
        "32-byte timing variance: {:.2}% (early: {}ns, late: {}ns)",
        variance * 100.0,
        time_early,
        time_late
    );

    let threshold = if cfg!(debug_assertions) { 0.8 } else { 0.1 };
    assert!(
        variance < threshold,
        "Timing variance too high for 32-byte tokens: {:.2}% (threshold: {:.0}%)",
        variance * 100.0,
        threshold * 100.0
    );
}

#[test]
fn test_timing_variance_128_byte_tokens() {
    // Test with longer tokens (128 bytes)
    let iterations = 50_000;

    let correct = vec![b'z'; 128];
    let wrong_early = {
        let mut t = correct.clone();
        t[0] = b'a';
        t
    };
    let wrong_late = {
        let mut t = correct.clone();
        t[127] = b'a';
        t
    };

    let time_early = measure_comparison_time(&correct, &wrong_early, iterations);
    let time_late = measure_comparison_time(&correct, &wrong_late, iterations);

    let variance = if time_early > time_late {
        (time_early - time_late) as f64 / time_early as f64
    } else {
        (time_late - time_early) as f64 / time_late as f64
    };

    println!(
        "128-byte timing variance: {:.2}% (early: {}ns, late: {}ns)",
        variance * 100.0,
        time_early,
        time_late
    );

    let threshold = if cfg!(debug_assertions) { 0.8 } else { 0.1 };
    assert!(
        variance < threshold,
        "Timing variance too high for 128-byte tokens: {:.2}% (threshold: {:.0}%)",
        variance * 100.0,
        threshold * 100.0
    );
}

#[test]
fn test_equal_tokens_timing() {
    // Verify that comparing equal tokens doesn't leak timing
    let iterations = 100_000;

    let token1 = vec![b'a'; 64];
    let token2 = vec![b'a'; 64];
    let token3 = vec![b'b'; 64];
    let token4 = vec![b'b'; 64];

    let time_a = measure_comparison_time(&token1, &token2, iterations);
    let time_b = measure_comparison_time(&token3, &token4, iterations);

    let variance = if time_a > time_b {
        (time_a - time_b) as f64 / time_a as f64
    } else {
        (time_b - time_a) as f64 / time_b as f64
    };

    println!(
        "Equal tokens timing variance: {:.2}% (a: {}ns, b: {}ns)",
        variance * 100.0,
        time_a,
        time_b
    );

    // Even equal comparisons should have consistent timing
    let threshold = if cfg!(debug_assertions) { 0.8 } else { 0.1 };
    assert!(
        variance < threshold,
        "Timing variance for equal tokens: {:.2}% (threshold: {:.0}%)",
        variance * 100.0,
        threshold * 100.0
    );
}
