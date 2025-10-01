//! Integer range validation applet
//!
//! Validates integer parameters to prevent overflow.

use crate::error::{Result, ValidationError};
use std::fmt::Display;

/// Validate integer is within range
///
/// # Rules
/// - Inclusive lower bound, exclusive upper bound: `min <= value < max`
/// - No overflow or wraparound
///
/// # Arguments
/// * `value` - Value to validate
/// * `min` - Minimum allowed value (inclusive)
/// * `max` - Maximum allowed value (exclusive)
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_range;
///
/// // Valid
/// assert!(validate_range(2, 0, 4).is_ok());
/// assert!(validate_range(1024, 1, 4096).is_ok());
///
/// // Invalid
/// assert!(validate_range(5, 0, 4).is_err());
/// assert!(validate_range(usize::MAX, 0, 100).is_err());
/// ```
///
/// # Errors
/// * `ValidationError::OutOfRange` - Value not in [min, max)
///
/// # Security
/// Prevents:
/// - Integer overflow: `max_tokens: usize::MAX`
/// - Invalid GPU index: `gpu_device: 999`
pub fn validate_range<T: PartialOrd + Display>(
    value: T,
    min: T,
    max: T,
) -> Result<()> {
    // Check if value is within range: min <= value < max
    // This is a simple comparison-only check with no arithmetic,
    // preventing any possibility of integer overflow or wraparound
    //
    // Security considerations:
    // 1. No arithmetic operations (no overflow risk)
    // 2. PartialOrd ensures type-safe comparisons
    // 3. Works with any comparable type (integers, floats, etc.)
    // 4. Inclusive lower bound, exclusive upper bound (standard range semantics)
    if value < min || value >= max {
        return Err(ValidationError::OutOfRange {
            value: value.to_string(),
            min: min.to_string(),
            max: max.to_string(),
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_ranges() {
        // Integers
        assert!(validate_range(2, 0, 4).is_ok());
        assert!(validate_range(0, 0, 4).is_ok());  // Min inclusive
        assert!(validate_range(3, 0, 4).is_ok());  // Max - 1
        
        // Unsigned
        assert!(validate_range(50u32, 0u32, 100u32).is_ok());
        
        // Floats
        assert!(validate_range(0.5, 0.0, 1.0).is_ok());
    }
    
    #[test]
    fn test_out_of_range_rejected() {
        // Below min
        assert!(matches!(
            validate_range(-1, 0, 4),
            Err(ValidationError::OutOfRange { .. })
        ));
        
        // At max (exclusive)
        assert!(matches!(
            validate_range(4, 0, 4),
            Err(ValidationError::OutOfRange { .. })
        ));
        
        // Above max
        assert!(matches!(
            validate_range(5, 0, 4),
            Err(ValidationError::OutOfRange { .. })
        ));
    }
    
    #[test]
    fn test_overflow_prevented() {
        assert!(matches!(
            validate_range(usize::MAX, 0, 100),
            Err(ValidationError::OutOfRange { .. })
        ));
        
        assert!(matches!(
            validate_range(u32::MAX, 0u32, 100u32),
            Err(ValidationError::OutOfRange { .. })
        ));
    }
    
    #[test]
    fn test_boundary_values() {
        // Min is inclusive
        assert!(validate_range(0, 0, 10).is_ok());
        
        // Max is exclusive
        assert!(validate_range(10, 0, 10).is_err());
        assert!(validate_range(9, 0, 10).is_ok());
        
        // One below minimum
        assert!(validate_range(-1, 0, 10).is_err());
    }
    
    #[test]
    fn test_negative_ranges() {
        // Negative range
        assert!(validate_range(-5, -10, 0).is_ok());
        
        // Range crossing zero
        assert!(validate_range(0, -10, 10).is_ok());
        
        // Below negative minimum
        assert!(validate_range(-11, -10, 0).is_err());
    }
    
    #[test]
    fn test_i64_extremes() {
        // i64::MAX out of range
        assert!(matches!(
            validate_range(i64::MAX, 0i64, 100i64),
            Err(ValidationError::OutOfRange { .. })
        ));
        
        // i64::MIN out of range
        assert!(matches!(
            validate_range(i64::MIN, 0i64, 100i64),
            Err(ValidationError::OutOfRange { .. })
        ));
    }
    
    #[test]
    fn test_floating_point_ranges() {
        // Float in range
        assert!(validate_range(0.5, 0.0, 1.0).is_ok());
        
        // Float at minimum (inclusive)
        assert!(validate_range(0.0, 0.0, 1.0).is_ok());
        
        // Float at maximum (exclusive)
        assert!(validate_range(1.0, 0.0, 1.0).is_err());
        
        // Fractional range
        assert!(validate_range(0.25, 0.1, 0.5).is_ok());
    }
    
    // ========== ROBUSTNESS TESTS ==========
    
    #[test]
    fn test_robustness_single_value_range() {
        // Range with single valid value [0, 1)
        assert!(validate_range(0, 0, 1).is_ok());
        assert!(validate_range(1, 0, 1).is_err());
        
        // Range with single valid value [100, 101)
        assert!(validate_range(100, 100, 101).is_ok());
        assert!(validate_range(101, 100, 101).is_err());
    }
    
    #[test]
    fn test_robustness_zero_width_range() {
        // Range where min == max (no valid values)
        assert!(validate_range(0, 0, 0).is_err());
        assert!(validate_range(5, 5, 5).is_err());
        assert!(validate_range(100, 100, 100).is_err());
    }
    
    #[test]
    fn test_robustness_inverted_range() {
        // Range where min > max (invalid range specification)
        // The function will reject all values
        assert!(validate_range(5, 10, 0).is_err());
        assert!(validate_range(0, 10, 0).is_err());
        assert!(validate_range(10, 10, 0).is_err());
    }
    
    #[test]
    fn test_robustness_unsigned_overflow_attempts() {
        // usize::MAX
        assert!(validate_range(usize::MAX, 0, 100).is_err());
        assert!(validate_range(usize::MAX, 0, usize::MAX).is_err());
        
        // u64::MAX
        assert!(validate_range(u64::MAX, 0u64, 100u64).is_err());
        
        // u32::MAX
        assert!(validate_range(u32::MAX, 0u32, 100u32).is_err());
        
        // u16::MAX
        assert!(validate_range(u16::MAX, 0u16, 100u16).is_err());
        
        // u8::MAX
        assert!(validate_range(u8::MAX, 0u8, 100u8).is_err());
    }
    
    #[test]
    fn test_robustness_signed_overflow_attempts() {
        // i64::MAX and i64::MIN
        assert!(validate_range(i64::MAX, 0i64, 100i64).is_err());
        assert!(validate_range(i64::MIN, 0i64, 100i64).is_err());
        
        // i32::MAX and i32::MIN
        assert!(validate_range(i32::MAX, 0i32, 100i32).is_err());
        assert!(validate_range(i32::MIN, 0i32, 100i32).is_err());
        
        // i16::MAX and i16::MIN
        assert!(validate_range(i16::MAX, 0i16, 100i16).is_err());
        assert!(validate_range(i16::MIN, 0i16, 100i16).is_err());
        
        // i8::MAX and i8::MIN
        assert!(validate_range(i8::MAX, 0i8, 100i8).is_err());
        assert!(validate_range(i8::MIN, 0i8, 100i8).is_err());
    }
    
    #[test]
    fn test_robustness_boundary_off_by_one() {
        // One below minimum
        assert!(validate_range(-1, 0, 10).is_err());
        assert!(validate_range(99, 100, 200).is_err());
        
        // Exactly at minimum (should pass)
        assert!(validate_range(0, 0, 10).is_ok());
        assert!(validate_range(100, 100, 200).is_ok());
        
        // One below maximum (should pass)
        assert!(validate_range(9, 0, 10).is_ok());
        assert!(validate_range(199, 100, 200).is_ok());
        
        // Exactly at maximum (should fail - exclusive)
        assert!(validate_range(10, 0, 10).is_err());
        assert!(validate_range(200, 100, 200).is_err());
        
        // One above maximum
        assert!(validate_range(11, 0, 10).is_err());
        assert!(validate_range(201, 100, 200).is_err());
    }
    
    #[test]
    fn test_robustness_negative_value_ranges() {
        // All negative range
        assert!(validate_range(-50, -100, -10).is_ok());
        assert!(validate_range(-100, -100, -10).is_ok()); // At min
        assert!(validate_range(-11, -100, -10).is_ok()); // One below max
        assert!(validate_range(-10, -100, -10).is_err()); // At max (exclusive)
        assert!(validate_range(-101, -100, -10).is_err()); // Below min
        
        // Range crossing zero
        assert!(validate_range(-5, -10, 10).is_ok());
        assert!(validate_range(0, -10, 10).is_ok());
        assert!(validate_range(5, -10, 10).is_ok());
        assert!(validate_range(-10, -10, 10).is_ok()); // At min
        assert!(validate_range(9, -10, 10).is_ok()); // One below max
        assert!(validate_range(10, -10, 10).is_err()); // At max
        assert!(validate_range(-11, -10, 10).is_err()); // Below min
    }
    
    #[test]
    fn test_robustness_large_ranges() {
        // Very large range
        assert!(validate_range(1_000_000, 0, 10_000_000).is_ok());
        
        // Value at start of large range
        assert!(validate_range(0, 0, 10_000_000).is_ok());
        
        // Value at end of large range
        assert!(validate_range(9_999_999, 0, 10_000_000).is_ok());
        assert!(validate_range(10_000_000, 0, 10_000_000).is_err());
    }
    
    #[test]
    fn test_robustness_floating_point_edge_cases() {
        // Very small positive values
        assert!(validate_range(0.001, 0.0, 1.0).is_ok());
        assert!(validate_range(0.0001, 0.0, 1.0).is_ok());
        
        // Very close to boundaries
        assert!(validate_range(0.999999, 0.0, 1.0).is_ok());
        assert!(validate_range(0.000001, 0.0, 1.0).is_ok());
        
        // Negative floats
        assert!(validate_range(-0.5, -1.0, 0.0).is_ok());
        assert!(validate_range(-1.0, -1.0, 0.0).is_ok()); // At min
        assert!(validate_range(0.0, -1.0, 0.0).is_err()); // At max (exclusive)
        
        // Very small range
        assert!(validate_range(0.05, 0.0, 0.1).is_ok());
        assert!(validate_range(0.0, 0.0, 0.1).is_ok());
        assert!(validate_range(0.1, 0.0, 0.1).is_err());
    }
    
    #[test]
    fn test_robustness_common_use_cases() {
        // GPU device index (0-7)
        assert!(validate_range(0, 0, 8).is_ok());
        assert!(validate_range(7, 0, 8).is_ok());
        assert!(validate_range(8, 0, 8).is_err());
        assert!(validate_range(999, 0, 8).is_err());
        
        // Max tokens (1-4096)
        assert!(validate_range(1, 1, 4097).is_ok());
        assert!(validate_range(2048, 1, 4097).is_ok());
        assert!(validate_range(4096, 1, 4097).is_ok());
        assert!(validate_range(4097, 1, 4097).is_err());
        assert!(validate_range(0, 1, 4097).is_err());
        
        // Temperature (0.0-2.0)
        assert!(validate_range(0.0, 0.0, 2.0).is_ok());
        assert!(validate_range(1.0, 0.0, 2.0).is_ok());
        assert!(validate_range(1.5, 0.0, 2.0).is_ok());
        assert!(validate_range(2.0, 0.0, 2.0).is_err());
        assert!(validate_range(-0.1, 0.0, 2.0).is_err());
        
        // Top-p (0.0-1.0)
        assert!(validate_range(0.0, 0.0, 1.0).is_ok());
        assert!(validate_range(0.9, 0.0, 1.0).is_ok());
        assert!(validate_range(1.0, 0.0, 1.0).is_err());
        
        // Batch size (1-128)
        assert!(validate_range(1, 1, 129).is_ok());
        assert!(validate_range(64, 1, 129).is_ok());
        assert!(validate_range(128, 1, 129).is_ok());
        assert!(validate_range(129, 1, 129).is_err());
        assert!(validate_range(0, 1, 129).is_err());
    }
    
    #[test]
    fn test_robustness_type_safety() {
        // Different integer types
        assert!(validate_range(50u8, 0u8, 100u8).is_ok());
        assert!(validate_range(50u16, 0u16, 100u16).is_ok());
        assert!(validate_range(50u32, 0u32, 100u32).is_ok());
        assert!(validate_range(50u64, 0u64, 100u64).is_ok());
        assert!(validate_range(50usize, 0usize, 100usize).is_ok());
        
        assert!(validate_range(50i8, 0i8, 100i8).is_ok());
        assert!(validate_range(50i16, 0i16, 100i16).is_ok());
        assert!(validate_range(50i32, 0i32, 100i32).is_ok());
        assert!(validate_range(50i64, 0i64, 100i64).is_ok());
        assert!(validate_range(50isize, 0isize, 100isize).is_ok());
        
        // Different float types
        assert!(validate_range(0.5f32, 0.0f32, 1.0f32).is_ok());
        assert!(validate_range(0.5f64, 0.0f64, 1.0f64).is_ok());
    }
    
    #[test]
    fn test_robustness_error_messages() {
        // Verify error messages contain useful information
        let result = validate_range(150, 0, 100);
        match result {
            Err(ValidationError::OutOfRange { value, min, max }) => {
                assert_eq!(value, "150");
                assert_eq!(min, "0");
                assert_eq!(max, "100");
            }
            _ => panic!("Expected OutOfRange error"),
        }
        
        // Negative values
        let result = validate_range(-50, 0, 100);
        match result {
            Err(ValidationError::OutOfRange { value, min, max }) => {
                assert_eq!(value, "-50");
                assert_eq!(min, "0");
                assert_eq!(max, "100");
            }
            _ => panic!("Expected OutOfRange error"),
        }
        
        // Float values
        let result = validate_range(1.5, 0.0, 1.0);
        match result {
            Err(ValidationError::OutOfRange { value, min, max }) => {
                assert_eq!(value, "1.5");
                assert_eq!(min, "0");
                assert_eq!(max, "1");
            }
            _ => panic!("Expected OutOfRange error"),
        }
    }
    
    #[test]
    fn test_robustness_no_arithmetic_overflow() {
        // This test verifies that the function uses only comparisons,
        // not arithmetic, so there's no risk of overflow
        
        // Even with extreme values, the function should work correctly
        // because it only compares, never adds/subtracts
        assert!(validate_range(i64::MAX - 1, i64::MAX - 10, i64::MAX).is_ok());
        assert!(validate_range(i64::MIN + 1, i64::MIN, i64::MIN + 10).is_ok());
        
        // These would overflow if we did arithmetic like (value - min)
        // but our comparison-only approach is safe
        assert!(validate_range(100, i64::MIN, i64::MAX).is_ok());
        assert!(validate_range(i64::MAX, 0, i64::MAX).is_err()); // At max (exclusive)
    }
}
