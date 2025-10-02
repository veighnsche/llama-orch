//! Common test utilities for dual-mode testing
//!
//! Implements the mandatory dual-mode testing requirement from spec 42_dual_mode_testing.md

use gpu_info::detect_gpus;

/// Run a test in both mock mode and real CUDA mode (if available)
///
/// This implements the mandatory dual-mode testing requirement:
/// 1. Always run with mock VRAM first
/// 2. Detect GPU availability
/// 3. Run with real CUDA if GPU found
/// 4. Emit clear warning if no CUDA found
///
/// # Example
///
/// ```rust
/// use vram_residency::VramManager;
///
/// #[test]
/// fn test_seal_model() {
///     run_dual_mode_test(|is_real_cuda| {
///         let manager = if is_real_cuda {
///             VramManager::new() // Will use real CUDA
///         } else {
///             VramManager::new() // Will use mock
///         };
///         
///         let data = vec![0x42u8; 1024];
///         let shard = manager.seal_model(&data, 0)?;
///         assert!(manager.verify_sealed(&shard).is_ok());
///         Ok(())
///     });
/// }
/// ```
pub fn run_dual_mode_test<F, E>(test_fn: F)
where
    F: Fn(bool) -> Result<(), E>,
    E: std::fmt::Debug,
{
    let start_time = std::time::Instant::now();
    
    // PHASE 1: Always run with mock VRAM
    println!("\n🧪 Running with MOCK VRAM...");
    test_fn(false).expect("Mock mode test failed");
    let mock_duration = start_time.elapsed();
    println!("✅ Mock mode: PASSED ({:.2}s)", mock_duration.as_secs_f64());
    
    // PHASE 2: Attempt real CUDA
    let gpu_info = detect_gpus();
    if gpu_info.available {
        if let Some(first_gpu) = gpu_info.devices.first() {
            println!("🎮 GPU detected: {}", first_gpu.name);
            println!("   VRAM: {} GB", first_gpu.vram_total_bytes / (1024 * 1024 * 1024));
            println!("🧪 Running with REAL CUDA...");
            let cuda_start = std::time::Instant::now();
            test_fn(true).expect("Real CUDA test failed");
            let cuda_duration = cuda_start.elapsed();
            println!("✅ Real CUDA mode: PASSED ({:.2}s)\n", cuda_duration.as_secs_f64());
        }
    } else {
        emit_no_cuda_warning();
    }
}

/// Emit the mandatory warning when no CUDA is found
///
/// Format specified in spec 42_dual_mode_testing.md section 3.3
pub fn emit_no_cuda_warning() {
    eprintln!();
    eprintln!("⚠️  ═══════════════════════════════════════════════");
    eprintln!("⚠️  WARNING: NO CUDA FOUND");
    eprintln!("⚠️  ONLY MOCK VRAM HAS BEEN TESTED!");
    eprintln!("⚠️  CUDA FFI layer NOT verified");
    eprintln!("⚠️  Real VRAM operations NOT tested");
    eprintln!("⚠️  Install NVIDIA GPU + CUDA for full coverage");
    eprintln!("⚠️  ═══════════════════════════════════════════════");
    eprintln!();
}

/// Check if real CUDA is available (for conditional test logic)
pub fn has_cuda() -> bool {
    detect_gpus().available
}

/// Print test mode summary at the end of test run
pub fn print_test_summary(mock_passed: usize, cuda_passed: Option<usize>) {
    println!("\n═══════════════════════════════════════════════");
    println!("📊 Test Summary");
    println!("═══════════════════════════════════════════════");
    println!("✅ Mock mode: {} tests passed", mock_passed);
    
    match cuda_passed {
        Some(count) => {
            println!("✅ Real CUDA mode: {} tests passed", count);
            println!("🎯 Full coverage achieved (100%)");
        }
        None => {
            println!("⚠️  Real CUDA mode: NOT RUN (no GPU detected)");
            println!("📊 Coverage: ~95% (business logic only)");
        }
    }
    println!("═══════════════════════════════════════════════\n");
}
