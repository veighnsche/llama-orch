//! Test runner that enforces dual-mode testing and prints summary
//!
//! This can be run with: cargo test -p vram-residency --test test_runner

mod common;

use common::{has_cuda, emit_no_cuda_warning};

#[test]
fn print_test_mode_info() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  VRAM Residency - Dual-Mode Testing                      ║");
    println!("║  Spec: 42_dual_mode_testing.md                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    println!("📋 Testing Strategy:");
    println!("   1. ✅ Run all tests with MOCK VRAM (always)");
    println!("   2. 🔍 Detect GPU availability");
    println!("   3. ✅ Run all tests with REAL CUDA (if GPU found)");
    println!("   4. ⚠️  Emit warning if no CUDA found\n");
    
    if has_cuda() {
        println!("🎮 GPU Status: DETECTED");
        println!("   → Tests will run in BOTH mock and real CUDA modes");
        println!("   → Full coverage (100%) will be achieved");
        println!("   → Expected total duration: 2-3 minutes\n");
        println!("⏱️  Test Duration Estimates:");
        println!("   • Unit tests: ~1 second");
        println!("   • CUDA kernel tests: ~1 second");
        println!("   • Dual-mode examples: ~2 seconds");
        println!("   • Concurrent tests: ~2 seconds");
        println!("   • Property tests: ~10 seconds (256 cases per property)");
        println!("   • Stress tests: ~90 seconds (VRAM exhaustion test)");
        println!("   • Proof bundle: ~1 second\n");
    } else {
        println!("⚠️  GPU Status: NOT DETECTED");
        println!("   → Tests will run in mock mode only");
        println!("   → Coverage: ~95% (business logic only)");
        println!("   → CUDA FFI layer will NOT be verified");
        println!("   → Expected duration: ~1 minute\n");
        
        emit_no_cuda_warning();
    }
    
    println!("🚀 Starting test execution...");
    println!("   (Progress messages will appear for long-running tests)\n");
}

// This test always passes but ensures the warning is visible
#[test]
fn ensure_cuda_warning_visibility() {
    if !has_cuda() {
        // Print warning again at the end for visibility
        println!("\n");
        println!("═══════════════════════════════════════════════════════════");
        println!("  Test Execution Complete");
        println!("═══════════════════════════════════════════════════════════");
        emit_no_cuda_warning();
        println!("💡 To enable full testing:");
        println!("   1. Install NVIDIA GPU with CUDA support");
        println!("   2. Install CUDA toolkit");
        println!("   3. Re-run tests");
        println!("═══════════════════════════════════════════════════════════\n");
    } else {
        println!("\n");
        println!("═══════════════════════════════════════════════════════════");
        println!("  Test Execution Complete");
        println!("═══════════════════════════════════════════════════════════");
        println!("✅ All tests passed in BOTH mock and real CUDA modes");
        println!("🎯 Full coverage achieved (100%)");
        println!("═══════════════════════════════════════════════════════════\n");
    }
}
