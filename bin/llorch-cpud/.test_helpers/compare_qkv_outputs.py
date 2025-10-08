#!/usr/bin/env python3
"""
Compare QKV outputs from llorch-cpud vs Candle/Mistral.rs references

This script compares the Q, K, V outputs from our implementation
against reference implementations to validate correctness.
"""

import sys
from pathlib import Path

def load_values(filename):
    """Load float values from a text file (one per line)"""
    try:
        with open(filename) as f:
            values = [float(line.strip()) for line in f if line.strip()]
        return values
    except FileNotFoundError:
        print(f"⚠️  File not found: {filename}")
        return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def compare_arrays(name, ours, reference, tolerance=1e-4):
    """Compare two arrays and report differences"""
    if ours is None or reference is None:
        return False
    
    # Compare first N values (in case arrays have different lengths)
    n = min(len(ours), len(reference))
    
    max_diff = 0.0
    max_rel_diff = 0.0
    failures = []
    
    for i in range(n):
        our_val = ours[i]
        ref_val = reference[i]
        
        abs_diff = abs(our_val - ref_val)
        rel_diff = abs_diff / abs(ref_val) if abs(ref_val) > 1e-10 else abs_diff
        
        max_diff = max(max_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        
        if abs_diff > tolerance:
            if len(failures) < 10:
                failures.append((i, our_val, ref_val, abs_diff, rel_diff))
    
    print(f"\n=== {name} Comparison ===")
    print(f"Comparing first {n} values")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Tolerance: {tolerance:.6e}")
    
    # Show sample values
    print(f"Our output (first 5):  {ours[:5]}")
    print(f"Ref output (first 5):  {reference[:5]}")
    
    if failures:
        print(f"\n❌ {len(failures)} elements exceed tolerance:")
        for i, our_val, ref_val, abs_diff, rel_diff in failures[:5]:
            print(f"  Element {i}: ours={our_val:.6f}, ref={ref_val:.6f}, "
                  f"diff={abs_diff:.6e} ({rel_diff*100:.2f}%)")
        return False
    else:
        print("✅ PASS: All values within tolerance")
        return True

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  QKV Projection Validation: llorch-cpud vs References   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # File paths
    our_q = "checkpoint_02_q_ours.txt"
    our_k = "checkpoint_02_k_ours.txt"
    our_v = "checkpoint_02_v_ours.txt"
    
    candle_q = "checkpoint_02_q_candle.txt"
    candle_k = "checkpoint_02_k_candle.txt"
    candle_v = "checkpoint_02_v_candle.txt"
    
    mistralrs_q = "checkpoint_02_q_mistralrs.txt"
    mistralrs_k = "checkpoint_02_k_mistralrs.txt"
    mistralrs_v = "checkpoint_02_v_mistralrs.txt"
    
    # Load our outputs
    print("\n📂 Loading our outputs...")
    q_ours = load_values(our_q)
    k_ours = load_values(our_k)
    v_ours = load_values(our_v)
    
    if not all([q_ours, k_ours, v_ours]):
        print("\n❌ Failed to load our outputs. Run the test first:")
        print("   cargo test --test isolated_checkpoint_02 test_isolated_checkpoint_02_all -- --nocapture")
        return 1
    
    # Load Candle reference
    print("\n📂 Loading Candle reference outputs...")
    q_candle = load_values(candle_q)
    k_candle = load_values(candle_k)
    v_candle = load_values(candle_v)
    
    # Load Mistral.rs reference
    print("\n📂 Loading Mistral.rs reference outputs...")
    q_mistralrs = load_values(mistralrs_q)
    k_mistralrs = load_values(mistralrs_k)
    v_mistralrs = load_values(mistralrs_v)
    
    # Compare against Candle
    candle_results = []
    if all([q_candle, k_candle, v_candle]):
        print("\n" + "="*60)
        print("CANDLE REFERENCE COMPARISON")
        print("="*60)
        candle_results.append(compare_arrays("Q (Candle)", q_ours, q_candle))
        candle_results.append(compare_arrays("K (Candle)", k_ours, k_candle))
        candle_results.append(compare_arrays("V (Candle)", v_ours, v_candle))
    else:
        print("\n⚠️  Candle reference not available. Run:")
        print("   cd .test_helpers/candle_qkv_test && cargo run --release")
    
    # Compare against Mistral.rs
    mistralrs_results = []
    if all([q_mistralrs, k_mistralrs, v_mistralrs]):
        print("\n" + "="*60)
        print("MISTRAL.RS REFERENCE COMPARISON")
        print("="*60)
        mistralrs_results.append(compare_arrays("Q (Mistral.rs)", q_ours, q_mistralrs))
        mistralrs_results.append(compare_arrays("K (Mistral.rs)", k_ours, k_mistralrs))
        mistralrs_results.append(compare_arrays("V (Mistral.rs)", v_ours, v_mistralrs))
    else:
        print("\n⚠️  Mistral.rs reference not available. Run:")
        print("   cd .test_helpers/mistralrs_qkv_test && cargo run --release")
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if candle_results and all(candle_results):
        print("✅ CANDLE: All QKV outputs match within tolerance")
    elif candle_results:
        print("❌ CANDLE: Some outputs exceed tolerance")
    else:
        print("⚠️  CANDLE: Not validated (reference not available)")
    
    if mistralrs_results and all(mistralrs_results):
        print("✅ MISTRAL.RS: All QKV outputs match within tolerance")
    elif mistralrs_results:
        print("❌ MISTRAL.RS: Some outputs exceed tolerance")
    else:
        print("⚠️  MISTRAL.RS: Not validated (reference not available)")
    
    # Exit code
    if (candle_results and all(candle_results)) or (mistralrs_results and all(mistralrs_results)):
        print("\n🎉 Checkpoint 2 validation PASSED!")
        return 0
    elif candle_results or mistralrs_results:
        print("\n❌ Checkpoint 2 validation FAILED!")
        return 1
    else:
        print("\n⚠️  No reference implementations available for comparison")
        return 2

if __name__ == "__main__":
    sys.exit(main())
