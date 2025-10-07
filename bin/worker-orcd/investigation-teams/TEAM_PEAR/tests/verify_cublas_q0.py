#!/usr/bin/env python3
"""
TEAM PEAR - Manual Q[0] Verification
Verify Team Sentinel's claim: "Manual Q[0]=-0.015185, cuBLAS Q[0]=-0.015182"

This script will:
1. Load weight matrix (attn_q_weight) from GGUF
2. Load normed input from test run
3. Manually compute Q[0] = dot(weight_row_0, normed)
4. Compare with cuBLAS output from logs
"""

import struct
import numpy as np
import sys

def load_fp16_weights(gguf_path, tensor_name, expected_shape):
    """Load FP16 tensor from GGUF file"""
    # TODO: Parse GGUF format to extract tensor
    # For now, placeholder
    print(f"TODO: Load {tensor_name} from {gguf_path}")
    print(f"Expected shape: {expected_shape}")
    return None

def manual_dot_product(weight_row, normed_input):
    """Compute dot product manually"""
    if weight_row is None or normed_input is None:
        return None
    
    result = np.dot(weight_row.astype(np.float32), normed_input.astype(np.float32))
    return result

def main():
    model_path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf"
    
    print("=" * 80)
    print("TEAM PEAR - Manual Q[0] Verification")
    print("=" * 80)
    
    # Step 1: Load attn_q_weight (layer 0)
    print("\n[1] Loading attn_q_weight from GGUF...")
    # Shape: [hidden_dim=896, q_dim=896] in row-major
    weight = load_fp16_weights(model_path, "blk.0.attn_q.weight", (896, 896))
    
    # Step 2: Load normed input from test run
    print("\n[2] Loading normed input from test run...")
    # TODO: Extract from haiku test logs
    # Expected: 896 FP16 values
    normed = None
    
    # Step 3: Manual calculation
    print("\n[3] Computing Q[0] manually...")
    if weight is not None and normed is not None:
        q0_manual = manual_dot_product(weight[0], normed)
        print(f"Manual Q[0] = {q0_manual:.6f}")
    else:
        print("BLOCKED: Need to implement GGUF parsing and log extraction")
        print("\nRequired:")
        print("1. GGUF parser to extract attn_q_weight")
        print("2. Log parser to extract normed input from test run")
        print("3. cuBLAS output from test run for comparison")
    
    # Step 4: Compare with cuBLAS
    print("\n[4] Comparing with cuBLAS output...")
    # TODO: Extract from test logs
    cublas_q0 = None  # From logs
    
    if q0_manual is not None and cublas_q0 is not None:
        diff = abs(q0_manual - cublas_q0)
        print(f"cuBLAS Q[0] = {cublas_q0:.6f}")
        print(f"Difference  = {diff:.6f}")
        
        if diff < 0.0001:
            print("✅ VERIFIED: Manual matches cuBLAS")
        else:
            print("❌ MISMATCH: Manual does not match cuBLAS")
    
    print("\n" + "=" * 80)
    print("STATUS: Test infrastructure incomplete - need GGUF parser")
    print("=" * 80)

if __name__ == "__main__":
    main()
