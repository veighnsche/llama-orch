#!/usr/bin/env python3
"""
Isolated LayerNorm test for tinygrad
Runs ONLY LayerNorm component with identical input to llorch-cpud
"""
import numpy as np
import sys

# Add tinygrad to path
sys.path.insert(0, '/home/vince/Projects/llama-orch/reference/tinygrad')

try:
    from tinygrad import Tensor
    from tinygrad.nn import LayerNorm
except ImportError as e:
    print(f"ERROR: Failed to import tinygrad: {e}", file=sys.stderr)
    sys.exit(1)

# Generate IDENTICAL input as Rust test
# Pattern: (i * 1024 + j) * 0.001).sin() * 0.5
input_data = np.zeros((2, 1024), dtype=np.float32)
for i in range(2):
    for j in range(1024):
        idx = i * 1024 + j
        input_data[i, j] = np.sin(idx * 0.001) * 0.5

print(f"Input shape: {input_data.shape}", file=sys.stderr)
print(f"Input sample (first 5): {input_data[0, :5]}", file=sys.stderr)

# Create LayerNorm with same params as llorch-cpud
# weight=ones, bias=zeros, eps=1e-5
dim = 1024
ln = LayerNorm(dim, eps=1e-5)

# Initialize with ones/zeros
ln.weight = Tensor.ones(dim)
ln.bias = Tensor.zeros(dim)

# Run forward
x = Tensor(input_data)
output = ln(x)

# Get output as numpy
try:
    output_np = output.numpy()
except Exception as e:
    print(f"ERROR: Failed to convert output to numpy: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Output shape: {output_np.shape}", file=sys.stderr)
print(f"Output sample (first 10): {output_np[0, :10]}", file=sys.stderr)

# Print for comparison
print("\n=== TINYGRAD OUTPUT ===")
print(f"Shape: {output_np.shape}")
print(f"First 10 elements: {output_np.flatten()[:10].tolist()}")
print(f"Mean: {output_np.mean()}")
print(f"Std: {output_np.std()}")
print(f"Min: {output_np.min()}, Max: {output_np.max()}")

# Save for Rust to load
np.save('/tmp/llorch_test_output_tinygrad.npy', output_np)
print("\n✅ SUCCESS: Output saved to /tmp/llorch_test_output_tinygrad.npy", file=sys.stderr)
