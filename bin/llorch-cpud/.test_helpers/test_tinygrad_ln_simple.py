#!/usr/bin/env python3
"""Isolated LayerNorm test for tinygrad - SIMPLE VERSION"""
import numpy as np
import sys
sys.path.insert(0, '/home/vince/Projects/llama-orch/reference/tinygrad')

from tinygrad import Tensor
from tinygrad.nn import LayerNorm

# Generate IDENTICAL input as Rust test
input_data = np.zeros((2, 1024), dtype=np.float32)
for i in range(2):
  for j in range(1024):
    idx = i * 1024 + j
    input_data[i, j] = np.sin(idx * 0.001) * 0.5

print(f"Input shape: {input_data.shape}", file=sys.stderr)
print(f"Input sample: {input_data[0, :5]}", file=sys.stderr)

# Create LayerNorm: weight=ones, bias=zeros, eps=1e-5
ln = LayerNorm(1024, eps=1e-5)
ln.weight = Tensor.ones(1024)
ln.bias = Tensor.zeros(1024)

# Run forward
x = Tensor(input_data)
output = ln(x)
output_np = output.numpy()

print(f"\nOutput shape: {output_np.shape}", file=sys.stderr)
print(f"Output sample (first 10): {output_np[0, :10]}", file=sys.stderr)

# Print for comparison
print("\n=== TINYGRAD LAYERNORM OUTPUT ===")
print(f"Shape: {output_np.shape}")
print(f"First 10: {output_np.flatten()[:10].tolist()}")
print(f"Mean: {output_np.mean():.6f}, Std: {output_np.std():.6f}")
print(f"Min: {output_np.min():.6f}, Max: {output_np.max():.6f}")

# Save
np.save('/tmp/llorch_test_output_tinygrad.npy', output_np)
print("\n✅ SUCCESS", file=sys.stderr)
