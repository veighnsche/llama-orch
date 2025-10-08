#!/usr/bin/env python3
"""
Compare llorch-cpud LayerNorm output with Candle and Mistral.rs references
"""
import sys

# Our output from test
ours = [
    -1.8595886, -1.8556184, -1.8516481, -1.8476778, -1.8437077,
    -1.8397374, -1.8357671, -1.831797, -1.8278267, -1.8238567
]

# Candle output
candle = [
    -1.8595952, -1.8556249, -1.8516545, -1.8476844, -1.8437141,
    -1.8397439, -1.8357736, -1.8318034, -1.8278332, -1.823863
]

# Mistral.rs output (uses same Candle LayerNorm, so identical to Candle)
mistralrs = candle

print("=== COMPARISON: llorch-cpud vs Candle/Mistral.rs ===\n")
print("Index | Ours        | Candle      | Diff        | Pass?")
print("-" * 60)

max_diff = 0.0
tolerance = 1e-4

for i in range(len(ours)):
    diff = abs(ours[i] - candle[i])
    max_diff = max(max_diff, diff)
    passed = "✅" if diff < tolerance else "❌"
    print(f"{i:5d} | {ours[i]:11.7f} | {candle[i]:11.7f} | {diff:11.7e} | {passed}")

print("-" * 60)
print(f"\nMax difference: {max_diff:.7e}")
print(f"Tolerance:      {tolerance:.7e}")
print(f"\nNote: Mistral.rs uses Candle's LayerNorm (mistralrs-core/src/layers.rs:11)")
print(f"      so Candle and Mistral.rs outputs are identical.")

if max_diff < tolerance:
    print("\n✅ PASS: All values within tolerance")
    sys.exit(0)
else:
    print(f"\n❌ FAIL: Max difference {max_diff:.7e} exceeds tolerance {tolerance:.7e}")
    sys.exit(1)
