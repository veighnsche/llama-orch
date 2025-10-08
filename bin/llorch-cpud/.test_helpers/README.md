# Test Helpers for llorch-cpud

This directory contains reference implementations and comparison tools for validating llorch-cpud components.

---

## Directory Structure

```
.test_helpers/
├── candle_ln_test/          ✅ Candle LayerNorm reference (WORKING)
│   ├── Cargo.toml
│   └── src/main.rs
├── compare_outputs.py       ✅ Automated comparison script
├── test_tinygrad_ln.py      ⚠️  Tinygrad test (segfaults)
└── test_tinygrad_ln_simple.py ⚠️ Simplified tinygrad test (segfaults)
```

---

## Candle Reference (✅ Working)

### Purpose
Validates llorch-cpud LayerNorm against Candle (Hugging Face's Rust ML framework).

### Usage
```bash
cd candle_ln_test
cargo run --release
```

### Output
```
=== CANDLE LAYERNORM OUTPUT ===
Shape: [2, 1024]
First 10: [-1.8595952, -1.8556249, -1.8516545, ...]
Mean: -0.000011, Std: 0.993939
```

---

## Comparison Script (✅ Working)

### Purpose
Compares llorch-cpud output with Candle reference output.

### Usage
```bash
python3 compare_outputs.py
```

### Output
```
=== COMPARISON: llorch-cpud vs Candle ===
Index | Ours        | Candle      | Diff        | Pass?
------------------------------------------------------------
    0 |  -1.8595886 |  -1.8595952 | 6.6000000e-06 | ✅
    ...

Max difference: 6.6000000e-06
Tolerance:      1.0000000e-04
✅ PASS: All values within tolerance
```

---

## Tinygrad Tests (⚠️ Blocked)

### Status
Both tinygrad test scripts segfault when calling `Tensor.numpy()`.

### Error
```bash
$ python3 test_tinygrad_ln_simple.py
Input shape: (2, 1024)
Segmentation fault (core dumped)
```

### Root Cause
Tinygrad environment issue on this system. Not needed since Candle validation works.

---

## Validation Workflow

1. **Run llorch-cpud test:**
   ```bash
   cd ..
   cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_all -- --nocapture
   ```

2. **Run Candle reference:**
   ```bash
   cd .test_helpers/candle_ln_test
   cargo run --release
   ```

3. **Compare outputs:**
   ```bash
   cd ../..
   python3 .test_helpers/compare_outputs.py
   ```

4. **Expected result:**
   ```
   ✅ PASS: All values within tolerance
   Max difference: 6.6000000e-06
   ```

---

## Validation Results

✅ **llorch-cpud LayerNorm validated against Candle**
- Maximum difference: 6.6e-06
- Tolerance: 1e-4
- Status: **PASS**

See `../VALIDATION_SUMMARY.md` for full details.
