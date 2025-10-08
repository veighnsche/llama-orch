# Test Helpers for llorch-cpud

This directory contains reference implementations and comparison tools for validating llorch-cpud components.

---

## Directory Structure

```
.test_helpers/
├── candle_ln_test/          ✅ Candle LayerNorm reference (WORKING)
│   ├── Cargo.toml
│   └── src/main.rs
├── mistralrs_ln_test/       ✅ Mistral.rs LayerNorm reference (WORKING)
│   ├── Cargo.toml
│   └── src/main.rs
├── compare_outputs.py       ✅ Automated comparison script
├── run_validation.sh        ✅ Complete validation suite
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

## Mistral.rs Reference (✅ Working)

### Purpose
Validates llorch-cpud LayerNorm against Mistral.rs (which uses Candle's LayerNorm).

### Usage
```bash
cd mistralrs_ln_test
cargo run --release
```

### Output
```
=== MISTRAL.RS LAYERNORM OUTPUT ===
(Uses Candle's LayerNorm - see mistralrs-core/src/layers.rs)
Shape: [2, 1024]
First 10: [-1.8595952, -1.8556249, -1.8516545, ...]
Mean: -0.000011, Std: 0.993939
```

**Note:** Mistral.rs imports `LayerNorm` from `candle_nn` (line 11 of `mistralrs-core/src/layers.rs`), so results are identical to Candle.

---

## Comparison Script (✅ Working)

### Purpose
Compares llorch-cpud output with Candle and Mistral.rs reference outputs.

### Usage
```bash
python3 compare_outputs.py
```

### Output
```
=== COMPARISON: llorch-cpud vs Candle/Mistral.rs ===
Index | Ours        | Candle      | Diff        | Pass?
------------------------------------------------------------
    0 |  -1.8595886 |  -1.8595952 | 6.6000000e-06 | ✅
    ...

Max difference: 6.6000000e-06
Tolerance:      1.0000000e-04
✅ PASS: All values within tolerance
```

---

## Validation Suite (✅ Working)

### Purpose
Runs all tests automatically: llorch-cpud, Candle, Mistral.rs, and comparison.

### Usage
```bash
./run_validation.sh
```

### Output
```
[1/4] Running llorch-cpud LayerNorm test...
[2/4] Running Candle reference implementation...
[3/4] Running Mistral.rs reference implementation...
[4/4] Comparing outputs...
✅ PASS: All values within tolerance
✅ Validation complete!
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
Tinygrad environment issue on this system. Not needed since Candle and Mistral.rs validation works.

---

## Validation Workflow

### Quick (Recommended)
```bash
./run_validation.sh
```

### Manual

1. **Run llorch-cpud test:**
   ```bash
   cd ..
   cargo test --test isolated_checkpoint_01 test_isolated_checkpoint_01_all -- --nocapture
   ```

2. **Run Candle reference:**
   ```bash
   cd .test_helpers/candle_ln_test
   cargo run --release
   cd ../..
   ```

3. **Run Mistral.rs reference:**
   ```bash
   cd .test_helpers/mistralrs_ln_test
   cargo run --release
   cd ../..
   ```

4. **Compare outputs:**
   ```bash
   python3 .test_helpers/compare_outputs.py
   ```

5. **Expected result:**
   ```
   ✅ PASS: All values within tolerance
   Max difference: 6.6000000e-06
   ```

---

## Validation Results

✅ **llorch-cpud LayerNorm validated against Candle & Mistral.rs**
- Maximum difference: 6.6e-06
- Tolerance: 1e-4
- Status: **PASS**
- Note: Mistral.rs uses Candle's LayerNorm, so both references are identical

See `../VALIDATION_SUMMARY.md` for full details.
