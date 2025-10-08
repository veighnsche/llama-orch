# Candle GPT-2 Reference Generator

**Purpose:** Generate reference checkpoint data from Candle for multi-reference validation.

**Created by:** TEAM-003  
**Lesson from worker-orcd:** Compare with reference from Day 1

---

## Why This Exists

From worker-orcd post-mortem (PHASE_1_FINAL_REPORT.md):

> **The Real Problem: What Nobody Did**
> 
> Compare intermediate values with llama.cpp at each step.
> 
> DICKINSON started this but never completed it:
> - Never instrumented llama.cpp
> - Never got reference values
> - Never did the comparison
> - Never found first divergence

**We will NOT repeat this mistake.**

---

## What It Does

Loads REAL GPT-2 base (124M) from HuggingFace using Candle and extracts:

1. **Checkpoint 0:** Embeddings (token + position)
2. **Checkpoint 1:** LayerNorm output (ln_1)
3. **Checkpoint 2:** Q/K/V projections
4. **Checkpoint 4:** Attention scores
5. **Checkpoint 5:** Attention output
6. **Checkpoint 6:** FFN output

All saved as `.npy` files for comparison with our implementation.

---

## Usage

### Build and Run

```bash
cd bin/llorch-cpud/.test_helpers/candle_gpt2_reference
cargo run --release
```

### Output

Files written to: `../../.test-models/gpt2/extracted_weights/`

- `checkpoint_00_embeddings_candle.npy`
- `checkpoint_01_ln1_output_candle.npy`
- `checkpoint_02_q_candle.npy`
- `checkpoint_02_k_candle.npy`
- `checkpoint_02_v_candle.npy`
- `checkpoint_04_scores_candle.npy`
- `checkpoint_05_output_candle.npy`
- `checkpoint_06_ffn_candle.npy`

### Run Tests

```bash
cd ../..
cargo test --test real_gpt2_checkpoint_01 test_checkpoint_01_multi_reference -- --nocapture
cargo test --test real_gpt2_checkpoint_06 test_checkpoint_06_multi_reference -- --nocapture
```

---

## Multi-Reference Validation

Each checkpoint test now validates against:

1. **PyTorch (HuggingFace)** - Primary reference
2. **Candle (Rust)** - Secondary reference
3. **Cross-validation** - PyTorch vs Candle agreement

### Example Output

```
╔══════════════════════════════════════════════════════════╗
║  Checkpoint 6: Multi-Reference Validation                ║
║  PyTorch + Candle Cross-Validation                       ║
╚══════════════════════════════════════════════════════════╝

✅ PYTORCH: FFN matches HuggingFace (max diff 1.5e-5)

📊 Candle Comparison:
  Max absolute difference: 2.1e-5
✅ CANDLE: Matches within tolerance

📊 Cross-Validation (PyTorch vs Candle):
  Max difference: 1.8e-5
✅ CROSS-VALIDATION: References agree

🎉 MULTI-REFERENCE VALIDATION PASSED!
   Our implementation matches BOTH PyTorch and Candle
```

---

## Why This Matters

### Single-Reference Risk (What We Had)

```
Our Code → PyTorch ✅
```

**Problem:** If PyTorch reference has bugs, we can't detect it.

### Multi-Reference Safety (What We Have Now)

```
Our Code → PyTorch ✅
Our Code → Candle ✅
PyTorch ↔ Candle ✅
```

**Benefit:** Three-way validation catches:
- Bugs in our code
- Bugs in PyTorch reference
- Bugs in Candle reference
- Numerical precision differences

---

## Lesson Applied

From worker-orcd post-mortem:

> **Lesson #1: Compare with Reference from Day 1**
> 
> ```rust
> // MANDATORY for every component
> #[test]
> fn test_matches_reference() {
>     let our_output = our_component(input);
>     let reference = reference_component(input);
>     assert_eq!(our_output, reference); // Must pass!
> }
> ```

**We implement this lesson with multi-reference validation.**

---

## Dependencies

- `candle-core` - Tensor operations
- `candle-nn` - Neural network layers
- `candle-transformers` - GPT-2 model
- `hf-hub` - Download from HuggingFace
- `tokenizers` - Tokenization
- `ndarray-npy` - Save as NumPy format

---

## Troubleshooting

### "Failed to download model"

Ensure internet connection and HuggingFace access:
```bash
curl -I https://huggingface.co/gpt2/resolve/main/model.safetensors
```

### "Tensor dimension mismatch"

Check GPT-2 config matches:
- 12 layers
- 12 heads
- 768 hidden dim
- 64 head dim

### "Output files not created"

Check write permissions:
```bash
ls -la ../../.test-models/gpt2/extracted_weights/
```

---

**Created:** 2025-10-08  
**Author:** TEAM-003  
**Status:** Production Ready
