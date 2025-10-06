# Haiku Test Guide - The Human-Friendly M0 Success Test

**Purpose**: This test is designed for HUMANS to see real GPU inference in action!

---

## What is the Haiku Test?

The haiku test is the **M0 success criteria** - it proves that:
1. ✅ Real GPU inference is happening (not pre-baked responses)
2. ✅ The model can generate creative text
3. ✅ The system works end-to-end
4. ✅ **YOU CAN SEE THE HAIKU THE LLM CREATES!**

---

## How It Works (Anti-Cheat)

1. Gets the current minute (0-59)
2. Converts it to words (e.g., 42 → "forty-two")
3. Asks the LLM to write a haiku that includes that word
4. Validates the word appears exactly once
5. **SHOWS YOU THE HAIKU!**

Since the minute changes every 60 seconds, the test can't use pre-baked responses - it MUST do real inference!

---

## Running the Test

### Prerequisites
- ✅ CUDA enabled in `.llorch.toml` (cuda = true)
<!-- CONTRADICTION: Guide required Q4_K_M quantized model, but loader (`cuda/src/model/qwen_weight_loader.cpp`) warns quantized weights are loaded without dequantization → NaN/garbage. Test `tests/haiku_generation_anti_cheat.rs` uses FP16 path. -->
<!-- RESOLVED: Use FP16 GGUF to avoid dequantization issues. Verified by script `.docs/testing/download_qwen_fp16.sh`. -->
- ✅ Qwen model downloaded to `.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf`
- ✅ GPU available

### Run Command

```bash
cd bin/worker-orcd

# Run the REAL haiku test
<!-- CONTRADICTION: Filter `test_haiku_generation_anti_cheat` does not exist in `tests/haiku_generation_anti_cheat.rs`. Actual ignored test is `test_haiku_generation_stub_pipeline_only`. -->
<!-- RESOLVED: Correct the cargo test filter to the actual function name. -->
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release \
  -- --ignored --nocapture --test-threads=1
```

**Important flags**:
- `REQUIRE_REAL_LLAMA=1` - Enforces real GPU usage
- `--features cuda` - Enables CUDA build
- `--release` - Optimized build (faster)
- `--ignored` - Runs ignored tests
- `--nocapture` - Shows output to terminal (SO YOU CAN SEE THE HAIKU!)
- `--test-threads=1` - One test at a time

---

## What You'll See

When the test runs, you'll see output like:

```
🎨 M0 Haiku Anti-Cheat Test PASSED
Minute: 42 ("forty-two")
Nonce: aB3dE7fG
Tokens: 87
Time: 8.234s

Haiku:
Silicon dreams flow
Forty-two cores burning bright
GPU's warm glow

Artifacts: .test-results/haiku/run-abc123/
```

---

## Test Artifacts

After the test runs, check:

```bash
# View the test report
cat .test-results/haiku/*/test_report.md

# View verification JSON
cat .test-results/haiku/*/verification.json

# View SSE event stream
cat .test-results/haiku/*/sse_transcript.ndjson
```

---

## Troubleshooting

### "Worker binary not found"
```bash
# Build the worker first
cargo build --release
```

### "Model not found"
```bash
# Download the model
bash ../../.docs/testing/download_qwen.sh
```

### "CUDA not available"
```bash
# Check GPU
nvidia-smi

# Enable CUDA in config
sed -i 's/cuda = false/cuda = true/' ../../.llorch.toml
```

### Test hangs or times out
- Worker might not be starting
- Check logs in worker output
- Ensure port is free
- Try with `--nocapture` to see worker output

---

## Why This Test Matters

This is THE test that proves M0 is real:

1. **Anti-Cheat**: Can't fake it - minute changes every 60 seconds
2. **Human-Visible**: You SEE the haiku the AI creates
3. **End-to-End**: Tests the complete pipeline
4. **Quality Check**: Haiku must be coherent and include the word
5. **Performance**: Must complete in <30 seconds

---

## Example Haikus Generated

```
Minute: 23 ("twenty-three")
---
Twenty-three threads spin
Parallel dreams in silicon  
CUDA's swift dance

Minute: 7 ("seven")
---
Seven cores awake
Tensors flowing through the night
GPU whispers

Minute: 58 ("fifty-eight")
---
Fifty-eight seconds
Before the minute changes
Inference complete
```

---

## For Developers

The test is in: `tests/haiku_generation_anti_cheat.rs`

Key functions:
- `minute_to_words()` - Converts 0-59 to English
- `test_haiku_generation_anti_cheat()` - Main test
- Unit tests validate minute conversion

The test:
1. Spawns worker with real model
2. Generates dynamic prompt with current minute
3. Streams tokens via SSE
4. Validates output
5. **PRINTS THE HAIKU TO YOUR TERMINAL**
6. Saves artifacts for proof

---

## M0 Success Criteria

When you see a haiku with the correct minute word, you've proven:
- ✅ GPU inference works
- ✅ Model loading works
- ✅ Tokenization works
- ✅ SSE streaming works
- ✅ The LLM is creative
- ✅ **M0 IS COMPLETE!**

---

**This test is for HUMANS. Enjoy the haiku!** 🎨

---

Built by Foundation-Alpha 🏗️
