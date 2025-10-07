# TEAM PRINTER — Go/No-Go Checklist

**Status:** ✅ READY TO RUN  
**Date:** 2025-10-07T01:33:09Z

---

## ✅ Wiring & Build

- [x] **Checkpoint logger compiled into build**
  - Added `src/utils/checkpoint_logger.cpp` to `cuda/CMakeLists.txt` line 58
  - Header: `cuda/src/utils/checkpoint_logger.h`
  - Implementation: `cuda/src/utils/checkpoint_logger.cpp`

- [x] **Logger integrated into transformer**
  - `#include "../utils/checkpoint_logger.h"` added to `qwen_transformer.cpp` line 5
  - `team_printer::init_checkpoint_logging()` called in constructor (line 302)
  - `team_printer::finalize_checkpoint_logging()` called in destructor (line 311)

- [x] **Output directory created**
  ```bash
  mkdir -p bin/worker-orcd/investigation-teams/TEAM_PRINTER_PARITY
  ```

- [x] **Full rebuild required**
  ```bash
  cd /home/vince/Projects/llama-orch/bin/worker-orcd
  cargo clean
  cargo build --release --features cuda
  ```

---

## ✅ Environment Variables

Set these before running:

```bash
export PRINTER_CHECKPOINT_LOGGING=1
export PRINTER_TOKEN_LIMIT=2
export PRINTER_OUTPUT_PATH="investigation-teams/TEAM_PRINTER_PARITY/ours.checkpoints"
export REQUIRE_REAL_LLAMA=1
```

**Note:** `PRINTER_OUTPUT_PATH` should NOT include `.npz` extension. The converter will add it.

---

## ✅ Determinism

Both engines must use identical settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.0 | Greedy sampling |
| Top-p | 1.0 | No nucleus sampling |
| Top-k | 0 | No top-k filtering |
| Seed | 12345 | Fixed seed |
| Model | qwen2.5-0.5b-instruct-fp16.gguf | Same file for both |
| Prompt | "Write a haiku about GPU computing" | Exact match |

---

## ✅ CUDA Safety

The checkpoint logger uses `cudaMemcpy(device → host)`:

- **Requirement:** Pointers passed to `log_checkpoint_*` must be valid device pointers
- **Error checking:** Added to `checkpoint_logger.h` inline functions
- **Stream safety:** Currently uses default stream (nullptr)

**Red flags to watch for:**
- Segfaults during memcpy → invalid device pointer
- All-zero data → wrong pointer or size
- CUDA errors in stderr → check with `cudaGetLastError()`

---

## ✅ llama.cpp Verification

- [x] **llama-cli exists**
  ```bash
  ls -lh /home/vince/Projects/llama-orch/reference/llama.cpp/build/bin/llama-cli
  ```
  ✅ Confirmed: executable exists

- [x] **CUDA enabled**
  - Built with `cmake .. -DGGML_CUDA=ON`
  - Uses same GPU as our engine

- [x] **Token limit set**
  - Use `--n-predict 2` to match our token 0/1 capture
  - Or `--n-predict 50` for full haiku comparison

---

## 🔧 Run Sequence

### Step 1: Clean Build

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo clean
cargo build --release --features cuda
```

**Expected:** Build succeeds, `checkpoint_logger.cpp` compiles without errors.

### Step 2: Run Our Engine

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
./investigation-teams/TEAM_PRINTER_PARITY/run_our_engine.sh
```

**Expected output:**
```
[TEAM PRINTER] Checkpoint logging ENABLED
[TEAM PRINTER] Token limit: 2
[TEAM PRINTER] Output path: investigation-teams/TEAM_PRINTER_PARITY/ours.checkpoints
...
[TEAM PRINTER] Finalizing checkpoint logging...
[TEAM PRINTER] Saved manifest: investigation-teams/TEAM_PRINTER_PARITY/ours.checkpoints.manifest.json
[TEAM PRINTER] ✅ Checkpoint logging complete
```

### Step 3: Convert to NPZ

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/investigation-teams/TEAM_PRINTER_PARITY
python3 convert_to_npz.py ours.checkpoints.manifest.json
```

**Expected:**
```
[TEAM PRINTER] Loading N checkpoints...
[TEAM PRINTER] Saved N arrays to ours.checkpoints.npz
[TEAM PRINTER] ✅ Conversion complete
```

### Step 4: Sanity Check

```bash
python3 - << 'PY'
import numpy as np
d = np.load('ours.checkpoints.npz')
print(f"Arrays: {len(d.files)}")
for k in d.files[:3]:
    a = d[k]
    print(f"{k}: shape={a.shape}, min={a.min():.6f}, max={a.max():.6f}, mean={a.mean():.6f}")
PY
```

**Expected:** Should print array stats, no errors.

### Step 5: Run llama.cpp

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
./investigation-teams/TEAM_PRINTER_PARITY/run_llamacpp.sh
```

**Expected:** llama.cpp generates output, logs saved to `llamacpp.run.log`.

### Step 6: Manual Comparison

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/investigation-teams/TEAM_PRINTER_PARITY

# Compare token IDs
grep -i "token" ours.run.log | head -20
grep -i "token" llamacpp.run.log | head -20

# Compare output quality
tail -50 ours.run.log
tail -50 llamacpp.run.log
```

---

## 🚨 Red Flags (Abort & Fix)

### No [TEAM PRINTER] logs at startup
**Symptom:** No "Checkpoint logging ENABLED" message  
**Cause:** Logger not initialized  
**Fix:** Verify `init_checkpoint_logging()` is called in transformer constructor

### manifest.json is empty
**Symptom:** File exists but has `"checkpoints": []`  
**Cause:** Token index filter cut everything off, or no checkpoints logged  
**Fix:** Check `PRINTER_TOKEN_LIMIT` and verify `log_checkpoint_*` calls exist

### cudaMemcpy failures
**Symptom:** CUDA errors in stderr, or segfaults  
**Cause:** Invalid device pointer or wrong size  
**Fix:** Add error checking after memcpy calls

### All-zero data
**Symptom:** Arrays load but all values are 0.0  
**Cause:** Wrong pointer space (host instead of device)  
**Fix:** Verify pointers passed to logger are device pointers

### Build fails
**Symptom:** Linker errors about `team_printer` symbols  
**Cause:** `checkpoint_logger.cpp` not in CMakeLists.txt  
**Fix:** Verify line 58 of `cuda/CMakeLists.txt` includes the file

---

## ✅ Success Criteria

You've succeeded if:

1. **Build completes** without errors
2. **[TEAM PRINTER] logs appear** at start and end of test
3. **manifest.json exists** and contains checkpoint entries
4. **NPZ file loads** in Python without errors
5. **Arrays have reasonable values** (not all zeros, not NaN)

---

## 📝 Current Status

**Files Created:**
- ✅ `cuda/src/utils/checkpoint_logger.h` (header)
- ✅ `cuda/src/utils/checkpoint_logger.cpp` (implementation)
- ✅ `investigation-teams/TEAM_PRINTER_PARITY/printer_meta.json` (metadata)
- ✅ `investigation-teams/TEAM_PRINTER_PARITY/README.md` (documentation)
- ✅ `investigation-teams/TEAM_PRINTER_PARITY/run_our_engine.sh` (runner script)
- ✅ `investigation-teams/TEAM_PRINTER_PARITY/run_llamacpp.sh` (runner script)
- ✅ `investigation-teams/TEAM_PRINTER_PARITY/convert_to_npz.py` (converter)
- ✅ `investigation-teams/TEAM_PRINTER_PARITY/collect_parity_data.py` (diff tool)

**Files Modified:**
- ✅ `cuda/CMakeLists.txt` (added checkpoint_logger.cpp to build)
- ✅ `cuda/src/transformer/qwen_transformer.cpp` (integrated logger init/finalize)

**Ready to Run:** YES ✅

---

## Next Steps

1. **Run the build:**
   ```bash
   cd /home/vince/Projects/llama-orch/bin/worker-orcd
   cargo clean && cargo build --release --features cuda
   ```

2. **Execute our engine:**
   ```bash
   ./investigation-teams/TEAM_PRINTER_PARITY/run_our_engine.sh
   ```

3. **Convert checkpoints:**
   ```bash
   cd investigation-teams/TEAM_PRINTER_PARITY
   python3 convert_to_npz.py ours.checkpoints.manifest.json
   ```

4. **Run llama.cpp:**
   ```bash
   ./run_llamacpp.sh
   ```

5. **Compare and document findings** in `diff_report.md`

6. **Append summary to** `../INVESTIGATION_CHRONICLE.md`

---

**TEAM PRINTER**  
**Mission:** Data collection only — find the first divergence, don't fix it.  
**Status:** 🟢 GO — All systems ready
