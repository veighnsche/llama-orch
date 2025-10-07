# Parity Artifacts - TEAM PICASSO

**Generated:** 2025-10-07T16:36Z  
**Purpose:** Numeric comparison between llama.cpp (ground truth) and worker-orcd

---

## Status

### ✅ llama.cpp Ground Truth
- **File:** `llama_hidden_states.jsonl` (14 entries)
- **Command:**
  ```bash
  cd reference/llama.cpp/build
  ORCH_LOG_FILE="$PWD/llama_hidden_states.jsonl" \
  ORCH_LOG_TEAM="PICASSO-LLAMA" \
  ORCH_LOG_VALUES=10 \
  ./bin/llama-cli \
    -m ../../../.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
    -p "Write a haiku about GPU computing" \
    -n 15 --temp 0.7 --top-k 0 --top-p 1.0 -no-cnv \
    </dev/null > llama_output.log 2>&1
  ```
- **Result:** ✅ SUCCESS - 14 logit entries generated

### ⚠️ worker-orcd Output
- **Status:** BLOCKED - Test infrastructure issue
- **Issue:** HTTP connection failure prevents test from reaching generation loop
- **Root cause:** Test tries to start HTTP server which fails
- **Workaround needed:** Direct inference test without HTTP layer

---

## Files

### Generated
- `llama_hidden_states.jsonl` - llama.cpp ground truth (copied from build dir)
- `llama_output.log` - llama.cpp stdout/stderr
- `compare_parity.py` - Comparison script (ready to use once both JSONLs exist)

### Pending
- `our_hidden_states.jsonl` - worker-orcd output (blocked on test infrastructure)
- `parity_report.csv` - Comparison results (pending both inputs)

---

## Sample JSONL (llama.cpp)

```json
{
  "checkpoint": "logits",
  "team": "PICASSO-LLAMA",
  "token_idx": 7,
  "dtype": "f32",
  "shape": "[151936]",
  "values": [0.0, 0.0, -1.01e+18, ...]
}
```

---

## Next Steps

1. **Fix test infrastructure** - Bypass HTTP layer or fix connection issue
2. **Generate worker-orcd JSONL** - Run inference directly
3. **Run comparison** - `python3 compare_parity.py > parity_report.csv`
4. **Analyze differences** - Identify first divergence point

---

**TEAM PICASSO**  
*Evidence-based debugging*
