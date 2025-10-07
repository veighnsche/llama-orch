# llama.cpp Probing Guide — Correct Usage

**Date:** 2025-10-07T01:55Z  
**Team:** TEAM PRINTER

---

## ❌ Common Mistake: Interactive Pipe Deadlock

**DO NOT DO THIS:**

```bash
./build/bin/llama-cli \
  -m model.gguf \
  -p "test" \
  -n 1 \
  --log-disable \
  2>&1 | grep -E "(im_start|im_end|bos|eos)" | head -20
```

**Why This Fails:**
1. `llama-cli` launches an **interactive REPL**, not a batch process
2. Piping to `grep | head` blocks waiting for user input that never comes
3. `--log-disable` silences the very strings you're trying to match
4. The pipe stalls forever — not a model issue, tool misuse

---

## ✅ Correct Approach 1: Redirect Then Grep

**Redirect output to file, then grep the file:**

```bash
# Run with output redirected to file (non-interactive)
./build/bin/llama-cli \
  --model "$MODEL_PATH" \
  --prompt "test prompt" \
  --n-predict 10 \
  --temp 0.0 \
  --seed 12345 \
  --verbose-prompt \
  > llamacpp_output.log 2>&1

# Now grep the file (not the running process)
grep -E "(im_start|im_end|bos|eos)" llamacpp_output.log | head -20
grep -i "token" llamacpp_output.log | head -30
tail -50 llamacpp_output.log
```

**Why This Works:**
- Direct redirect (`>`) avoids interactive mode
- File is fully written before grep runs
- No pipe deadlock
- Can grep multiple times without re-running

---

## ✅ Correct Approach 2: HTTP API (Batch Mode)

**Use llama-server for pure batch inference:**

```bash
# Start server
./build/bin/llama-server -m model.gguf --port 8080

# Send batch request
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "test prompt",
    "n_predict": 10,
    "temperature": 0.0,
    "seed": 12345
  }' | jq .
```

**Advantages:**
- No interactive REPL
- JSON output (easy to parse)
- Can script multiple requests
- No pipe issues

---

## ✅ Correct Approach 3: Source Patching

**For internal tensor values, modify llama.cpp source:**

```cpp
// In llama.cpp source (e.g., llama.cpp:llama_decode)
fprintf(stderr, "[DEBUG] Token %d: embedding[0..5] = %.6f %.6f %.6f %.6f %.6f\n",
        token_id, emb[0], emb[1], emb[2], emb[3], emb[4]);
```

Then rebuild and run normally:

```bash
cmake --build build --config Release
./build/bin/llama-cli -m model.gguf -p "test" -n 10 > output.log 2>&1
grep "\[DEBUG\]" output.log
```

---

## ✅ Correct Approach 4: Debugger

**Use GDB to extract values without modifying source:**

```bash
# Set breakpoints at key functions
gdb --args ./build/bin/llama-cli -m model.gguf -p "test" -n 1

(gdb) break llama_decode
(gdb) run
(gdb) print token_id
(gdb) print embeddings[0]@10  # Print first 10 floats
```

---

## Summary: The Golden Rule

**Never pipe llama-cli to grep/head directly.**

Always:
1. Redirect to file first
2. Or use HTTP API
3. Or patch source code
4. Or use debugger

The interactive REPL will block forever on piped input.

---

## Example: Correct Tokenizer Probe

```bash
#!/bin/bash
set -euo pipefail

MODEL="/path/to/model.gguf"
OUTPUT="probe.log"

# Run llama-cli with redirect (not pipe)
./build/bin/llama-cli \
  --model "$MODEL" \
  --prompt "test" \
  --n-predict 5 \
  --temp 0.0 \
  --verbose-prompt \
  > "$OUTPUT" 2>&1

# Now analyze the file
echo "=== Vocabulary Info ==="
grep -E "vocab|special" "$OUTPUT" | head -10

echo ""
echo "=== Token IDs ==="
grep -i "token" "$OUTPUT" | head -20

echo ""
echo "=== Special Tokens ==="
grep -E "(im_start|im_end|bos|eos)" "$OUTPUT"

echo ""
echo "=== Generated Output ==="
tail -20 "$OUTPUT"
```

---

**TEAM PRINTER**  
**Corrected:** 2025-10-07T01:55Z  
*"Use the tool correctly, get correct results."*
