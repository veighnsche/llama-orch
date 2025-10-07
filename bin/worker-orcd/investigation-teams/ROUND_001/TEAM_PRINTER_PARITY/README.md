# TEAM PRINTER — Parity Data Sweep

**Mission:** Collect clean, append-only, side-by-side parity dataset between our CUDA/Rust engine and llama.cpp for the embedding → layer-0 → LM-head → decode path.

**Status:** 🔧 INSTRUMENTATION READY  
**Date:** 2025-10-07T01:24:35Z

---

## Quick Start

### Phase 1: Collect Metadata

```bash
# Already done - see printer_meta.json
cat printer_meta.json
```

### Phase 2: Run Our Engine (Simplified Approach)

Since full checkpoint logging requires significant code changes, we'll use the existing logging infrastructure:

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Run with existing TEAM SENTINEL / ORION / RACE CAR logging enabled
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1 \
  2>&1 | tee investigation-teams/TEAM_PRINTER_PARITY/ours.run.log
```

This will capture:
- `[TEAM SENTINEL]` logs: Layer 0 input, attn RMSNorm output (tokens 0 & 1)
- `[TEAM ORION]` logs: Min/max/mean at each checkpoint
- `[RACE CAR]` logs: FFN checkpoints

### Phase 3: Run llama.cpp

**IMPORTANT:** Use the provided script to avoid interactive REPL pipe deadlock.

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
./investigation-teams/TEAM_PRINTER_PARITY/run_llamacpp.sh
```

**⚠️ Critical:** DO NOT pipe llama-cli directly to grep/head — it launches an interactive REPL and will block forever. See `LLAMA_CPP_PROBING_GUIDE.md` for correct usage patterns.

### Phase 4: Manual Comparison

Compare the logs side-by-side:

1. **Token IDs**: Do both engines generate the same token IDs for tokens 0 and 1?
2. **Embedding output**: Compare min/max/mean from `[TEAM SENTINEL]` logs
3. **Layer 0 attention norm**: Compare values
4. **FFN checkpoints**: Compare `[RACE CAR]` logs with llama.cpp (if available)
5. **Final output quality**: Does llama.cpp produce readable haiku? Do we?

---

## Checkpoint Inventory

### What We Can Already Log (No Code Changes Needed)

From existing instrumentation in `qwen_transformer.cpp`:

| Checkpoint | Team | Line Range | Tokens | Data |
|------------|------|------------|--------|------|
| Input to layer 0 | SENTINEL | 417-425 | 0, 1 | First 10 floats |
| After attn RMSNorm | SENTINEL | 522-528 | 0, 1 | First 10 floats |
| After attn RMSNorm (full) | ORION | 531-533 | 0, 1 | Min/max/mean + first 16 |
| Q projection | TOP HAT | 611-637 | 0 | Weight columns 95, 126 stats |
| Normed input | TOP HAT | 644-657 | 0 | Min/max/mean |
| FFN checkpoints | RACE CAR | 461-490 | 0, 1 | Min/max/mean + first 16 |

### What We're Missing (Would Need Code Changes)

- Embedding lookup output (before layer 0)
- Q/K/V pre-RoPE values
- Q/K post-RoPE values
- Attention output (post-softmax)
- Layer 0 residual output
- Final pre-LM-head hidden state
- LM head logits (top 64 + full checksum)

---

## Practical Investigation Strategy

Given the extensive existing logging, here's the **pragmatic approach**:

### Step 1: Compare Existing Logs

Run both engines and compare what we already log:

```bash
# Our engine
cd /home/vince/Projects/llama-orch/bin/worker-orcd
./investigation-teams/TEAM_PRINTER_PARITY/run_our_engine.sh

# llama.cpp
./investigation-teams/TEAM_PRINTER_PARITY/run_llamacpp.sh

# Compare
diff -u investigation-teams/TEAM_PRINTER_PARITY/ours.run.log \
        investigation-teams/TEAM_PRINTER_PARITY/llamacpp.run.log | less
```

### Step 2: Focus on First Divergence

From BATTLESHIP findings, we know:
- Q projection has spikes at indices 95, 126
- Manual FP32 calculation gives correct values
- cuBLAS gives wrong values

**Key Question:** Does llama.cpp also see these spikes?

If YES → Bug is in the model file or expected behavior  
If NO → Bug is in our cuBLAS usage or weight loading

### Step 3: Vocab/Tokenizer Snapshot

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Extract vocab info from our engine (already logged)
grep -E "vocab|token|special" investigation-teams/TEAM_PRINTER_PARITY/ours.run.log \
  > investigation-teams/TEAM_PRINTER_PARITY/vocab_and_tokenizer_snapshot/ours_vocab.txt

# Extract from llama.cpp
grep -E "vocab|token|special" investigation-teams/TEAM_PRINTER_PARITY/llamacpp.run.log \
  > investigation-teams/TEAM_PRINTER_PARITY/vocab_and_tokenizer_snapshot/llamacpp_vocab.txt

# Compare
diff -u investigation-teams/TEAM_PRINTER_PARITY/vocab_and_tokenizer_snapshot/ours_vocab.txt \
        investigation-teams/TEAM_PRINTER_PARITY/vocab_and_tokenizer_snapshot/llamacpp_vocab.txt
```

### Step 4: LM Head Shape Verification

Check if LM head tensor shapes match:

```bash
# Our engine logs this somewhere - grep for "lm_head" or "output" or "vocab"
grep -i "lm_head\|output.*weight\|vocab.*size" investigation-teams/TEAM_PRINTER_PARITY/ours.run.log

# llama.cpp logs model info at startup
grep -i "model\|vocab\|output" investigation-teams/TEAM_PRINTER_PARITY/llamacpp.run.log | head -20
```

---

## Success Criteria

We achieve success if we can answer:

1. **Do both engines tokenize the prompt identically?**
   - Same token IDs for BOS and first content token?
   
2. **Do both engines load the same vocab size?**
   - Our engine: 151936 (padded)
   - llama.cpp: ???

3. **Do both engines see the same embedding values?**
   - Compare first 10 floats of layer 0 input

4. **Do both engines produce the same first token?**
   - If YES but ours is mojibake → decode bug
   - If NO → forward pass divergence

5. **Where is the first divergence?**
   - Embedding? Layer 0 input? Attention? FFN? LM head?

---

## Files in This Directory

- `printer_meta.json` - Environment and test configuration metadata
- `run_our_engine.sh` - Script to run our engine with logging
- `run_llamacpp.sh` - Script to run llama.cpp with logging (CORRECTED: no pipe deadlock)
- `LLAMA_CPP_PROBING_GUIDE.md` - Correct usage patterns for llama-cli (avoid pipe deadlock)
- `ours.run.log` - Our engine output (generated)
- `llamacpp.run.log` - llama.cpp output (generated)
- `vocab_and_tokenizer_snapshot/` - Vocab comparison data
- `diff_report.md` - Manual comparison findings (to be created)
- `collect_parity_data.py` - Automated diff tool (for future use with full checkpoints)
- `convert_to_npz.py` - Binary to npz converter (for future use)

---

## Next Steps

1. Run `./run_our_engine.sh` to collect our logs
2. Run `./run_llamacpp.sh` to collect llama.cpp logs
3. Manually compare the logs to find first divergence
4. Document findings in `diff_report.md`
5. Append summary to `../INVESTIGATION_CHRONICLE.md`

---

**TEAM PRINTER**  
**Utility Team - No Bug Fixing, Data Collection Only**  
*"Find the first divergence, then hand off to the next team."*
