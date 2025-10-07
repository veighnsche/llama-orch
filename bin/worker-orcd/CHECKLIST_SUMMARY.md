# Bug Hunt Checklist — Quick Start Guide

**Generated:** 2025-10-07T08:20Z  
**Purpose:** Focused, de-duplicated checklist for hunting the garbage output bug

---

## 📁 What Was Created

### 1. **`Checklist.md`** (494 lines)
Comprehensive bug hunt checklist with:
- **Top 5 Highest Priority Suspects** — Start here!
- Full categorized checklist (22 items total)
- What NOT to re-investigate (18 verified-correct items)
- Recommended investigation sequence
- Verification commands and pass/fail criteria

### 2. **`logs/checklist_index.json`** (267 lines)
Machine-readable index with:
- All checklist items with metadata (file, line, bucket, confidence)
- Time estimates for each probe
- Investigation statistics
- Recommended sequence

### 3. **Inline Code Markers** (3 locations)
Append-only investigation markers added to code following the mission protocol:
- **`cuda/src/transformer/qwen_transformer.cpp:1845-1865`** — LM head projection probe
- **`cuda/kernels/rope.cu:377-403`** — RoPE numeric output probe
- **`cuda/src/transformer/qwen_transformer.cpp:2087-2111`** — Config verification probe

---

## 🎯 Quick Start — Where to Begin

### Phase 1: Quick Wins (30 minutes)

Run these three probes in order:

**1. Config Parameter Verification** (3 min)
```bash
# Enable the probe in qwen_transformer.cpp line 2100-2109 (uncomment the code)
# Build and run:
cd bin/worker-orcd
cargo build --release --features cuda
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1 | grep CONFIG_PROBE
```

**Expected output:**
```
[CONFIG_PROBE] num_heads=14 num_kv_heads=2 head_dim=64 hidden_dim=896
[CONFIG_PROBE] ffn_dim=4864 rope_freq_base=1000000.0 rms_norm_eps=0.000001000
```

**Pass criteria:** All values match llama.cpp (see `investigation-teams/TEAM_PRINTER_PARITY/llamacpp.run.log` lines 20-26, 69, 73)

---

**2. Tokenizer Decode Path** (5 min)
```rust
// Add in src/inference/cuda_backend.rs around line 580:
eprintln!("[TOKENIZER_PROBE] Token {} decoded as: {:?}", token_id, 
          self.tokenizer.decode(&[token_id])?);
```

Compare first 20 generated token IDs with llama.cpp output for same prompt.

---

**3. Embedding Scaling** (5 min)
```cpp
// Add in cuda/kernels/embedding.cu after line 145:
if (token_idx == 0 && threadIdx.x < 8 && blockIdx.x == 0) {
    printf("[EMBED_PROBE] Token 0 embed[%d]=%.6f\n", 
           threadIdx.x, __half2float(output[threadIdx.x]));
}
```

Compare with llama.cpp embedding values. If ours are ~30x different → missing `sqrt(896)` scaling.

---

### Phase 2: Deep Investigation (1 hour)

If Phase 1 doesn't find the bug:

**4. LM Head Projection Probe**
- See `Checklist.md` Top 5 #1
- Verify logits show peaked distribution (not flat)

**5. RoPE Numeric Output**
- See `Checklist.md` Top 5 #2
- Compare Q/K values after RoPE with llama.cpp

**6. Run TEAM PRINTER Parity**
- Already has infrastructure in `investigation-teams/TEAM_PRINTER_PARITY/`
- Systematic comparison of all checkpoints

---

## 📊 Key Findings from Chronicle Analysis

### ✅ Already Verified Correct (DO NOT RE-TEST)

Based on 15+ teams, 33+ hours, 26+ experiments:

1. **Tokenization** — Special tokens correctly inserted as single IDs
2. **Token Embeddings** — Valid FP16 values, correct lookup
3. **cuBLAS Q[0]** — Mathematically verified correct
4. **KV Cache** — Positions, reads, writes all correct
5. **Causal Masking** — Implemented correctly in kernel
6. **Sampling Order** — Fixed by HELIOS (softmax before top-p)
7. **FFN Pipeline** — All weights loaded, activations healthy
8. **Attention Filtering** — Q spikes filtered by softmax (BATTLESHIP)
9. **RMSNorm/SwiGLU Formulas** — Verified correct
10. **RoPE Formula** — Conceptually correct
11. ... and 8 more (see `Checklist.md` "DO NOT RE-INVESTIGATE" section)

### 🔍 Gaps Identified

**Highest Leverage Suspects:**

1. **LM Head Projection** — Last untested GEMM, all activations healthy upstream
2. **RoPE Numeric Output** — Formula verified but actual values never compared
3. **Config Mismatch** — Assumed correct, never explicitly verified
4. **Embedding Scaling** — May be missing `sqrt(hidden_dim)` factor
5. **Tokenizer Decode** — All internals healthy but output is garbage

---

## 📈 Investigation Statistics

**Synthesized from:**
- 50+ investigation documents
- 15 teams (BLUE, PURPLE, GREEN, WATER, CHARLIE, HYPERION, BATTLESHIP, RACE CAR, HELIOS, etc.)
- 26 major experiments
- 18 components verified correct
- 14 false leads documented
- 4 critical bugs found (3 fixed, 1 remains: the garbage output)

**Chronicle Location:** `investigation-teams/INVESTIGATION_CHRONICLE.md` (1128 lines)

---

## 🚦 Success Criteria

**Test passes when:**
```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

**Output contains:**
- ✅ Human-readable English haiku
- ✅ No mojibake (Éķ, âĪ¬, ĳľ, etc.)
- ✅ No code tokens (FileWriter, strcasecmp, etc.)
- ✅ Contains requested nonce word

---

## 🔗 Related Documents

- **`Checklist.md`** — Full detailed checklist (start here)
- **`logs/checklist_index.json`** — Machine-readable index
- **`investigation-teams/INVESTIGATION_CHRONICLE.md`** — Complete investigation history
- **`investigation-teams/BUG_HUNT_MISSION_TEMPLATE.md`** — Rules and protocols
- **`investigation-teams/FALSE_LEADS_SUMMARY.md`** — What NOT to retry

---

## 💡 Investigation Protocol Reminder

From `BUG_HUNT_MISSION_TEMPLATE.md`:

1. **APPEND-ONLY** — Never delete previous teams' comments
2. **COMMENT EVERYTHING** — Write what you're thinking in the code
3. **USE MARKERS** — `SUSPECT:` `PLAN:` `OBSERVED:` `FALSE_LEAD:` `FIXED:` `FALSE_FIX:`
4. **NO CLI PIPING** — Use HTTP API, not `llama-cli | grep`
5. **BLOCKING TESTS** — Run tests in foreground, not background

---

**Built by CHECKLIST_BUILDER**  
**Based on comprehensive codebase sweep + chronicle synthesis**  
**All investigation markers follow append-only protocol**
