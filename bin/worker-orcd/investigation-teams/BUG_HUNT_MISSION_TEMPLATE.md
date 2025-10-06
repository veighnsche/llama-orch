# 🔎 Collaborative Bug & Inconsistency Hunt — Team SEA

**Mission:** Hunt down the **garbage output** bug.
This is **not** an infra/pipeline issue. Assume weights/VRAM/kernels/IPC are fine unless proven otherwise.
Your job is to **trace and fix model-logic or prompt/tokenization issues** that yield unreadable tokens.

---

## 🚦 Hard Rules (read first)

1. **APPEND-ONLY COMMENTS**

   * **Never overwrite or delete** any previous team’s comments.
   * Always **append** your notes **beneath** the prior comment block.

2. **COMMENT EVERYTHING YOU THINK**

   * Every suspicion you have while reading a line must be **written in the code right there**.
   * Don’t keep thoughts in your head; **write the hypothesis, how you’ll test it, and what you learned**.

3. **CORRECT FALSE CLAIMS WITHOUT EDITING THEM**

   * If a prior comment claims “FIXED” but it isn’t, **append** under it with `FALSE_FIX:` + proof.
   * Do **not** edit their text.

4. **NO CLI PIPING / INTERACTIVE SESSIONS**

   * Do **not** pipe `llama-cli` output into `head`/`tail`. The CLI session is interactive and will hang.
   * **Use the HTTP API** for probe runs (examples below).

5. **BLOCKING TESTS ONLY (FOREGROUND)**

   * Always run tests **in the foreground** (blocking). **No background jobs**, or you will lose logs.

---

## 🎯 Scope (narrow)

* **Prompt & tokenization path:** role templating, BOS/EOS, special tokens, whitespace/newlines, chat template selection, tokenizer/model vocab consistency.
* **Logits & projection correctness:** output_norm, final projection matrix, logits scaling/temperature, invalid byte-BPE handling, Unicode decoding.
* **Sampling step:** top-k/p/temperature, repetition penalties, bad stop sequences producing junk.
* **Explicitly out of scope:** pool manager, IPC, HTTP server boot, CUDA context wiring (unless you prove it corrupts logits).

---

## 🧪 Verification (blocking, foreground)

**Primary (blocking):**

```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

Only mark **FIXED** if this **passes** and the output is **human-readable** (no mojibake / control glyph soup).

**Probe via HTTP API (not CLI):**

```bash
curl -sS -X POST http://localhost:{PORT}/execute \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Write a haiku about GPU computing that includes the word \"forty-two\" (nonce: TEST)",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Capture the **exact response text** in your comments when relevant.

---

## 🗒️ Comment Protocol (append-only)

Always tag with team + UTC time. **Never edit or delete earlier lines.**

Use exactly these markers:

* `SUSPECT:` your hypothesis and why (cite code path)
* `PLAN:` how you’ll verify (what log/add/assert you’ll add)
* `OBSERVED:` concrete output/logs from your probe/test
* `FALSE_LEAD:` why the suspicion was wrong (and proof)
* `CONTRADICTION:` spec/test/docs vs actual behavior
* `FIXED:` what you changed + the single proof line (test green / decoded text looks human)
* `FALSE_FIX:` prior team’s fix claim is wrong; attach proof (log excerpt)

**Rust example**

```rust
// [TEAM {TEAM_NAME}] 2025-10-06T17:15Z
// SUSPECT: Chat template inserts extra BOS and trailing newline → tokenizer emits junk byte-BPE early.
// PLAN: Log final rendered prompt before tokenization; compare to llama.cpp template text.
// OBSERVED: Rendered prompt ends with "<|im_start|>assistant\n\n" (double newline).
// CONTRADICTION: Our README claims single newline before assistant.
// FALSE_LEAD: Removing one newline did NOT fix mojibake; tokens still map to invalid UTF-8.
// SUSPECT: Tokenizer/vocab mismatch (Qwen chat template vs loaded GGUF tokenizer metadata).
```

**C++ example**

```cpp
// [TEAM {TEAM_NAME}] 2025-10-06T17:22Z
// SUSPECT: output_norm scale applied twice before final projection.
// PLAN: Dump pre/post norm ranges; compare to expected RMS ~[2..4] not exploding to ~40.
// OBSERVED: AFTER norm Std≈7.26 (too high), logits top-10 unstable across steps.
// FALSE_FIX: Prior team claimed "norm values OK"—HTTP output still mojibake; see curl proof below.
```

**Markdown (spec) example**

```md
<!-- [TEAM {TEAM_NAME}] 2025-10-06T17:30Z
SUSPECT: Spec §4.1 says BOS only if model requires; code path adds BOS unconditionally in chat mode.
OBSERVED: Token IDs show duplicate BOS at positions 0 and 24.
CONTRADICTION: GGUF tokenizer.chat_template expects single BOS.
-->
```

---

## 🔍 Minimal Workflow (what to actually do)

1. **Open the exact code path** that constructs the final prompt string (before tokenization).

   * Add a **temporary log** that dumps the **verbatim final prompt string**.
2. **Compare against llama.cpp’s chat template rules** (role order, start/end tags, BOS/EOS, newlines).

   * If you spot a discrepancy, **comment it immediately** (`SUSPECT` + `PLAN`), then check it.
3. **Trace tokenization**

   * Log the **first 50 token IDs** and attempt decoding to text; **paste** short snippets under `OBSERVED`.
4. **Trace final-projection & sampling**

   * Sanity-check logits scale (no wild spikes), temperature application, and decoding path.
5. **Test in foreground**

   * Run the primary test. If it’s still garbage, **append** `FALSE_FIX` under any earlier “fixed” claims and continue.
6. **When you truly fix it**

   * Append `FIXED:` with **one** proof line (green test or readable HTTP output). Do **not** erase any history.

---

## 🧰 Practical probes (safe & quick)

* **Render-only check (no gen):** add a mode/flag that prints the **final prompt string** + **token IDs** and exits.
* **Tokenizer identity check:** round-trip a small ASCII prompt; ensure IDs decode back to the same text.
* **Unicode sentinel:** include a known multi-byte char (e.g., “—”, “é”) and ensure decode survives.
* **BOS/EOS audit:** ensure exactly one BOS at start (if required by template) and controlled EOS insertion.

---

## ❗ Common traps (avoid wasting time)

* Don’t assume weights are wrong if llama.cpp runs the same GGUF fine.
* Don’t tweak CUDA, cuBLAS, or KV cache when the **symptom is mojibake text**.
* Don’t background tests; you’ll miss the logs you need.
* Don’t use `llama-cli | head`; **use the HTTP API** for deterministic captures.

---

## ✅ Definition of Done (for this mission only)

* The **garbage output stops**: generated text is readable and contains normal words.
* The **primary test passes in the foreground**.
* Your **append-only** comment trail captures your hypotheses, probes, and proof.
* Any **premature “FIXED”** claims above your section have a **`FALSE_FIX:` append** with concrete evidence.
