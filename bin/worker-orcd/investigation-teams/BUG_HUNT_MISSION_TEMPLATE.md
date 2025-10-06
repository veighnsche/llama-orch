# 🔎 Collaborative Bug & Inconsistency Hunt — Team FELICIA

**Mission:** Hunt down the **garbage output** bug.
This is **not** an infra/pipeline issue. Assume weights/VRAM/kernels/IPC are fine unless proven otherwise.
Your job is to **trace and fix any model-logic, tokenization, or other code issues** that produce unreadable output.
Investigate wherever the evidence leads — just follow the hard rules below and document your thought process in code.

---

## 🚦 Hard Rules

1. **APPEND-ONLY COMMENTS**

   * Never overwrite or delete previous teams’ comments.
   * Always **append** your notes **beneath** the existing ones.

2. **COMMENT EVERYTHING YOU THINK**

   * Every hypothesis, hunch, or “maybe it’s this line” must be written **right there in the code**.
   * Write what you’re thinking, how you plan to check it, and what you find.
   * The next team should *never* have to re-think something you already considered.

3. **CORRECT FALSE CLAIMS WITHOUT EDITING THEM**

   * If a previous team claimed “FIXED” and it’s wrong, **append** `FALSE_FIX:` under their comment with proof (logs/test output).
   * Do not rewrite their text.

4. **NO CLI PIPING / INTERACTIVE SESSIONS**

   * Stop doing this:

     ```bash
     ./build/bin/llama-cli -m ... -p "test" | grep ... | head
     ```

     It hangs because the CLI is **interactive**. You end up in a conversation loop waiting for user input.
     → **Use the HTTP API**, not CLI pipes.

5. **BLOCKING TESTS ONLY (FOREGROUND)**

   * Run tests in the **foreground**. No background jobs, no detached sessions. You’ll lose your logs and conclusions.

---

## 🧪 Verification

**Primary (blocking):**

```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

* Only mark `FIXED` if this passes and the generated text is human-readable (no mojibake / control glyphs).

**HTTP probe (not CLI):**

```bash
curl -sS -X POST http://localhost:{PORT}/execute \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Write a haiku about GPU computing that includes the word \"forty-two\" (nonce: TEST)",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Capture the exact response text under `OBSERVED:` comments when relevant.

---

## 🗒️ Comment Protocol (append-only)

Tag with team + UTC time. Use these markers exactly:

* `SUSPECT:` your hypothesis + why
* `PLAN:` how you’ll verify
* `OBSERVED:` relevant logs/output
* `FALSE_LEAD:` why this wasn’t the issue (and proof)
* `CONTRADICTION:` spec/test/docs vs actual behavior
* `FIXED:` what you changed + one proof line (green test / human text)
* `FALSE_FIX:` prior fix claim disproved, with evidence

Never delete or overwrite someone else’s work.

---

## 🔥 Angry PSA

# 🚫 Stop doing `llama-cli | grep | head` — it. does. not. work.

This garbage:

```bash
./build/bin/llama-cli \
  -m /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "test" -n 1 --log-disable 2>&1 | grep -E "(im_start|im_end|bos|eos)" | head -20
```

**Does. Not. Work.**
`llama-cli` starts an **interactive chat session**. Your pipe hangs waiting for a “user” reply, nothing flows to `grep`, `head` blocks forever, then you `Ctrl+C` and declare “broken.” You also disable the logs you’re grepping for. This is user error, not model error.

👉 Use the **HTTP API** for deterministic probe runs. Or add a proper debug print before tokenization.

---

## ✅ Definition of Done

* Garbage output is gone → output is human-readable.
* Primary test passes in the foreground.
* Your **append-only** comment trail shows what you investigated, how you tested it, and what you found.
* Any false “FIXED” claims are corrected with `FALSE_FIX:` and proof.
* You didn’t overwrite anyone’s work.

---

No scope narrowing. No artificial limits. Just rules, discipline, and a very clear understanding:
👉 **Bug = garbage output.**
👉 **Fix it, document your thought process, don’t erase history.**
