# 🔎 Collaborative Bug & Inconsistency Hunt — Team GEMMA DELTA

**Mission:** Find and fix **defects in code** *or* **contradictions between documents/specs/tests/configs**.
**Win Condition:** The **Haiku test passes** *and* your investigation is **fully written into the code as comments**, so the next team doesn’t re-think what you already explored.

---

## 🎯 Scope (choose what applies)

* **Code Bugs:** Logic, off-by-one, races, unwraps, lifetimes, FFI, build flags.
* **Spec/Doc Contradictions:** Requirements vs code, tests vs behavior, README vs defaults, ADR vs implementation, CLI help vs flags.
* **Config/Env Mismatches:** Feature flags, profiles, CUDA/CPU switches, targets, paths, CI matrix vs local scripts.

> Always tie contradictions to a **check** (test, build, lint, schema/contract check).

---

## 🧪 Verification Standard

* **Primary:**

  ```bash
  REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
    --test haiku_generation_anti_cheat \
    test_haiku_generation_stub_pipeline_only \
    -- --ignored --nocapture --test-threads=1
  ```

  Only claim **FIXED** if this **passes**.

* **Do NOT use the interactive CLI with `head`/`tail`.**
  `llama-cli` is conversational and blocks; piping output is unreliable.

* **Use the HTTP API for runtime checks instead of the CLI.**
  Example (adjust to your actual endpoint/model):

  ```bash
  curl -sS -X POST {HTTP_API_URL}/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
      "model": "{MODEL_NAME}",
      "messages": [
        {"role":"system","content":"You are a helpful assistant"},
        {"role":"user","content":"Write a haiku about autumn:"}
      ],
      "temperature": 0.7,
      "max_tokens": 50
    }'
  ```

  Use this **only** to confirm behavior once your reasoning is documented.

* **If you “fix” a doc/spec inconsistency**, show the **assertion** that proves alignment (updated test or exact command output).

---

## 🗒️ Comment-Driven Forensics (MANDATORY)

**Write what you think, where you think it.**
Every time you form a hypothesis while reading a line, **leave a comment right there**. Then **edit the same comment** as you learn more. Make your thinking visible so the next team never re-thinks the same thing.

Use these markers (exact words):

* `SUSPECT:` why this might be wrong
* `THOUGHT:` immediate reasoning or question you’re exploring
* `TRACE:` observed values/logs relevant to this spot
* `CONTRADICTION:` what disagrees with what (A vs B, cite lines/sections)
* `FALSE_LEAD:` why it’s actually fine (how you verified)
* `FIXED:` code defect fixed and verified (reference passing test/commit)
* `RESOLVED:` alignment change + how it’s enforced (test/command)
* `FALSE_FIX:` prior team claimed fixed; your verification shows failing

**Examples**

```rust
// SUSPECT: Off-by-one: expecting exactly 3 lines in haiku formatter.
// THOUGHT: Is trailing newline causing a 4th line after split('\n')?
// TRACE: observed output lines = 4 (last is ""), input endswith('\n') = true
// CONTRADICTION: README says "no trailing newline", tests tolerate 1.
// FALSE_LEAD: Adjusting split logic did not change test failure; see run log below.
// FIXED: Trim trailing newline in render(); test now green (cmd in Verification Standard).
```

```cpp
// SUSPECT: BOS token inserted unconditionally here; may shift structure.
// THOUGHT: Qwen’s chat template might add BOS already—double BOS?
// CONTRADICTION: Spec §4.1 says conditional BOS; code always inserts.
// FALSE_FIX: Previous team marked FIXED at line 88; re-run still fails.
// RESOLVED: Insert BOS only if tokenizer requires; verified by HTTP API roundtrip.
```

```yaml
# SUSPECT: batch_size=4 conflicts with spec §M0 (batch=1)
# THOUGHT: CI matrix sets BATCH=1 but local default is 4 → mismatch
# RESOLVED: Set batch_size=1 here; CI + local align; test passes.
```

```md
<!-- CONTRADICTION: CLI help suggests using -p for system, but llama.cpp expects --system-prompt. -->
<!-- RESOLVED: Docs updated; examples use --system-prompt. -->
```

---

## 🔁 Correcting Other Teams’ Premature Claims

If a previous comment says **FIXED** but your verification **fails**:

* **Do not delete** their comment. **Append** in place:

```
// FALSE_FIX: Haiku still failing after this change; see failing cmd/output below.

```
- Add your own `SUSPECT` / `CONTRADICTION` nearby with your new lead.

---

## 🧭 Minimal Workflow (follow in order)

1. **Alignment Pass (quick):** Decide bug vs contradiction (or both); list files.
2. **Baseline:** Run the **Primary** command once; paste a **short failing excerpt** under the nearest relevant comment.
3. **Read & Write:** As you read lines, **leave `THOUGHT` and `SUSPECT`** notes inline—don’t keep ideas in your head.
4. **Instrument & Narrow:** Add assertions/logs; capture `TRACE` snippets where it matters.
5. **Fix / Align (smallest change):**
 - Code: minimal, targeted change.
 - Docs/Specs: update the authoritative source and add/adjust a test.
6. **Verify:** Re-run the **Primary** command (and HTTP API if needed). Only then mark `FIXED`/`RESOLVED`.
7. **Police Claims:** If you debunk a prior fix, add `FALSE_FIX` with proof.

---

## 🚫 Don’t Blame the Model (and don’t pipe the CLI)

Interactive CLI example (do **not** pipe with `head`/`tail`):
```

/home/vince/Projects/llama-orch/reference/llama.cpp/build/bin/llama-cli 
-m /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf 
-p "Write a haiku about autumn:" -n 50 --temp 0.7

```

Use the **HTTP API** instead for reproducible, non-interactive checks (see Verification Standard).

---

### Final reminder
> If you **thought** about it, **write it down in the code**.  
> Future teams must never have to re-think the same line you just studied.
