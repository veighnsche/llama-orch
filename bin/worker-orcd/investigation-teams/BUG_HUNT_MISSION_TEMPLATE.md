# 🔎 Collaborative Bug & Inconsistency Hunt — Team PROMPT

**Mission:** Find and fix **defects in code** *or* **contradictions between documents/specs/tests/configs**.
**Win Condition:** The **Haiku test passes** and your investigation trail is crystal clear for the next team.

---

## 🎯 Scope (choose what applies)

* **Code Bugs:** Logic, off-by-one, race conditions, null/None/unwrap, lifetimes, FFI boundaries, build flags, etc.
* **Spec/Doc Contradictions:** Requirements vs code, test names vs behaviors, README vs config defaults, ADR vs implementation, API contract vs handler, CLI help vs actual flags, comments vs code.
* **Config/Env Mismatches:** Feature flags, build profiles, CUDA/CPU switches, target triples, paths, CI matrix vs local script.

> Always tie contradictions to a **failing or meaningful check** (test, build, lint, schema validation, contract test).

---

## 🧪 Verification Standard

* **Primary:** Run **REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only -- --ignored --nocapture --test-threads=1** (the Haiku test). Only claim **FIXED** if it **passes**.
* **Secondary (if relevant):** (e.g., `cargo clippy -D warnings`, `cargo test -E contract`, schema checkers).
* If you “fix” a **doc/spec** inconsistency, show the **assertion** (test added/updated, or exact command output) that proves alignment.

---

## 🗒️ Comment-Driven Forensics (in every touched file)

Wherever you **suspect** a bug or contradiction, add a **comment** in place. Then **edit the same comment** as you learn more.

Use these exact markers:

* `SUSPECT:` why this might be wrong
* `FALSE_LEAD:` why it’s actually fine (include how you verified)
* `CONTRADICTION:` what disagrees with what (A vs B, include links/lines)
* `RESOLVED:` how you aligned and how it’s now enforced (test/command)
* `FIXED:` code defect fixed and verified (reference passing test/commit)
* `FALSE_FIX:` prior team claimed fixed; verification shows still failing

**Style by file type** (pick what applies in context):

```rust
// SUSPECT: Off-by-one in loop bound; haiku expects 3 lines exactly.
// FALSE_LEAD: Bound is correct; failure came from trailing newline in formatter.
// CONTRADICTION: README says "2 lines"; tests require 3.
// RESOLVED: Updated README §"Output format" to 3 lines; test haiku passes.
// FIXED: Trim newline in render(); `cargo test -p haiku -- --exact` now green.
// FALSE_FIX: Prior team said fixed in fmt.rs:42; test still red on CI job linux-x86_64.
```

```md
<!-- CONTRADICTION: Spec §2.1 says VRAM-only; Worker-AARMD supports UMA fallback. -->
<!-- RESOLVED: Marked VRAM-only as Worker-ORCD-specific; linked to AARMD exceptions. Verified by test: worker_aarmd_uma_smoketest. -->
```

```yaml
# SUSPECT: Default batch_size=4 conflicts with spec §M0 (batch=1).
# RESOLVED: Set batch_size=1; CI matrix and local run now consistent.
```

---

## 🔁 Correcting Other Teams’ Premature Claims

If a previous comment says **FIXED** but your verification **fails**:

* **Do not delete** their comment. **Append** a correction **in place**:

  * `FALSE_FIX: Haiku still failing after this spot; see run log below.`
  * Add the exact command + short failing output excerpt.
* Add your own `SUSPECT` / `CONTRADICTION` comment nearby if you have a new lead.

---

## 🧭 Minimal Workflow (follow in order)

1. **Alignment Pass (quick):** Identify what you’re hunting today (bug, contradiction, or both). Note the files of interest.
2. **Reproduce:** Run `REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only -- --ignored --nocapture --test-threads=1` to get a baseline failure. Paste a **short excerpt** under the nearest relevant comment.
3. **Instrument & Narrow:** Add asserts/logs, or cross-reference spec vs code. Leave `SUSPECT` comments in the exact lines.
4. **Fix / Align:**

   * Code: implement smallest viable change.
   * Docs/Specs: update the authoritative source; add or adjust a test to enforce the rule.
5. **Verify:** Rerun **REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only -- --ignored --nocapture --test-threads=1**. Only then mark `FIXED` / `RESOLVED`.
6. **Police Claims:** If a prior comment says `FIXED` but tests disagree, append `FALSE_FIX` with proof.
7. **Handoff Note (at end of file or PR description):** 3 bullets: What failed → What changed → What proves green.

---

## ✅ Handoff Checklist (keep it short)

* [ ] A failing log excerpt is attached near the relevant comment.
* [ ] Each suspicion has been updated to `FALSE_LEAD` or `FIXED`/`RESOLVED`.
* [ ] Any **premature FIXED** from prior teams is annotated with `FALSE_FIX` + proof.
* [ ] The **Haiku test passes**: `REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only -- --ignored --nocapture --test-threads=1` output captured once near the final change.
* [ ] If you changed a doc/spec, there’s a **test or check** that enforces it.

---

## Don't blame the model

 /home/vince/Projects/llama-orch/reference/llama.cpp/build/bin/llama-cli \
  -m /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7
