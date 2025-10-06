# 🧠 BUG HUNT MISSION — TEAM <YOUR TEAM NAME>

Welcome, Team <YOUR TEAM NAME> 👩‍🚀👨‍🚀
A sneaky bug has been hidden somewhere in the codebase.
Your mission: Find it, test your fix, and leave a clear investigative trail so the next team can follow your work.

---

## 📝 Rules of Engagement

### 1) Comment Every Suspicion
- Add comments directly in the code whenever you suspect a bug.
- Explain why a spot is suspicious and what you think might be wrong.
- Use clear, investigative notes for future teams.

### 2) Test Before You Claim
- Do NOT claim “FIXED” unless you have actually tested your fix and verified that the Haiku Test passes.
- Premature “fixed” claims mislead the next teams. Be honest and disciplined.

### 3) Update or Correct Previous Team’s Comments
- If a previous team claimed to have fixed the bug, but your testing shows the bug is still there, you must update their comment to mark it as a false fix and add your evidence.

Example update to their comment (append a new line below theirs):
```cpp
// ❌ Previous team claimed FIXED — but haiku test still fails here. Suspect race condition remains.
// Evidence: Haiku test output on 2025-10-06 18:44 UTC still repeats token ID 64362 at steps 2-9.
```

This ensures false claims don’t mislead future teams.

### 4) Edit Your Own Comments as You Go
- If a suspicion turns out wrong → update to say ❌ False lead + why.
- If it’s right → mark clearly as ✅ FOUND BUG and explain the root cause and the fix.

### 5) Leave a Clear Trail
- Your job isn’t just to fix — it’s to document the investigation in the code.
- Think like detectives leaving case notes: what you suspected, what you tested, what you observed, and what you concluded.

---

## 🧭 Objective
- Locate the bug.
- Test and confirm the fix.
- Pass the Haiku Test 🟢.
- Correct any false claims from previous teams.
- Leave a readable trail of reasoning in the code for the next team.

---

## 💡 Tips
Use clear markers like:
```cpp
// 🕵️ Suspicion: This looks like an off-by-one issue in loop
// ❌ False lead: Confirmed with test_case_X, works as intended
// ❌ Previous team claimed FIXED — test still fails here
// ✅ FOUND BUG: Null pointer not checked here; fixed by adding guard
```

Over-comment rather than under-comment.
Be honest: only mark FIXED when the test actually passes.

---

## 🧪 Haiku Test
Run from `bin/worker-orcd/`:
```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```
Save your terminal output and reference timestamps and token IDs in your comments.

---

## 🏁 Victory Condition
- ✅ The Haiku Test passes
- 📝 All investigative notes are left in the code, including any corrections of previous teams’ false claims

Good luck, Team <YOUR TEAM NAME>.
May your code comments expose every lie 👀 and uncover the truth 🕵️‍♀️✨
