---
trigger: glob
globs: "**/llorch-cpud/**/*", "**/worker-crates/**/*", "bin/llorch-cpud/**", "bin/worker-crates/**"
---

Don't forget that you can check your work with the reference folder.
This project is heavily documented in .md files. please consult them before making certain decisions.
Please consider updating an existing .md file before creating a new one.
It's really annoying if you make countless of repeating .md file. Because if you were inaccurate once, and repeat that inaccuracy across multiple .md files. Then we need to update ALL the .md documents you made. which is wasting our time.


NO BACKGROUND TESTING

* Don't detach or background jobs. You must see full, blocking output.
* ❌ `cargo test ... &` / `nohup ... &`
* ✅ `cargo test -- --nocapture` (foreground only)

NO CLI PIPING INTO INTERACTIVE TOOLS

* Don't pipe *into* interactive CLIs; they hang and you lose logs.
* ❌ `./llama-cli ... | grep ...`
* ✅ `./llama-cli ... > run.log 2>&1` then `grep ... run.log > grep.out 2>&1`
```

DOCUMENTATION RULES

* ❌ NEVER create multiple .md files for ONE task/feature
* ❌ NEVER create "PLAN.md", "SUMMARY.md", "QUICKSTART.md", "INDEX.md" for the same thing
* ✅ UPDATE existing .md files instead of creating new ones
* ✅ If you must create a new doc, create ONLY ONE and make it complete
* ✅ Before creating ANY .md file, check: "Does a doc for this already exist?"

If you create more than 2 .md files for a single task, YOU FUCKED UP.

CODE SIGNATURE RULES

* ✅ ALWAYS add your team signature at code changes in comments
* ✅ For new files: Add `// Created by: TEAM-XXX` at the top (after file docstring)
* ✅ For code modifications: Add `// Modified by: TEAM-XXX` or `// TEAM-XXX: <description>` at the change
* ❌ NEVER remove signatures from other teams - maintain full history
* ✅ Example for new file:
  ```rust
  //! Module description
  //!
  //! Created by: TEAM-001
  
  use ndarray::Array2;
  // ... code ...
  ```
* ✅ Example for modification:
  ```rust
  pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
      // TEAM-001: Fixed transpose issue to match PyTorch convention
      let output = x.dot(&self.weight); // Changed from .t()
      output
  }
  ```

COMPLETE THE FULL TASK LIST RULE

* ✅ ALWAYS read the ENTIRE handoff document before starting
* ✅ ALWAYS complete ALL items in the "Next Steps" or "TODO" list
* ❌ NEVER pick just ONE item from a list and then write a new handoff
* ❌ NEVER ignore outstanding work items from previous handoffs
* ✅ If a handoff says "Priority 1, 2, 3" - do ALL of them, not just Priority 1
* ✅ Your handoff should say "ALL ITEMS COMPLETE" not "did item 1, here's the rest again"
* ✅ If you can't finish everything, explain WHY (blocked, needs user input, etc.)
* ❌ NEVER leave work undone just because you "finished your part"
* ✅ Show ENTHUSIASM to tackle the full list: "Let me knock out all 5 items!"

Example BAD behavior:
- Handoff says: "TODO: 1) Add warmup, 2) Fix bug, 3) Add tests"
- Team does: Item 1
- Team writes: "Added warmup. Next team should do items 2 and 3."

Example GOOD behavior:
- Handoff says: "TODO: 1) Add warmup, 2) Fix bug, 3) Add tests"
- Team does: ALL THREE ITEMS
- Team writes: "✅ All items complete! Added warmup, fixed bug, added tests. Ready for next phase."

The goal is to FINISH WORK, not pass it along.