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

COMPLETE THE ACTUAL TODO LIST RULE

* ✅ ALWAYS read the "Next Steps" section from the previous handoff
* ✅ ALWAYS complete ALL priorities in order (Priority 1, 2, 3, etc.)
* ❌ NEVER do just Priority 1 and then write a new handoff
* ❌ NEVER ignore the existing TODO list and make up your own priorities
* ❌ NEVER invent new work items that derail the plan
* ✅ If you finish Priority 1, immediately start Priority 2
* ✅ If you finish Priority 2, immediately start Priority 3
* ✅ Only hand off when ALL priorities are complete
* ✅ If you discover bugs during work, fix them AND continue with the plan
* ❌ NEVER use "discovered a bug" as an excuse to abandon the TODO list

Example of BAD behavior (what TEAM-013 actually did):
- TEAM-012 handoff: "Priority 1: CUDA, Priority 2: Sampling, Priority 3: Production"
- TEAM-013 does: Priority 1 only
- TEAM-013 writes: "Priority 1: GPU Warmup, Priority 2: SSE, Priority 3: Multi-GPU"
- Result: Completely derailed! Priorities 2 & 3 ignored, new random work invented

Example of GOOD behavior:
- TEAM-012 handoff: "Priority 1: CUDA, Priority 2: Sampling, Priority 3: Production"
- TEAM-013 does: ALL THREE priorities
- TEAM-013 writes: "✅ All priorities complete: CUDA validated, sampling optimized, production ready"
- Result: Work actually progresses, no derailment

The handoff TODO list is THE PLAN. Follow it. Don't make up your own plan.