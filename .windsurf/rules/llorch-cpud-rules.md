---
trigger: glob
globs: bin/llorch-cpud/**/*, bin/worker-crates/**/*
---

Don't forget that you can check your work with the reference folder.
This project is heavily documented in .md files. please consult them before making certain decisions.
Please consider updating an existing .md file before creating a new one.
It's really annoying if you make countless of repeating .md file. Because if you were inaccurate once, and repeat that inaccuracy across multiple .md files. Then we need to update ALL the .md documents you made. which is wasting our time.

NO BACKGROUND TESTING

* Don’t detach or background jobs. You must see full, blocking output.
* ❌ `cargo test ... &` / `nohup ... &`
* ✅ `cargo test -- --nocapture` (foreground only)

NO CLI PIPING INTO INTERACTIVE TOOLS

* Don’t pipe *into* interactive CLIs; they hang and you lose logs.
* ❌ `./llama-cli ... | grep ...`
* ✅ `./llama-cli ... > run.log 2>&1` then `grep ... run.log > grep.out 2>&1`