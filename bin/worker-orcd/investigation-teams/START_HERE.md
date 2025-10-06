# 🚀 START HERE - Investigation Teams

**Welcome!** You're part of a 5-team investigation to solve the repetitive token bug.

---

## Read These Files In Order

### 1️⃣ First: `DEPLOYMENT_SUMMARY.md` (5 min)
Get the big picture - what are we doing and why?

### 2️⃣ Second: `QUICK_START_GUIDE.md` (10 min)
Learn HOW to investigate - code examples, test patterns, tips.

### 3️⃣ Third: Your team's brief (15 min)
- Team Alpha: `TEAM_ALPHA_MEMORY_FORENSICS.md`
- Team Bravo: `TEAM_BRAVO_REFERENCE_COMPARISON.md`
- Team Charlie: `TEAM_CHARLIE_MANUAL_VERIFICATION.md`
- Team Delta: `TEAM_DELTA_INSTRUMENTATION.md`
- Team Echo: `TEAM_ECHO_FIRST_PRINCIPLES.md`

### 4️⃣ Fourth: `README.md` (reference as needed)
Master coordination document with all the details.

---

## The One Thing You Need to Know

**You CAN change code to extract data!**

Add logging, implement tests, copy GPU data to host, try different parameters - whatever you need to gather EVIDENCE. Just **revert your changes** after you've collected the data.

---

## Your Mission

1. **Gather evidence** - Run tests, extract data, see what's actually happening
2. **Understand the bug** - Why do positions 8850, 44394, 137131 have garbage?
3. **Prove your theory** - Show the data that supports your conclusion
4. **Propose a fix** - Specific code change with justification
5. **Document everything** - Create `TEAM_YOUR_NAME_RESULTS.md`

---

## Quick Test Command

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Clean build
cargo clean -p worker-orcd

# Run test (save output)
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1 \
  2>&1 | tee my_team_output.txt
```

---

## Need Help?

- **How do I add tests?** → See `QUICK_START_GUIDE.md`
- **What's my team's approach?** → See your `TEAM_*_*.md` file
- **What files do I edit?** → Usually `cuda/src/transformer/qwen_transformer.cpp`
- **How do I revert changes?** → `git checkout <filename>` or manually remove test code

---

## Success Looks Like

Your `TEAM_*_RESULTS.md` file contains:
- ✅ **Test output** showing actual runtime data
- ✅ **Analysis** of what the data means
- ✅ **Root cause** explanation with evidence
- ✅ **Proposed fix** with justification
- ✅ **Verification** that your theory is correct

---

## Let's Go! 🔍

Read the docs above, add your investigation code, run tests, gather data, and solve this bug!

**Remember**: Evidence over guesses. Data over assumptions. Prove it!
