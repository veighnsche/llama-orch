# rbee: Pronunciation and Primary Use Case

**Date:** 2025-10-10

---

## Pronunciation

**rbee** is pronounced **"are-bee"** (like the letters R-B)

❌ NOT "erbee"  
❌ NOT "rubee"  
✅ **"are-bee"** (R-B)

---

## Primary Target Audience

### 🎯 THE MAIN GOAL

**Developers who build with AI but don't want to depend on big AI providers**

**THE FEAR:** You're building complex codebases with AI assistance (Claude, GPT-4). What if the AI provider changes their models, shuts down, or changes pricing? Your codebase becomes unmaintainable without AI. You've created a dependency you can't control.

**THE SOLUTION:** rbee provides an **OpenAI-compatible API** that lets you build your own AI infrastructure using **ALL your home network hardware**. Build AI coders from scratch with agentic API. Never depend on external providers again.

### How It Works

```bash
# 1. Start rbee on your homelab
rbee-keeper daemon start
rbee-keeper hive start --pool default
rbee-keeper worker start --gpu 0 --backend cuda

# 2. Configure Zed IDE to use rbee instead of OpenAI
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=your-rbee-token

# 3. Now Zed's AI agents run on YOUR GPUs!
# Zero API costs. Full control. Use ALL your computers' GPU power.
```

### Benefits

- ✅ **Independence** - Never depend on external providers again
- ✅ **Control** - Your models, your rules, never change without permission
- ✅ **Always available** - Your hardware, your uptime
- ✅ **Zero ongoing costs** - Electricity only (no API fees)
- ✅ **Full privacy** - Code never leaves your network
- ✅ **Use all your GPUs** - Across multiple computers
- ✅ **Drop-in replacement** - OpenAI-compatible API
- ✅ **Multi-backend** - CUDA, Metal, CPU

### Compatible Tools

rbee's OpenAI-compatible API works with:
- **Zed IDE** (AI agents)
- **Cursor IDE**
- **Continue.dev**
- **Any tool using OpenAI SDK**

---

## Key Messages for All Documents

When writing about rbee, always include:

1. **Pronunciation:** rbee (pronounced "are-bee")
2. **Primary use case:** Power Zed IDE's AI agents with your homelab GPUs
3. **Key feature:** OpenAI-compatible API = drop-in replacement
4. **Main benefit:** Zero API costs, use ALL your computers' GPU power

---

## Example Introductions

### Short Version
```
rbee (pronounced "are-bee"): Build your own AI infrastructure to escape 
dependency on big AI providers. Never depend on external providers again.
```

### Medium Version
```
rbee (pronounced "are-bee"): Developers building complex codebases with AI 
assistance fear provider dependency. What if the AI changes, shuts down, or 
changes pricing? Your code becomes unmaintainable. rbee lets you build your 
own AI infrastructure using ALL your home network hardware. Build AI coders 
from scratch with agentic API. llama-orch-utils provides TypeScript utilities 
for building AI agents. OpenAI-compatible = drop-in replacement. Independence 
from external providers.
```

### Long Version
```
rbee (pronounced "are-bee", formerly llama-orch): Escape dependency on big AI 
providers. THE PROBLEM: You're building complex codebases with AI assistance 
(Claude, GPT-4). What if they change models, shut down, or change pricing? 
Your codebase becomes unmaintainable without AI. You've created a dependency 
you can't control. THE SOLUTION: Build your own AI infrastructure using ALL 
your home network hardware. rbee provides OpenAI-compatible API + agentic API 
for building AI coders from scratch. llama-orch-utils: TypeScript library for 
building LLM pipelines and AI agents (file ops, LLM invocation, prompt 
management, response extraction). Your models, your rules, never change without 
permission. Always available. Zero ongoing costs (electricity only). Complete 
control. Full privacy. Conservative projections: Year 1 (€70K, 35 customers), 
Year 2 (€360K, 100 customers), Year 3 (€1M+, 200+ customers).
```

---

## Documents Updated

✅ **AI_DEVELOPMENT_STORY.md** - Added pronunciation, primary use case, OpenAI compatibility  
✅ **VIDEO_SCRIPTS.md** - Added pronunciation and Zed IDE use case to all scripts  
✅ **README.md** - Completely rewritten intro with primary use case front and center  

---

## Summary

**Pronunciation:** rbee (pronounced "are-bee")  
**Target Audience:** Developers who build with AI but fear provider dependency  
**The Fear:** Complex codebases become unmaintainable if provider changes/shuts down  
**The Solution:** Build your own AI infrastructure using home network hardware  
**Key Advantage:** 11 shared crates already built (saves 5 days)  
**30-Day Plan:** Detailed execution plan to first customer (€200 MRR)  
**Year 1 Goal:** 35 customers, €10K MRR, €70K revenue  
**Status:** 68% complete (42/62 BDD scenarios passing)

---

*Always emphasize: rbee (pronounced "are-bee") helps developers escape dependency on big AI providers* 🐝
