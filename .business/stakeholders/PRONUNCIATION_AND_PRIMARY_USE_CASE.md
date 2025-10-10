# rbee: Pronunciation and Primary Use Case

**Date:** 2025-10-10

---

## Pronunciation

**rbee** is pronounced **"are-bee"** (like the letters R-B)

❌ NOT "erbee"  
❌ NOT "rubee"  
✅ **"are-bee"** (R-B)

---

## Primary Use Case

### 🎯 THE MAIN GOAL

**Power Zed IDE's AI agents with your homelab GPUs**

rbee provides an **OpenAI-compatible API** that lets you use **ALL your computers' GPU power** for AI coding instead of paying for cloud APIs.

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

- ✅ **Zero API costs** - Your hardware, your control
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
rbee (pronounced "are-bee"): Power Zed IDE's AI agents with your homelab GPUs. 
OpenAI-compatible API = drop-in replacement.
```

### Medium Version
```
rbee (pronounced "are-bee") is an OpenAI-compatible AI orchestration platform 
that lets you use ALL your computers' GPU power for AI coding. Drop-in 
replacement for OpenAI API. Primary use case: Run Zed IDE's AI agents on YOUR 
homelab GPUs instead of paying for cloud APIs.
```

### Long Version
```
rbee (pronounced "are-bee", formerly llama-orch) is an OpenAI-compatible AI 
orchestration platform. THE MAIN GOAL: Power Zed IDE's AI agents with your 
homelab GPUs. Use ALL your computers' GPU power for AI coding instead of 
paying for cloud APIs. OpenAI-compatible API means drop-in replacement for 
Zed IDE, Cursor, Continue.dev, or any tool using OpenAI SDK. Zero API costs. 
Full control. Complete privacy.
```

---

## Documents Updated

✅ **AI_DEVELOPMENT_STORY.md** - Added pronunciation, primary use case, OpenAI compatibility  
✅ **VIDEO_SCRIPTS.md** - Added pronunciation and Zed IDE use case to all scripts  
✅ **README.md** - Completely rewritten intro with primary use case front and center  

---

*Always emphasize: rbee (pronounced "are-bee") powers Zed IDE with your homelab GPUs* 🐝
