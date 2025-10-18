# TEAM-038 Critical Correction - Cascading Shutdown

**Team:** TEAM-038 (Implementation Team)  
**Date:** 2025-10-10T15:00  
**Status:** ✅ CORRECTED  
**Priority:** CRITICAL

---

## 🚨 Critical Error Found and Fixed

### ❌ WRONG (What I Initially Documented)

**Phase 14: Worker stays alive (idle timeout)**
- Worker transitions to idle
- rbee-hive starts 5-minute timer
- After 5 minutes, rbee-hive shuts down worker
- Worker narration visible to user during shutdown

**This was WRONG because:**
- rbee-keeper exits after getting result
- rbee-keeper kills queen-rbee (if it spawned it)
- queen-rbee kills all rbee-hive instances
- rbee-hive kills all workers
- **Everything shuts down immediately, no 5-minute wait**

---

## ✅ CORRECT (Fixed)

### Phase 14: Cascading Shutdown

**Shutdown sequence:**

**1. rbee-keeper exits:**
```
rbee-keeper completes inference
rbee-keeper displays final result to user
rbee-keeper sends SIGTERM to queen-rbee (if it spawned it)
rbee-keeper exits
```

**2. queen-rbee shuts down:**
```
queen-rbee receives SIGTERM
queen-rbee sends shutdown to all rbee-hive instances via SSH
queen-rbee exits
```

**3. rbee-hive shuts down:**
```
rbee-hive receives shutdown signal via SSH
rbee-hive sends POST /shutdown to all workers
rbee-hive waits for workers to exit
rbee-hive exits
```

**4. Worker shuts down:**
```
worker receives shutdown request
worker narrate("Shutting down gracefully")
  → stdout → rbee-hive captures
  → rbee-hive logs (queen-rbee already exited, NOT relayed to user)
worker narrate("Freeing 669 MB VRAM")
  → stdout → rbee-hive logs (NOT seen by user)
worker narrate("Worker exiting")
  → stdout → rbee-hive logs (NOT seen by user)
worker process exits
```

**Final state:**
- rbee-keeper: exited
- queen-rbee: exited
- rbee-hive: exited
- worker: exited
- VRAM: freed (available for games/other apps)

**Critical point:** Shutdown narration is NOT seen by user because rbee-keeper has already exited. This is by design - user got their result and moved on.

---

## 📝 Files Corrected

### 1. `/home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001.md`
**Changed:**
- Phase 14 title: "Worker stays alive (idle timeout)" → "Cascading Shutdown"
- Added 4-step cascading shutdown sequence
- Clarified shutdown narration is NOT seen by user
- Added note about rbee-keeper exiting first

### 2. `/home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001-mvp.md`
**Changed:**
- Updated cascading shutdown section
- Clarified all processes exit
- Added note about shutdown narration not being visible
- Updated "Critical Corrections Applied" section

### 3. `/home/vince/Projects/llama-orch/test-harness/bdd/tests/features/test-001-mvp.feature`
**Changed:**
- Updated RULE 5 (Cascading Shutdown)
- Added: "IMPORTANT: Worker does NOT stay alive after rbee-keeper exits"
- Added: "IMPORTANT: No 5-minute idle timeout in testing mode (ephemeral)"

---

## 🎯 Key Corrections

### Lifecycle Understanding

**OLD (WRONG):**
- Worker stays alive after inference
- 5-minute idle timeout
- Worker shuts down after timeout
- Shutdown narration visible to user

**NEW (CORRECT):**
- rbee-keeper exits immediately after result
- Cascading shutdown: rbee-keeper → queen-rbee → rbee-hive → workers
- All processes exit within seconds
- Shutdown narration NOT visible to user (rbee-keeper already exited)

### Persistent vs Ephemeral Mode

**Testing Mode (rbee-keeper spawns queen-rbee):**
- rbee-keeper owns queen-rbee lifecycle
- When rbee-keeper exits, everything shuts down
- No idle timeout (ephemeral)
- VRAM freed immediately

**Production Mode (queen-rbee pre-started):**
- queen-rbee runs as daemon
- Workers may have idle timeout
- rbee-keeper just connects, doesn't control lifecycle
- VRAM freed after idle timeout

**For MVP testing:** We're in ephemeral mode, so cascading shutdown applies.

---

## 🔄 Narration Flow During Shutdown

### What User Sees
```bash
$ rbee-keeper infer --node mac --model tinyllama --prompt "hello"

[rbee-hive] 🌅 Starting pool manager on port 9200
[llm-worker-rbee] 🌅 Worker starting on port 8001
[model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!
[candle-backend] 🚀 Starting inference
[tokenizer] 🍰 Tokenized prompt (1 token)
Hello world, this is a test...
[candle-backend] 🎉 Complete! 20 tokens in 150ms

✅ Done!

$ █  # rbee-keeper exits, user back at shell
```

### What User Does NOT See
```
# These happen after rbee-keeper exits:
[http-server] 👋 Shutting down gracefully     # NOT VISIBLE
[device-manager] 🧹 Freeing 669 MB VRAM        # NOT VISIBLE
[llm-worker-rbee] 💤 Worker exiting            # NOT VISIBLE
```

**Why?** rbee-keeper has already exited. The shutdown narration goes to rbee-hive logs, but queen-rbee has also exited, so it's not relayed anywhere.

---

## ✅ Verification

### Corrected Understanding
- ✅ rbee-keeper exits after displaying result
- ✅ Cascading shutdown: rbee-keeper → queen-rbee → rbee-hive → workers
- ✅ All processes exit within seconds
- ✅ Shutdown narration NOT visible to user
- ✅ VRAM freed immediately
- ✅ No 5-minute idle timeout in testing mode

### Files Updated
- ✅ `bin/.specs/.gherkin/test-001.md`
- ✅ `bin/.specs/.gherkin/test-001-mvp.md`
- ✅ `test-harness/bdd/tests/features/test-001-mvp.feature`

### Documentation Consistency
- ✅ All files now show cascading shutdown
- ✅ All files clarify shutdown narration not visible
- ✅ All files distinguish ephemeral vs persistent mode

---

## 🎓 Lessons Learned

### 1. Ephemeral vs Persistent
**Ephemeral (testing):**
- rbee-keeper spawns queen-rbee
- rbee-keeper owns lifecycle
- Everything shuts down when rbee-keeper exits

**Persistent (production):**
- queen-rbee runs as daemon
- rbee-keeper just connects
- Workers may stay alive with idle timeout

### 2. Shutdown Narration
**During inference:** User sees narration (rbee-keeper active)  
**During shutdown:** User does NOT see narration (rbee-keeper exited)

This is by design - user got their result and moved on.

### 3. VRAM Management
**Ephemeral mode:** VRAM freed immediately when rbee-keeper exits  
**Persistent mode:** VRAM freed after idle timeout (5 minutes)

For homelab use case (gaming), ephemeral mode is better - VRAM freed right away.

---

## 🚀 Impact on TEAM-039

### What TEAM-039 Needs to Know

**1. Cascading Shutdown Implementation:**
- rbee-keeper must send SIGTERM to queen-rbee on exit
- queen-rbee must send shutdown to all rbee-hive instances
- rbee-hive must send shutdown to all workers
- All processes must exit gracefully

**2. Shutdown Narration:**
- Shutdown narration still emitted (for logs)
- But NOT relayed to user (rbee-keeper exited)
- This is expected behavior, not a bug

**3. Testing:**
- Test cascading shutdown works
- Test VRAM is freed
- Test all processes exit
- Don't expect shutdown narration in user output

---

## ✅ Definition of Done

**This correction is complete when:**

1. ✅ All documentation shows cascading shutdown
2. ✅ All files clarify shutdown narration not visible
3. ✅ Lifecycle rules updated in feature files
4. ✅ Gherkin specs updated
5. ✅ Critical correction document created

---

**TEAM-038 Critical Correction Complete ✅**

**Thank you for catching this critical error!**

The documentation now correctly reflects that:
- Worker does NOT stay alive after rbee-keeper exits
- Cascading shutdown happens immediately
- Shutdown narration is NOT visible to user
- VRAM is freed right away

---

**Signed:** TEAM-038 (Implementation Team)  
**Date:** 2025-10-10T15:00  
**Status:** ✅ CORRECTED AND VERIFIED
