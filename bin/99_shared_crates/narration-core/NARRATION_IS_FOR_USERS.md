# Narration Is For USERS

**Created by:** TEAM-204  
**Date:** 2025-10-22  

---

## The Correct Understanding

### Narration = For Users 👥

**Users see narration to understand what's happening:**

```rust
NARRATE
    .action("model_load")
    .human("🐝 Loading llama-3.1-8b from /models/llama-3.1-8b.gguf")
    .emit();
```

**User sees in web UI:**
```
[worker] model_load: 🐝 Loading llama-3.1-8b from /models/llama-3.1-8b.gguf
```

**Why no redaction?**
- Users need full context to understand what's happening
- Users are running their own jobs - they should see everything
- Narration is job-scoped (User A can't see User B's narration)

---

### Audit Logging = Hidden from Users 📋

**Audit logs are for compliance, completely separate:**

```rust
AUDIT_LOG
    .event("auth.failure")
    .actor(fingerprint_ip(&ip))  // Redacted
    .details(json!({ "reason": "invalid_password" }))
    .emit();
```

**User sees:** NOTHING (audit logs are hidden)

**Audit logs go to:**
- File on disk (never in UI)
- Security monitoring system
- Compliance database

**Why redacted?**
- Legal requirements (GDPR, PCI-DSS)
- Security (don't expose attack patterns)
- Compliance (tamper-evident logs)

---

## Where Narration Goes

### 1. Web UI (SSE Streams)

```
User starts job → Narration flows via SSE → User sees in web UI
```

**Job-scoped:**
- User A sees only their job's narration
- User B sees only their job's narration
- No cross-contamination

### 2. CLI (stderr)

```
User runs: ./rbee hive status
Narration → stderr → User sees in terminal
```

### 3. Logs (for operators)

```
Narration → log files → Operators can debug issues
```

---

## Where Audit Logs Go

### File Only (Hidden from Users)

```
Security event → Audit log → File on disk
```

**Never in:**
- ❌ Web UI
- ❌ SSE streams
- ❌ User-visible logs

**Only in:**
- ✅ `/var/log/audit/` (file system)
- ✅ Security monitoring system
- ✅ Compliance database

---

## Key Differences

| Aspect | Narration | Audit Logging |
|--------|-----------|---------------|
| **For** | Users | Auditors/Legal |
| **Visible** | ✅ YES (UI/CLI) | ❌ NO (hidden) |
| **Redacted** | ❌ NO | ✅ YES |
| **Purpose** | Show what's happening | Compliance/Security |
| **Location** | SSE/stderr/logs | Files only |
| **Job-scoped** | ✅ YES | N/A |
| **Cute** | ✅ YES 🐝 | ❌ NO |

---

## Examples

### ✅ Good Narration (For Users)

```rust
NARRATE
    .action("inference_start")
    .human("Starting inference with prompt: 'Tell me about bees'")
    .cute("The model is thinking... 🤔")
    .emit();
```

**User sees:** Full prompt, cute message, understands what's happening

### ✅ Good Audit Log (Hidden from Users)

```rust
AUDIT_LOG
    .event("data.access")
    .actor(user_id)
    .resource(fingerprint_data(&customer_data))  // Redacted
    .emit();
```

**User sees:** NOTHING  
**Audit file gets:** Redacted, tamper-evident record

---

## Why This Matters

### Users Need Context

If you redact narration, users can't understand what's happening:

```
❌ BAD: "Loading model from [REDACTED]"
   → User: "What model? Where? Why isn't it working?"

✅ GOOD: "Loading llama-3.1-8b from /models/llama-3.1-8b.gguf"
   → User: "Oh, it's loading that model from that path. Got it!"
```

### Compliance Needs Separation

If you mix narration and audit logging:

```
❌ BAD: User sees audit logs in UI
   → Exposes security information
   → User can tamper with evidence

✅ GOOD: Audit logs hidden in files
   → Users can't see security events
   → Tamper-evident for legal purposes
```

---

## Summary

**Narration:**
- For users 👥
- Visible in UI/CLI ✅
- No redaction ❌
- Full context ✅
- Job-scoped ✅

**Audit Logging:**
- For compliance 📋
- Hidden from users ❌
- Redacted ✅
- Tamper-evident ✅
- File-only ✅

**They are completely separate systems!**

---

**END OF DOCUMENT**
