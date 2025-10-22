# Compliance Signposts Added

**Created by:** TEAM-204  
**Date:** 2025-10-22  
**Purpose:** Redirect engineers from compliance thinking to audit-logging

---

## Where Signposts Were Added

### 1. Module Documentation (`src/lib.rs`)

**Top of file:**
```rust
//! # ⚠️ NOT FOR COMPLIANCE/AUDIT LOGGING
//!
//! **This is for CUTE DEBUGGING, not compliance!** 🐝
//!
//! - ❌ Don't use for GDPR/PCI-DSS/SOC 2 compliance
//! - ❌ Don't use for security audit trails
//! - ❌ Don't use for tamper-evident logs
//!
//! **For compliance/audit logging, see:** `bin/99_shared_crates/audit-logging/`
```

### 2. Main Function (`narrate_at_level`)

**Function docs:**
```rust
/// # ⚠️ NOT FOR COMPLIANCE
///
/// This is for CUTE DEBUGGING, not audit logging!
/// - NO redaction (developers need full context)
/// - NO tamper-evident logs
/// - NO compliance guarantees
///
/// **For compliance/audit logging, see:** `bin/99_shared_crates/audit-logging/`
```

**Inline comment:**
```rust
// TEAM-204: No redaction - narration is for cute debugging, not compliance
// For audit logging, see: bin/99_shared_crates/audit-logging/
```

### 3. Redaction Module (`src/redaction.rs`)

**Module docs:**
```rust
//! # ⚠️ NOT USED BY NARRATION ANYMORE
//!
//! **TEAM-204:** Narration is for cute debugging, not compliance.
//! This module exists for backward compatibility but is NOT used by narration-core.
//!
//! **For compliance/audit logging with redaction, see:** `bin/99_shared_crates/audit-logging/`
```

### 4. README.md

**Top of file:**
```markdown
## ⚠️ NOT FOR COMPLIANCE/AUDIT LOGGING

**This is for DEVELOPERS, not auditors!**

- ❌ Don't use for GDPR/PCI-DSS/SOC 2 compliance
- ❌ Don't use for security audit trails  
- ❌ Don't use for tamper-evident logs
- ❌ NO redaction (developers need full context)

**For compliance/audit logging, see:** `bin/99_shared_crates/audit-logging/`
```

### 5. New Document: `NOT_FOR_COMPLIANCE.md`

**Comprehensive guide:**
- Decision tree
- Examples of what NOT to do
- Examples of what TO do
- Quick reference table
- Clear redirect to `audit-logging`

---

## Key Messages

### What This Crate IS For

🐝 **Cute debugging**  
🐛 **Developer observability**  
🎀 **Human-readable narration**  
💬 **Understanding what's happening**

### What This Crate IS NOT For

❌ **GDPR/PCI-DSS/SOC 2 compliance**  
❌ **Security audit trails**  
❌ **Tamper-evident logs**  
❌ **Legal evidence**

### Where To Go Instead

**For compliance:** `bin/99_shared_crates/audit-logging/`

---

## Decision Tree

```
Are you logging this for...

├─ Debugging? → Use narration-core ✅
├─ Compliance? → Use audit-logging ✅
├─ Security? → Use audit-logging ✅
├─ Legal evidence? → Use audit-logging ✅
└─ Cute messages? → Use narration-core ✅
```

---

## Files Modified

1. ✅ `src/lib.rs` - Module docs + function docs + inline comments
2. ✅ `src/redaction.rs` - Module docs
3. ✅ `README.md` - Top warning section

## Files Created

4. ✅ `NOT_FOR_COMPLIANCE.md` - Comprehensive guide

---

## Verification

```bash
$ cargo check --package observability-narration-core
✅ Finished `dev` profile [unoptimized + debuginfo]
```

---

**END OF SIGNPOSTS**

**Engineers thinking about compliance will now be redirected to:** `bin/99_shared_crates/audit-logging/`
