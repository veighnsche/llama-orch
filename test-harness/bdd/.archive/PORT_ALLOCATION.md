# Port Allocation Reference

**Status:** NORMATIVE  
**Source:** `bin/.specs/.gherkin/test-001.md`  
**Last Updated:** 2025-10-10 (TEAM-054)  
**Created by:** TEAM-054

---

## Official Port Allocation

| Component | Port | Location | Purpose | Status |
|-----------|------|----------|---------|--------|
| queen-rbee | 8080 | Control node (blep.home.arpa) | Orchestrator HTTP API | ✅ Active |
| rbee-hive | 9200 | Remote nodes (workstation, mac) | Pool manager HTTP API | ✅ Active |
| llm-worker-rbee | 8001+ | Remote nodes | Worker HTTP API (sequential) | ✅ Active |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Control Node (blep.home.arpa)                               │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ rbee-keeper  │────────>│ queen-rbee   │                │
│  │ (CLI tool)   │         │ port 8080    │                │
│  └──────────────┘         └──────┬───────┘                │
│                                   │                         │
└───────────────────────────────────┼─────────────────────────┘
                                    │ SSH
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Remote Node (workstation.home.arpa)                         │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ rbee-hive    │────────>│ llm-worker   │                │
│  │ port 9200    │         │ port 8001    │                │
│  └──────────────┘         └──────────────┘                │
│                                 │                           │
│                         ┌──────────────┐                   │
│                         │ llm-worker   │                   │
│                         │ port 8002    │                   │
│                         └──────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Historical Context

### Before TEAM-037/TEAM-038 (Old Architecture)
- rbee-hive used port **8080**
- rbee-keeper connected DIRECTLY to rbee-hive
- No queen-rbee orchestrator

### After TEAM-037/TEAM-038 (Current Architecture)
- queen-rbee introduced on port **8080**
- rbee-hive moved to port **9200**
- rbee-keeper connects to queen-rbee
- queen-rbee orchestrates via SSH

**Architecture change date:** 2025-10-10 (around 14:00)

---

## Mock Server Configuration

### For BDD Tests

**Mock queen-rbee:**
```rust
let addr: SocketAddr = "127.0.0.1:8080".parse()?;
```

**Mock rbee-hive:**
```rust
let addr: SocketAddr = "127.0.0.1:9200".parse()?;  // NOT 8080 or 8090!
```

**Mock worker:**
```rust
let addr: SocketAddr = "127.0.0.1:8001".parse()?;
```

---

## Verification Commands

### Check if ports are in use
```bash
# Check queen-rbee
lsof -i :8080

# Check rbee-hive
lsof -i :9200

# Check workers
lsof -i :8001
lsof -i :8002
```

### Test connectivity
```bash
# Test queen-rbee
curl http://localhost:8080/health

# Test rbee-hive (on remote node)
curl http://workstation.home.arpa:9200/v1/health

# Test worker
curl http://workstation.home.arpa:8001/v1/health
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using 8080 for rbee-hive
**Wrong:**
```rust
let rbee_hive_url = "http://127.0.0.1:8080".to_string();  // This is queen-rbee!
```

**Correct:**
```rust
let rbee_hive_url = "http://127.0.0.1:9200".to_string();  // rbee-hive port
```

### ❌ Mistake 2: Using 8090 for rbee-hive
**Wrong:**
```rust
let rbee_hive_url = "http://127.0.0.1:8090".to_string();  // Made up number!
```

**Correct:**
```rust
let rbee_hive_url = "http://127.0.0.1:9200".to_string();  // From spec
```

### ❌ Mistake 3: Copying from old handoffs
**Wrong approach:**
- Read TEAM-043's handoff
- Copy port numbers
- Don't check spec

**Correct approach:**
- Read normative spec (`test-001.md`)
- Verify port numbers
- Cross-reference with this document

---

## References

**Normative Spec:**
- `bin/.specs/.gherkin/test-001.md` (lines 231, 243, 254)

**Architecture Docs:**
- `bin/.specs/LIFECYCLE_CLARIFICATION.md`
- `bin/.specs/ARCHITECTURE_MODES.md`
- `bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`

**Mistake Analysis:**
- `test-harness/bdd/MISTAKES_AND_CORRECTIONS.md`
- `test-harness/bdd/HISTORICAL_MISTAKES_ANALYSIS.md`

---

**This is the ONLY correct port allocation. Always verify against this document!**
