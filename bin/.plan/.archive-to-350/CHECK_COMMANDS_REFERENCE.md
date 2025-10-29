# rbee-keeper Check Commands Reference

**Last Updated:** Oct 27, 2025  
**Teams:** TEAM-309 (self-check), TEAM-312 (queen-check), TEAM-313 (hive-check)

## Overview

rbee-keeper provides three check commands to validate different parts of the system:

| Command | Scope | Execution | Purpose |
|---------|-------|-----------|---------|
| `self-check` | rbee-keeper | Local CLI | Test narration system in CLI mode |
| `queen-check` | queen-rbee | Via queen job server | Test narration through SSE streaming |
| `hive-check` | Hive lifecycle | Local (rbee-keeper) | Test hive lifecycle management |

## Commands

### 1. self-check

**Purpose:** Test narration system in CLI mode (local execution)

**Usage:**
```bash
rbee-keeper self-check
```

**What it tests:**
- Narration macro (`n!()`) functionality
- All three narration modes (Human, Cute, Story)
- Format specifiers (hex, debug, float)
- Sequential narrations
- Configuration loading
- Tracing subscriber integration

**Output:** Narration events printed to stderr via tracing subscriber

**Implementation:** `bin/00_rbee_keeper/src/handlers/self_check.rs`

**Team:** TEAM-309

---

### 2. queen-check

**Purpose:** Test narration through entire SSE streaming pipeline

**Usage:**
```bash
rbee-keeper queen-check
```

**What it tests:**
- Job submission to queen-rbee
- SSE streaming from queen to client
- Job ID routing through narration system
- Narration context propagation
- All three narration modes via SSE
- Sequential narrations streamed in real-time

**Output:** Narration events streamed via SSE from queen-rbee

**Implementation:** 
- Client: `bin/00_rbee_keeper/src/main.rs` (routes to queen)
- Server: `bin/10_queen_rbee/src/handlers/queen_check.rs`

**Team:** TEAM-312

**Architecture:**
```
rbee-keeper → POST /v1/jobs → queen-rbee → job_router
                                              ↓
                                         queen_check handler
                                              ↓
                                         SSE stream → client
```

---

### 3. hive-check

**Purpose:** Test hive lifecycle management operations

**Usage:**
```bash
rbee-keeper hive-check
```

**What it tests:**
- Local hive operations (install, start, stop, uninstall)
- Remote hive operations (via SSH)
- Status checks (local and remote)
- Error handling scenarios
- Configuration validation
- Binary resolution
- SSH configuration
- Narration modes during lifecycle operations

**Output:** Narration events printed to stderr (local execution)

**Implementation:** `bin/00_rbee_keeper/src/handlers/hive_check.rs`

**Team:** TEAM-313

**Lifecycle Responsibilities:**
rbee-keeper has lifecycle responsibility for:
- Queen-rbee (install, uninstall, start, stop, rebuild)
- Hives (install, uninstall, start, stop, status)
  - Local hives (direct process control)
  - Remote hives (via SSH)

---

## Comparison Matrix

| Feature | self-check | queen-check | hive-check |
|---------|------------|-------------|------------|
| **Execution** | Local | Via queen | Local |
| **Narration Output** | stderr (tracing) | SSE stream | stderr (tracing) |
| **Job ID** | No | Yes | No |
| **SSE Routing** | No | Yes | No |
| **Tests Narration** | ✅ | ✅ | ✅ |
| **Tests SSE** | ❌ | ✅ | ❌ |
| **Tests Lifecycle** | ❌ | ❌ | ✅ |
| **Requires Queen** | ❌ | ✅ | ❌ |
| **Duration** | ~1 second | ~1 second | ~2 seconds |

## When to Use Each

### Use `self-check` when:
- Testing narration system locally
- Debugging CLI narration output
- Verifying tracing subscriber setup
- Quick sanity check of rbee-keeper

### Use `queen-check` when:
- Testing SSE streaming pipeline
- Debugging job routing
- Verifying narration context propagation
- Testing queen-rbee integration
- Validating job_id routing

### Use `hive-check` when:
- Testing hive lifecycle management
- Validating SSH configuration
- Debugging hive install/start/stop operations
- Verifying binary resolution
- Testing keeper's orchestration capabilities

## Development Workflow

**Typical development flow:**

1. **Start with `self-check`** - Verify basic narration works
   ```bash
   rbee-keeper self-check
   ```

2. **Then `queen-check`** - Verify SSE streaming works
   ```bash
   rbee-keeper queen-check
   ```

3. **Finally `hive-check`** - Verify lifecycle management works
   ```bash
   rbee-keeper hive-check
   ```

**All three should pass before committing changes to narration or lifecycle code.**

## Implementation Details

### Narration Modes

All three checks test the three narration modes:

1. **Human Mode** (default)
   - Technical, precise messages
   - Example: "Starting rbee-keeper self-check"

2. **Cute Mode**
   - Whimsical, emoji-rich messages
   - Example: "🐝 Testing narration in cute mode!"

3. **Story Mode**
   - Narrative, third-person messages
   - Example: "'Testing narration', said the keeper"

### Narration Output Format

**CLI mode (self-check, hive-check):**
```
<function_name> <action>
                <message>
```

Example:
```
rbee_keeper::handlers::self_check::handle_self_check self_check_start
                Starting rbee-keeper self-check
```

**SSE mode (queen-check):**
```
data: <formatted_message>
```

## Code Locations

```
bin/00_rbee_keeper/src/
├── handlers/
│   ├── self_check.rs      # TEAM-309: self-check implementation
│   └── hive_check.rs      # TEAM-313: hive-check implementation
├── cli/
│   └── commands.rs        # All three commands defined here
└── main.rs                # Command routing

bin/10_queen_rbee/src/
└── handlers/
    └── queen_check.rs     # TEAM-312: queen-check implementation
```

## Future Enhancements

Potential additions to the check suite:

1. **worker-check** - Test worker lifecycle management
2. **model-check** - Test model download/management
3. **infer-check** - Test inference pipeline end-to-end
4. **integration-check** - Full system integration test
5. **performance-check** - Benchmark operations
6. **chaos-check** - Fault injection testing

## Related Documentation

- `.windsurf/rules/engineering-rules.md` - Engineering standards
- `bin/NARRATION_AND_JOB_ID_ARCHITECTURE.md` - Narration architecture
- `bin/.plan/TEAM_309_SELF_CHECK.md` - self-check details
- `bin/.plan/TEAM_312_QUEEN_CHECK.md` - queen-check details
- `bin/.plan/TEAM_313_HIVE_CHECK.md` - hive-check details
