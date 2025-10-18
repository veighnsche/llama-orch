# Anti-Freeze Enforcer Solution

## Problem Summary

AI developers repeatedly violate engineering rules by piping command output into `grep`, `tail`, or `head`:
```bash
❌ cargo test | grep ...
❌ ./binary | tail -n 20
❌ command | head -n 50
```

**Impact:** Output is filtered before completion, making it impossible to detect if programs hang.

## Why Rule Enforcement Doesn't Work

Simply adding stronger rules to `.windsurf/rules/engineering-rules.md` **fails** because:
- Rules fade from AI memory after 2-3 responses
- When the "real deal" testing commands appear, AI reverts to piping
- Not sustainable long-term

## Solution: Command-Level Enforcement

Replaced wrapper scripts that tried to show live output (impossible in Cascade) with **enforcers** that **REFUSE to work in pipeline mode**.

### How It Works

**Installed enforcers:**
- `/home/vince/.local/bin/grep` (53 lines)
- `/home/vince/.local/bin/head` (47 lines)  
- `/home/vince/.local/bin/tail` (47 lines)

**Behavior:**
1. ✅ **Interactive mode**: Works normally (output to TTY)
2. ✅ **File mode**: Works normally (`grep pattern file.txt`)
3. ❌ **Pipeline mode**: **REFUSES TO WORK** and shows error with correct pattern

**Example:**
```bash
$ cargo test | grep "PASSED"

❌ ERROR: grep in pipeline mode is BANNED

🚫 You tried: <command> | grep ...

✅ Use the two-step pattern instead:

   Step 1: Run to file (see ALL output)
   command > output.log 2>&1

   Step 2: Filter the file
   grep ... output.log

💡 Why: Piping hides output and makes it impossible to detect hangs.

See: .windsurf/rules/engineering-rules.md (lines 68-93)
```

### Why This Works

1. **Forces compliance** - Commands fail immediately if AI tries to pipe
2. **Shows the correct pattern** - AI sees the error message and learns
3. **Self-reinforcing** - Each failure reinforces the correct behavior
4. **No memory required** - The system enforces the rule mechanically

## Technical Details

### PATH Configuration
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Set in `~/.bashrc` - enforcer scripts intercept commands before real binaries.

### Detection Logic
```bash
# If writing to TTY (interactive), use real command
if [[ -t 1 ]]; then 
  exec "$real" "$@"
fi

# If reading from files, use real command
has_file=0
for a in "$@"; do
  if [[ "$a" != -* && -e "$a" ]]; then
    has_file=1
    break
  fi
done

if [[ $has_file -eq 1 ]]; then
  exec "$real" "$@"
fi

# Pipeline mode detected - REFUSE TO WORK
cat >&2 <<'EOF'
❌ ERROR: <command> in pipeline mode is BANNED
...
EOF
exit 1
```

### Supported Modes

| Mode | Example | Behavior |
|------|---------|----------|
| Interactive | `grep pattern` (user typing) | ✅ Uses real grep |
| File input | `grep pattern file.txt` | ✅ Uses real grep |
| Pipeline stdin | `command \| grep pattern` | ❌ **REFUSES**, shows error |

## Verification

```bash
# Test enforcer is active
$ grep --wrapper-test
WRAPPER IS ACTIVE - Anti-freeze enforcer enabled

# Test file mode works
$ echo "test" > test.txt
$ grep test test.txt
test

# Test pipeline mode fails
$ echo "test" | grep test
❌ ERROR: grep in pipeline mode is BANNED
...
```

## Benefits

1. **Sustainable** - No reliance on AI memory
2. **Educational** - Error messages teach correct pattern
3. **Immediate** - Fails fast, no wasted time
4. **Precise** - Only blocks problematic patterns
5. **Transparent** - Normal usage unaffected

## Related Documentation

- `.windsurf/rules/engineering-rules.md` (lines 68-93) - Original rules
- `ANTI_FREEZE_WRAPPER_ISSUE.md` - Previous failed approach (live output)

## Migration Notes

**Old approach (failed):**
- Tried to show live output while buffering
- Couldn't work in Cascade's captured environment
- Complex with `tee`, temp files, TTY devices

**New approach (working):**
- Simple: just refuse to work and show error
- Works in any environment
- Self-documenting through error messages

---

**Status:** ✅ Deployed and active  
**Date:** October 18, 2025  
**Files Modified:** 3 wrapper scripts (grep, head, tail)
