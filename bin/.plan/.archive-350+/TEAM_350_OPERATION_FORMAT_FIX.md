# TEAM-350: Fixed Operation JSON Format

**Status:** ✅ COMPLETE

## The Error We Found

Thanks to the enhanced logging, we finally saw the **real error**:

```
[RHAI Test] Error caught: "Failed to parse operation: Error: missing field `operation`"
```

## Root Cause

### Serde Tagged Enum Format

**File:** `bin/97_contracts/operations-contract/src/lib.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    RhaiScriptTest {
        content: String,
    },
    // ...
}
```

The `#[serde(tag = "operation")]` attribute means the JSON format must be:

```json
{
  "operation": "rhai_script_test",
  "content": "..."
}
```

### What We Were Sending (WRONG)

```typescript
const operation = {
  RhaiScriptTest: { content }  // ← Wrong format!
}
```

This produces:
```json
{
  "RhaiScriptTest": {
    "content": "..."
  }
}
```

**Result:** `missing field 'operation'` error ❌

### What We Should Send (CORRECT)

```typescript
const operation = {
  operation: 'rhai_script_test',  // ← Tag field
  content                          // ← Flattened content
}
```

This produces:
```json
{
  "operation": "rhai_script_test",
  "content": "..."
}
```

**Result:** Parses correctly ✅

## The Fix

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts`

```typescript
// TEAM-350: Operation uses #[serde(tag = "operation")] format
const operation = {
  operation: 'rhai_script_test',  // ← Tag
  content                          // ← Fields
}
```

## Serde Tagged Enum Pattern

This is a common Rust pattern for JSON APIs:

**Rust:**
```rust
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    RhaiScriptTest { content: String },
    RhaiScriptSave { name: String, content: String, id: Option<String> },
    // ...
}
```

**JSON:**
```json
// Test
{
  "operation": "rhai_script_test",
  "content": "print('hello')"
}

// Save
{
  "operation": "rhai_script_save",
  "name": "my_script",
  "content": "print('hello')",
  "id": null
}
```

The `tag` field identifies the variant, and the variant's fields are flattened into the object.

## Testing

```bash
# 1. Rebuild the React package
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build

# 2. Rebuild the app (or let build.rs do it)
cd ../app
pnpm exec vite build

# 3. Or just rebuild queen-rbee (build.rs does everything now!)
cargo build --bin queen-rbee

# 4. Start queen-rbee
cargo run --bin queen-rbee

# 5. Open http://localhost:7834
# 6. Press Test button
```

**Expected result:**
```
[RHAI Test] Starting test...
[RHAI Test] Client created, baseUrl: "http://localhost:7833"
[RHAI Test] Operation: { operation: 'rhai_script_test', content: '...' }
[RHAI Test] Submitting and streaming...
[RHAI Test] SSE line: data: {"action":"rhai_test_start",...}
[RHAI Test] Narration event: {...}
[RHAI Test] SSE line: [DONE]
[RHAI Test] Stream complete, receivedDone: true
✅ Test completed successfully
```

## Files Changed

1. **useRhaiScripts.ts** - Fixed operation format to match serde tagged enum

## Related Operations

All RHAI operations use the same pattern:

```typescript
// Test
{ operation: 'rhai_script_test', content: '...' }

// Save
{ operation: 'rhai_script_save', name: '...', content: '...', id: null }

// Get
{ operation: 'rhai_script_get', id: '...' }

// List
{ operation: 'rhai_script_list' }

// Delete
{ operation: 'rhai_script_delete', id: '...' }
```

---

**TEAM-350 Signature:** Fixed operation JSON format to match Rust serde tagged enum
