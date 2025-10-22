# Quick Start Guide - narration-core v0.5.0

**For teams using narration-core** 🎀

---

## 🚀 TL;DR

```rust
use observability_narration_core::NarrationFactory;

// 1. Define factory once per file (actor ≤10 chars)
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

// 2. Use .action() everywhere (action ≤15 chars)
NARRATE.action("queen_start")
    .context("http://localhost:8500")
    .human("Starting queen on {}")
    .emit();

// Output:
// [keeper    ] queen_start    : Starting queen on http://localhost:8500
```

---

## 📏 Rules

### 1. Actor ≤ 10 characters (compile-time enforced)

```rust
// ✅ GOOD
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");
const NARRATE: NarrationFactory = NarrationFactory::new("queen");
const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");
const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

// ❌ BAD - Compile error!
const NARRATE: NarrationFactory = NarrationFactory::new("keeper/queen-life");  // 17 chars
const NARRATE: NarrationFactory = NarrationFactory::new("queen-router");       // 12 chars
```

### 2. Action ≤ 15 characters (runtime enforced)

```rust
// ✅ GOOD
NARRATE.action("queen_start")     // 11 chars
NARRATE.action("job_submit")      // 10 chars
NARRATE.action("hive_install")    // 12 chars
NARRATE.action("status")          // 6 chars

// ❌ BAD - Runtime panic!
NARRATE.action("queen_status_check")  // 18 chars
```

### 3. Use `.action()` not `.narrate()`

```rust
// ✅ GOOD
NARRATE.action("start")

// ❌ BAD - Method doesn't exist anymore!
NARRATE.narrate("start")
```

---

## 🎯 Pattern: One Factory Per File

Each file defines its own `const NARRATE`:

```rust
// src/main.rs
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

// src/queen_lifecycle.rs
const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

// src/job_client.rs
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");
```

**Benefits:**
- ✅ No imports needed
- ✅ Each file controls its own actor
- ✅ Cleaner, shorter code

---

## 📊 Output Format

```
[actor     ] action         : message
│          │                │
│          │                └─ Your message (starts at column 31)
│          └─ Action (15 chars, left-aligned, space-padded)
└─ Actor (10 chars, left-aligned, space-padded)
```

**Total prefix**: 30 characters

**Example:**
```
[keeper    ] queen_start    : Starting queen on http://localhost:8500
[kpr-life  ] queen_check    : Queen is already running and healthy
[queen     ] listen         : Listening on http://127.0.0.1:8500
[qn-router ] job_create     : Job abc123 created, waiting for client connection
```

---

## 🔧 Common Patterns

### Basic Narration

```rust
NARRATE.action("start")
    .human("Starting service")
    .emit();
```

### With Context Interpolation

```rust
NARRATE.action("queen_start")
    .context("http://localhost:8500")  // {0} or {}
    .context("8500")                   // {1}
    .human("Starting queen on {0}, port {1}")
    .emit();
```

### With Metadata

```rust
NARRATE.action("job_complete")
    .correlation_id("req-abc123")
    .duration_ms(150)
    .human("Job completed successfully")
    .emit();
```

### Error Handling

```rust
NARRATE.action("queen_start")
    .context(error.to_string())
    .human("Failed to start queen: {}")
    .error_kind("startup_failed")
    .emit_error();  // ERROR level
```

### Table Output

```rust
use serde_json::json;

let data = vec![
    json!({"hive_id": "hive-1", "status": "running"}),
    json!({"hive_id": "hive-2", "status": "stopped"}),
];

NARRATE.action("status")
    .human("Found 2 hives:")
    .table(data)
    .emit();
```

---

## 🎨 Logging Levels

```rust
NARRATE.action("start").human("Starting").emit();         // INFO (default)
NARRATE.action("retry").human("Retrying").emit_warn();    // WARN
NARRATE.action("failed").human("Failed").emit_error();    // ERROR
```

---

## ✅ Migration Checklist

Migrating from v0.4.0 to v0.5.0:

- [ ] Shorten all actors to ≤10 chars
  - Remove emojis if needed
  - Use abbreviations: `"keeper"`, `"queen"`, `"qn-router"`
- [ ] Replace `.narrate()` with `.action()`
  - Find: `.narrate(`
  - Replace: `.action(`
- [ ] Verify all actions are ≤15 chars
  - Most should already comply
  - Shorten if needed
- [ ] Test compilation
  - Actors >10 chars will fail at compile time
  - Clear error messages will guide you
- [ ] Test runtime
  - Actions >15 chars will panic at runtime
  - Clear error messages will guide you

---

## 🐛 Common Errors

### Compile Error: Actor Too Long

```
error[E0080]: evaluation panicked: Actor string is too long! Maximum 10 characters allowed.
   --> src/main.rs:10:35
    |
 10 | const NARRATE: NarrationFactory = NarrationFactory::new("keeper/queen-life");
    |                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

**Fix:** Shorten the actor string to ≤10 characters.

### Runtime Panic: Action Too Long

```
thread 'main' panicked at 'Action string is too long! Maximum 15 characters allowed. Got 'queen_status_check' (18 chars)'
```

**Fix:** Shorten the action string to ≤15 characters.

---

## 📚 More Resources

- **Full README**: [README.md](README.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Examples**: [`examples/`](examples/)
- **Tests**: [`tests/`](tests/)

---

## 💡 Tips

1. **Keep actors short and memorable**: `"keeper"`, `"queen"`, `"hive"`
2. **Use snake_case for actions**: `"queen_start"`, `"job_submit"`
3. **Use context interpolation**: `{0}`, `{1}`, `{2}` or just `{}`
4. **One factory per file**: Each file defines its own `const NARRATE`
5. **Test early**: Compile-time checks catch actor length issues immediately

---

**Happy narrating! 🎀**
