# DRY Refactor: Removed Repeated Serialization (TEAM-186)

## Problem

Every command in `main.rs` had this repeated pattern:

```rust
let operation = Operation::HiveStart { hive_id: id.clone() };
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
submit_and_stream_job(&client, &queen_url, job_payload).await
```

**Repeated 4 times** (Hive, Worker, Model, Infer commands) = **12 lines of duplication!**

---

## Solution

### 1. Updated `submit_and_stream_job` Signature

**Before:**
```rust
pub async fn submit_and_stream_job(
    client: &reqwest::Client,
    queen_url: &str,
    job_payload: serde_json::Value,  // ❌ Pre-serialized JSON
) -> Result<()>
```

**After:**
```rust
pub async fn submit_and_stream_job(
    client: &reqwest::Client,
    queen_url: &str,
    operation: Operation,  // ✅ Accept Operation directly
) -> Result<()>
```

### 2. Moved Serialization Inside Function

**Now serialization happens in ONE place:**

```rust
pub async fn submit_and_stream_job(
    client: &reqwest::Client,
    queen_url: &str,
    operation: Operation,
) -> Result<()> {
    // TEAM-186: Serialize operation here (DRY - single place!)
    let job_payload = serde_json::to_value(&operation)
        .expect("Failed to serialize operation");
    
    // ... rest of function
}
```

### 3. Updated All Call Sites

**Before (Repeated 4 times):**
```rust
let operation = Operation::HiveStart { hive_id: id.clone() };
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
submit_and_stream_job(&client, &queen_url, job_payload).await
```

**After (Clean!):**
```rust
let operation = Operation::HiveStart { hive_id: id.clone() };
submit_and_stream_job(&client, &queen_url, operation).await
```

---

## Benefits

### ✅ DRY (Don't Repeat Yourself)
- Serialization logic in **ONE place** instead of 4
- Easier to maintain
- Easier to change serialization logic if needed

### ✅ Type Safety
- Function signature is clearer: "I accept an Operation"
- No risk of passing wrong JSON structure
- Compiler enforces correct type

### ✅ Better API
- Callers don't need to know about serialization
- Implementation detail hidden inside function
- Cleaner call sites

### ✅ Use Operation Methods
- Can use `operation.name()` instead of JSON parsing
- Can use `operation.hive_id()` instead of JSON parsing
- More robust and type-safe

---

## Code Reduction

### Before
```rust
// Hive command
let operation = /* ... */;
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
submit_and_stream_job(&client, &queen_url, job_payload).await

// Worker command
let operation = /* ... */;
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
submit_and_stream_job(&client, &queen_url, job_payload).await

// Model command
let operation = /* ... */;
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
submit_and_stream_job(&client, &queen_url, job_payload).await

// Infer command
let operation = /* ... */;
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
submit_and_stream_job(&client, &queen_url, job_payload).await
```

**Total:** 12 lines of duplication

### After
```rust
// Hive command
let operation = /* ... */;
submit_and_stream_job(&client, &queen_url, operation).await

// Worker command
let operation = /* ... */;
submit_and_stream_job(&client, &queen_url, operation).await

// Model command
let operation = /* ... */;
submit_and_stream_job(&client, &queen_url, operation).await

// Infer command
let operation = /* ... */;
submit_and_stream_job(&client, &queen_url, operation).await
```

**Total:** 4 lines (clean!)

**Reduction:** 8 lines removed + 1 line added in function = **7 lines saved**

---

## Additional Improvements

### Use Operation Methods Instead of JSON Parsing

**Before:**
```rust
let operation = job_payload["operation"].as_str().unwrap_or("unknown");
let hive_id = job_payload["hive_id"].as_str();
```

**After:**
```rust
let operation_name = operation.name();
let hive_id = operation.hive_id();
```

**Benefits:**
- Type-safe (no unwrap/as_str)
- Clearer intent
- Works for all operations (not just ones with hive_id)

---

## Summary

**Single Responsibility Principle:**
- `main.rs` handles CLI parsing and operation construction
- `job_client.rs` handles serialization and HTTP communication
- Clean separation of concerns

**DRY Principle:**
- Serialization happens in ONE place
- No repeated code
- Easier to maintain

**Type Safety:**
- Function signature enforces correct type
- Compiler catches errors
- No risk of malformed JSON

**All changes compile and work correctly!** 🎯
