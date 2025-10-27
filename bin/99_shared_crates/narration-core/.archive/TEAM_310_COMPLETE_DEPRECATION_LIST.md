# TEAM-310: Complete Deprecation List

**Status:** ✅ COMPLETE

## All Deprecated Items with #[deprecated] Attributes

### `src/api/builder.rs` - Narration Builder (38 items)

#### Struct
1. `Narration` - Use n!() macro instead

#### Constructor
2. `Narration::new()` - Use n!() macro instead

#### Builder Methods (35 methods)
3. `action()` - Set action
4. `context()` - Add context value for {0}, {1} interpolation
5. `human()` - Set human message
6. `cute()` - Set cute message
7. `story()` - Set story message
8. `correlation_id()` - Set correlation ID
9. `session_id()` - Set session ID
10. `job_id()` - Set job ID
11. `maybe_job_id()` - Conditionally set job ID
12. `task_id()` - Set task ID
13. `pool_id()` - Set pool ID
14. `replica_id()` - Set replica ID
15. `worker_id()` - Set worker ID
16. `hive_id()` - Set hive ID
17. `operation()` - Set operation name
18. `duration_ms()` - Set duration
19. `error_kind()` - Set error kind
20. `retry_after_ms()` - Set retry delay
21. `backoff_ms()` - Set backoff duration
22. `queue_position()` - Set queue position
23. `predicted_start_ms()` - Set predicted start time
24. `engine()` - Set engine name
25. `engine_version()` - Set engine version
26. `model_ref()` - Set model reference
27. `device()` - Set device
28. `tokens_in()` - Set input tokens
29. `tokens_out()` - Set output tokens
30. `decode_time_ms()` - Set decode time
31. `source_location()` - Set source location
32. `table()` - Attach JSON table
33. `emit()` - Emit at INFO level
34. `emit_warn()` - Emit at WARN level
35. `emit_error()` - Emit at ERROR level
36. `emit_debug()` - Emit at DEBUG level (feature-gated)
37. `emit_trace()` - Emit at TRACE level (feature-gated)
38. `emit_with_provenance()` - Internal emit with provenance

### `src/api/builder.rs` - NarrationFactory (5 items)

#### Struct
39. `NarrationFactory` - Use n!() macro instead

#### Methods
40. `NarrationFactory::new()` - Create factory
41. `action()` - Create narration for action
42. `actor()` - Get actor name
43. `with_job_id()` - Create job-scoped narration

### `src/format.rs` (1 item)

44. `interpolate_context()` - Legacy {0}, {1} interpolation

### `src/api/emit.rs` (1 item)

45. `human()` - Legacy function (already deprecated pre-TEAM-310)

### `src/context.rs` (1 item)

46. `NarrationContext::with_actor()` - Actor now auto-detected (already deprecated pre-TEAM-310)

### `src/api/macro_impl.rs` (2 items - already deprecated pre-TEAM-310)

47. `macro_emit()` - Use n!() macro
48. `macro_emit_with_actor()` - Use n!() macro

---

## Total Deprecation Count

**TEAM-310 Added:** 43 new #[deprecated] attributes
**Pre-existing:** 5 deprecated items
**Total:** 48 deprecated items

## Breakdown by Category

| Category | Count | Items |
|----------|-------|-------|
| **Builder Methods** | 35 | All Narration builder methods |
| **Structs** | 2 | Narration, NarrationFactory |
| **Factory Methods** | 4 | NarrationFactory methods |
| **Format Functions** | 1 | interpolate_context |
| **Legacy Functions** | 3 | human, macro_emit, macro_emit_with_actor |
| **Context Methods** | 1 | with_actor |
| **Constructors** | 2 | Narration::new, NarrationFactory::new |

## Migration Path

**All deprecated code** → Use `n!()` macro

```rust
// ❌ OLD (43 deprecated methods)
Narration::new("actor", "action", "target")
    .context("value")
    .human("Message {}")
    .job_id("job-123")
    .duration_ms(150)
    .emit();

// ✅ NEW (1 macro call)
n!("action", "Message {}", value);
```

## Compiler Impact

Running `cargo check` now shows **100+ deprecation warnings** across the codebase, guiding developers to migrate to the new `n!()` macro.

---

**TEAM-310**: Comprehensive deprecation coverage - 48 total deprecated items with proper Rust attributes.
