# observability-narration-macros

Procedural macros for the narration system, providing compile-time template expansion and automatic actor inference.

## Features

- **`#[trace_fn]`** - Automatic function entry/exit tracing with timing
- **`#[narrate(...)]`** - Template-based narration with compile-time expansion
- **Actor inference** - Automatically infer actor from module path
- **Template interpolation** - Compile-time template expansion with stack buffers
- **Conditional compilation** - Zero overhead in production builds

## Usage

### `#[trace_fn]` - Function Tracing

```rust
use observability_narration_macros::trace_fn;

#[trace_fn]
async fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    // Function body
}
```

Generates:
- Entry trace: `ENTER dispatch_job`
- Exit trace: `EXIT dispatch_job (5ms)`
- Automatic timing measurement
- Zero overhead when `trace-enabled` feature is disabled

### `#[narrate(...)]` - Template Narration

```rust
use observability_narration_macros::narrate;

#[narrate(
    action: "dispatch",
    human: "Dispatched job {job_id} to worker {worker_id}",
    cute: "Sent job {job_id} off to its new friend {worker_id}! 🎫"
)]
fn dispatch_job(job_id: &str, worker_id: &str) -> Result<()> {
    // Function body
}
```

## Performance

- **Template interpolation**: <100ns (compile-time expansion)
- **Stack buffers**: Zero heap allocations for templates <256 chars
- **Production builds**: Zero overhead (code removed via conditional compilation)

## Architecture

- `actor_inference.rs` - Extract service name from module path
- `template.rs` - Compile-time template parsing and validation
- `trace_fn.rs` - `#[trace_fn]` implementation
- `narrate.rs` - `#[narrate(...)]` implementation

---

*Built with love by the Narration Core Team 🎀*
