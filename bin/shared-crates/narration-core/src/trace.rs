// trace.rs — Ultra-lightweight TRACE-level narration
// Optimized for minimal overhead in hot paths

use tracing::{event, Level};

/// Minimal trace event for ultra-fine-grained logging.
/// 
/// **Performance**: ~10x faster than full `narrate()` because:
/// - No redaction (TRACE is dev-only, secrets shouldn't be in hot paths)
/// - No struct allocation
/// - No optional field handling
/// - Direct tracing macro invocation
/// 
/// **Use ONLY for**:
/// - Loop iterations in hot paths
/// - FFI boundary crossings (frequent)
/// - CUDA kernel launches (high frequency)
/// - Lock acquisition/release (very frequent)
/// - Memory operations (extremely frequent)
/// 
/// **DO NOT use for**:
/// - Production code (use INFO/DEBUG instead)
/// - Anything with secrets (use full `narrate()` with redaction)
/// - User-facing events (use INFO)
/// 
/// # Example
/// ```rust
/// use observability_narration_core::trace_tiny;
/// 
/// // Hot path: processing tokens in a loop
/// for (i, token) in tokens.iter().enumerate() {
///     trace_tiny!("tokenizer", "decode", format!("token_{}", i), 
///                 format!("Decoding token {} of {}", i, tokens.len()));
///     
///     // ... actual work ...
/// }
/// ```
/// 
/// # Example: FFI boundary
/// ```rust
/// trace_tiny!("worker-orcd", "ffi_call", "llama_cpp_eval",
///             format!("ENTER llama_cpp_eval(ctx={:?}, n_tokens={})", ctx_ptr, n_tokens));
/// 
/// let result = unsafe { llama_cpp_eval(ctx_ptr, tokens.as_ptr(), n_tokens) };
/// 
/// trace_tiny!("worker-orcd", "ffi_call", "llama_cpp_eval",
///             format!("EXIT llama_cpp_eval → {:?} ({}ms)", result, elapsed_ms));
/// ```
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = $target,
            human = $human,
            "trace"
        );
    };
}

/// Trace with correlation ID (slightly heavier, but still lightweight).
/// 
/// Use when you need request tracking in TRACE-level events.
/// 
/// # Example
/// ```rust
/// use observability_narration_core::trace_with_correlation;
/// 
/// trace_with_correlation!(
///     "orchestratord", 
///     "select_worker", 
///     "worker-gpu0-r1",
///     format!("Evaluating worker-gpu0-r1: load={}/8, latency={}ms", load, latency),
///     correlation_id
/// );
/// ```
#[macro_export]
macro_rules! trace_with_correlation {
    ($actor:expr, $action:expr, $target:expr, $human:expr, $correlation_id:expr) => {
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = $target,
            human = $human,
            correlation_id = $correlation_id,
            "trace"
        );
    };
}

/// Trace function entry (ultra-lightweight).
/// 
/// # Example
/// ```rust
/// use observability_narration_core::trace_enter;
/// 
/// fn dispatch_job(job_id: &str, pool_id: &str) -> Result<()> {
///     trace_enter!("orchestratord", "dispatch_job", 
///                  format!("job_id={}, pool_id={}", job_id, pool_id));
///     
///     // ... function body ...
/// }
/// ```
#[macro_export]
macro_rules! trace_enter {
    ($actor:expr, $function:expr, $args:expr) => {
        tracing::trace!(
            actor = $actor,
            action = "enter",
            target = $function,
            human = format!("ENTER {}({})", $function, $args),
            "trace_enter"
        );
    };
}

/// Trace function exit (ultra-lightweight).
/// 
/// # Example
/// ```rust
/// use observability_narration_core::trace_exit;
/// 
/// fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
///     trace_enter!("orchestratord", "dispatch_job", 
///                  format!("job_id={}, pool_id={}", job_id, pool_id));
///     
///     // ... function body ...
///     
///     let worker_id = select_worker(pool_id)?;
///     
///     trace_exit!("orchestratord", "dispatch_job", 
///                 format!("→ {} ({}ms)", worker_id, elapsed_ms));
///     
///     Ok(worker_id)
/// }
/// ```
#[macro_export]
macro_rules! trace_exit {
    ($actor:expr, $function:expr, $result:expr) => {
        tracing::trace!(
            actor = $actor,
            action = "exit",
            target = $function,
            human = format!("EXIT {} {}", $function, $result),
            "trace_exit"
        );
    };
}

/// Trace loop iteration (ultra-lightweight).
/// 
/// # Example
/// ```rust
/// use observability_narration_core::trace_loop;
/// 
/// for (i, worker) in workers.iter().enumerate() {
///     trace_loop!("orchestratord", "select_worker", i, workers.len(),
///                 format!("worker={}, load={}/8", worker.id, worker.load));
///     
///     // ... evaluation logic ...
/// }
/// ```
#[macro_export]
macro_rules! trace_loop {
    ($actor:expr, $action:expr, $index:expr, $total:expr, $detail:expr) => {
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = format!("iter_{}/{}", $index, $total),
            human = format!("Iteration {}/{}: {}", $index, $total, $detail),
            "trace_loop"
        );
    };
}

/// Trace state change (ultra-lightweight).
/// 
/// # Example
/// ```rust
/// use observability_narration_core::trace_state;
/// 
/// trace_state!("orchestratord", "queue_depth", 
///              format!("{} → {}", old_depth, new_depth),
///              format!("Queue depth changed: {} → {} (added job-{})", old_depth, new_depth, job_id));
/// ```
#[macro_export]
macro_rules! trace_state {
    ($actor:expr, $state_name:expr, $transition:expr, $human:expr) => {
        tracing::trace!(
            actor = $actor,
            action = "state_change",
            target = $state_name,
            transition = $transition,
            human = $human,
            "trace_state"
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_tiny_compiles() {
        trace_tiny!("test", "action", "target", "human message");
    }

    #[test]
    fn test_trace_with_correlation_compiles() {
        let correlation_id = "req-123";
        trace_with_correlation!("test", "action", "target", "human message", correlation_id);
    }

    #[test]
    fn test_trace_enter_compiles() {
        trace_enter!("test", "test_function", "arg1=value1, arg2=value2");
    }

    #[test]
    fn test_trace_exit_compiles() {
        trace_exit!("test", "test_function", "→ Ok(result) (5ms)");
    }

    #[test]
    fn test_trace_loop_compiles() {
        trace_loop!("test", "process", 1, 10, "processing item");
    }

    #[test]
    fn test_trace_state_compiles() {
        trace_state!("test", "counter", "5 → 6", "Counter incremented from 5 to 6");
    }
}
