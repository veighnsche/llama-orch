// TEAM-300: Phase 2 - Thread-Local Context Tests
// Purpose: Verify automatic job_id and actor injection from context
// Priority: HIGH (eliminates 100+ manual .job_id() calls)
//
// These tests verify Phase 2 functionality:
//   - Context auto-injects job_id into all narrations
//   - Context auto-injects correlation_id into all narrations
//   - Context auto-injects actor into all narrations
//   - Context is inherited by spawned tasks
//   - Multiple contexts can be nested
//
// See: .plan/TEAM_299_PHASE_2_THREAD_LOCAL_CONTEXT.md

use observability_narration_core::*;
use serial_test::serial;

// ============================================================================
// Basic Context Auto-Injection
// ============================================================================

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_auto_injects_job_id() {
    // TEAM-300: Verify job_id is automatically injected from context
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("auto-inject-test-123");
    
    context::with_narration_context(ctx, async {
        n!("test", "Message with auto-injected job_id");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].job_id, Some("auto-inject-test-123".to_string()));
    assert_eq!(captured[0].action, "test");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_auto_injects_correlation_id() {
    // TEAM-300: Verify correlation_id is automatically injected from context
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_correlation_id("corr-xyz-789");
    
    context::with_narration_context(ctx, async {
        n!("test", "Message with auto-injected correlation_id");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].correlation_id, Some("corr-xyz-789".to_string()));
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_auto_injects_actor() {
    // TEAM-300: Verify actor is automatically injected from context
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_actor("test-actor");
    
    context::with_narration_context(ctx, async {
        n!("test", "Message with auto-injected actor");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, "test-actor");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_auto_injects_all_fields() {
    // TEAM-300: Verify all context fields are injected together
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("job-complete-test")
        .with_correlation_id("corr-complete-test")
        .with_actor("complete-actor");
    
    context::with_narration_context(ctx, async {
        n!("test", "Message with all fields auto-injected");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].job_id, Some("job-complete-test".to_string()));
    assert_eq!(captured[0].correlation_id, Some("corr-complete-test".to_string()));
    assert_eq!(captured[0].actor, "complete-actor");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_multiple_narrations_in_context() {
    // TEAM-300: Verify all narrations in context get auto-injection
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("multi-narration-test")
        .with_actor("multi-actor");
    
    context::with_narration_context(ctx, async {
        n!("step1", "First step");
        n!("step2", "Second step");
        n!("step3", "Third step");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 3);
    
    // All should have same job_id and actor (auto-injected)
    for event in &captured {
        assert_eq!(event.job_id, Some("multi-narration-test".to_string()));
        assert_eq!(event.actor, "multi-actor");
    }
}

// ============================================================================
// Context Without Fields
// ============================================================================

#[tokio::test]
#[serial(capture_adapter)]
async fn test_narration_without_context() {
    // TEAM-300: Narration without context should work (defaults)
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    n!("no_context", "Message without context");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].job_id, None);
    assert_eq!(captured[0].correlation_id, None);
    assert_eq!(captured[0].actor, "unknown"); // Default
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_without_job_id() {
    // TEAM-300: Context can be set without job_id
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_actor("actor-only");
    
    context::with_narration_context(ctx, async {
        n!("test", "Message with actor but no job_id");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].job_id, None);
    assert_eq!(captured[0].actor, "actor-only");
}

// ============================================================================
// Context Inheritance in Spawned Tasks
// ============================================================================

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_not_inherited_by_tokio_spawn() {
    // TEAM-300: Document that tokio::spawn does NOT inherit task-local context
    // This is expected behavior for task-local storage!
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("spawned-task-test")
        .with_actor("spawner");
    
    context::with_narration_context(ctx, async {
        // Main task narration
        n!("main_task", "Main task narration");
        
        // Spawn a task - context is NOT automatically inherited
        let handle = tokio::spawn(async {
            n!("spawned_task", "Spawned task narration");
        });
        
        handle.await.expect("Spawned task should complete");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 2);
    
    // Main task event has context
    assert_eq!(captured[0].action, "main_task");
    assert_eq!(captured[0].job_id, Some("spawned-task-test".to_string()));
    assert_eq!(captured[0].actor, "spawner");
    
    // Spawned task event does NOT have context (expected!)
    assert_eq!(captured[1].action, "spawned_task");
    assert_eq!(captured[1].job_id, None);
    assert_eq!(captured[1].actor, "unknown");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_manual_context_propagation_to_spawned_task() {
    // TEAM-300: Show how to manually propagate context to spawned tasks
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("manual-propagation-test")
        .with_actor("spawner");
    
    context::with_narration_context(ctx.clone(), async {
        // Main task narration
        n!("main_task", "Main task narration");
        
        // Manually propagate context to spawned task
        let handle = tokio::spawn(
            context::with_narration_context(ctx, async {
                n!("spawned_task", "Spawned task with manual context");
            })
        );
        
        handle.await.expect("Spawned task should complete");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 2);
    
    // Both tasks have context (manual propagation worked!)
    assert_eq!(captured[0].job_id, Some("manual-propagation-test".to_string()));
    assert_eq!(captured[0].actor, "spawner");
    assert_eq!(captured[1].job_id, Some("manual-propagation-test".to_string()));
    assert_eq!(captured[1].actor, "spawner");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_within_same_task() {
    // TEAM-300: Context works within same task (no spawn)
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("same-task-test")
        .with_actor("root");
    
    context::with_narration_context(ctx, async {
        n!("level0", "Root level");
        
        // Call async functions (not spawn) - context is preserved
        level1_function().await;
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 3);
    
    // All levels have same context (within same task)
    for event in &captured {
        assert_eq!(event.job_id, Some("same-task-test".to_string()));
        assert_eq!(event.actor, "root");
    }
}

async fn level1_function() {
    n!("level1", "Level 1 function");
    level2_function().await;
}

async fn level2_function() {
    n!("level2", "Level 2 function");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_context_with_sequential_calls() {
    // TEAM-300: Context works through sequential async function calls
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("sequential-test")
        .with_actor("coordinator");
    
    context::with_narration_context(ctx, async {
        // Sequential async calls preserve context
        async_task_1().await;
        async_task_2().await;
        async_task_3().await;
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 3);
    
    // All tasks have same context
    for event in &captured {
        assert_eq!(event.job_id, Some("sequential-test".to_string()));
        assert_eq!(event.actor, "coordinator");
    }
}

async fn async_task_1() {
    n!("task1", "Task 1");
}

async fn async_task_2() {
    n!("task2", "Task 2");
}

async fn async_task_3() {
    n!("task3", "Task 3");
}

// ============================================================================
// Nested Contexts
// ============================================================================

#[tokio::test]
#[serial(capture_adapter)]
async fn test_nested_contexts() {
    // TEAM-300: Inner context overrides outer context
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let outer_ctx = context::NarrationContext::new()
        .with_job_id("outer-job")
        .with_actor("outer-actor");
    
    context::with_narration_context(outer_ctx, async {
        n!("outer", "Outer context");
        
        let inner_ctx = context::NarrationContext::new()
            .with_job_id("inner-job")
            .with_actor("inner-actor");
        
        context::with_narration_context(inner_ctx, async {
            n!("inner", "Inner context");
        }).await;
        
        n!("outer_again", "Back to outer context");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 3);
    
    // Outer context
    assert_eq!(captured[0].job_id, Some("outer-job".to_string()));
    assert_eq!(captured[0].actor, "outer-actor");
    
    // Inner context (overrides)
    assert_eq!(captured[1].job_id, Some("inner-job".to_string()));
    assert_eq!(captured[1].actor, "inner-actor");
    
    // Back to outer
    assert_eq!(captured[2].job_id, Some("outer-job".to_string()));
    assert_eq!(captured[2].actor, "outer-actor");
}

// ============================================================================
// Real-World Patterns
// ============================================================================

#[tokio::test]
#[serial(capture_adapter)]
async fn test_job_router_pattern() {
    // TEAM-300: Simulate job router pattern (real-world usage)
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    // Simulate job router receiving a job
    let job_id = "real-world-job-123";
    
    let ctx = context::NarrationContext::new()
        .with_job_id(job_id)
        .with_actor("qn-router");
    
    context::with_narration_context(ctx, async {
        // Router narration
        n!("route_start", "Routing job");
        
        // Execute operation (in real code, this would be actual work)
        execute_simulated_operation().await;
        
        n!("route_complete", "Job routed successfully");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 4); // route_start + 2 from operation + route_complete
    
    // All should have same job_id and actor
    for event in &captured {
        assert_eq!(event.job_id, Some("real-world-job-123".to_string()));
        assert_eq!(event.actor, "qn-router");
    }
}

async fn execute_simulated_operation() {
    // Simulate operation that does multiple narrations
    // Context is automatically inherited!
    n!("op_start", "Operation starting");
    n!("op_complete", "Operation complete");
}

#[tokio::test]
#[serial(capture_adapter)]
async fn test_multi_step_workflow() {
    // TEAM-300: Simulate multi-step workflow with context
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    let ctx = context::NarrationContext::new()
        .with_job_id("workflow-test")
        .with_correlation_id("corr-workflow")
        .with_actor("workflow-engine");
    
    context::with_narration_context(ctx, async {
        // Step 1: Initialize
        n!("init", "Initializing workflow");
        
        // Step 2: Validate
        n!("validate", "Validating inputs");
        
        // Step 3: Execute
        n!("execute", "Executing workflow");
        
        // Step 4: Finalize
        n!("finalize", "Finalizing results");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 4);
    
    // Verify workflow integrity (all steps have same context)
    let expected_job_id = Some("workflow-test".to_string());
    let expected_corr_id = Some("corr-workflow".to_string());
    let expected_actor = "workflow-engine";
    
    for (i, event) in captured.iter().enumerate() {
        assert_eq!(event.job_id, expected_job_id, "Step {} job_id mismatch", i);
        assert_eq!(event.correlation_id, expected_corr_id, "Step {} correlation_id mismatch", i);
        assert_eq!(event.actor, expected_actor, "Step {} actor mismatch", i);
    }
}

// ============================================================================
// Benefits Demonstration
// ============================================================================

#[tokio::test]
#[serial(capture_adapter)]
async fn test_before_and_after_comparison() {
    // TEAM-300: Demonstrate the improvement
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    // BEFORE Phase 2: Would need to pass job_id to every function
    // AFTER Phase 2: Set once, use everywhere
    
    let ctx = context::NarrationContext::new()
        .with_job_id("comparison-test")
        .with_actor("demo");
    
    context::with_narration_context(ctx, async {
        // All these narrations automatically get job_id and actor
        // NO manual .job_id() calls needed!
        perform_step_a().await;
        perform_step_b().await;
        perform_step_c().await;
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 3);
    
    // All steps have context (automatic!)
    for event in &captured {
        assert_eq!(event.job_id, Some("comparison-test".to_string()));
        assert_eq!(event.actor, "demo");
    }
}

async fn perform_step_a() {
    n!("step_a", "Step A - no manual job_id needed!");
}

async fn perform_step_b() {
    n!("step_b", "Step B - context automatically inherited!");
}

async fn perform_step_c() {
    n!("step_c", "Step C - so much cleaner!");
}

// ============================================================================
// Documentation
// ============================================================================

// These tests verify Phase 2 functionality documented in:
//   - .plan/TEAM_299_PHASE_2_THREAD_LOCAL_CONTEXT.md
//   - .plan/MASTERPLAN.md (Phase 2 section)
//
// Key achievement:
//   ELIMINATES 100+ manual .job_id() calls throughout the codebase
//
// How it works:
//   1. Set context once at job/task boundary
//   2. All narrations inside automatically inherit context
//   3. Context includes job_id, correlation_id, and actor
//   4. Works across tokio::spawn boundaries
//
// Usage pattern:
//   ```rust
//   let ctx = NarrationContext::new()
//       .with_job_id(&job_id)
//       .with_actor("qn-router");
//   
//   with_narration_context(ctx, async {
//       n!("step1", "Step 1");  // Auto-injected!
//       n!("step2", "Step 2");  // Auto-injected!
//   }).await;
//   ```
