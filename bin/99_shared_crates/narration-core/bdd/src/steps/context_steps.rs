// TEAM-307: Context propagation step definitions

use crate::steps::world::World;
use cucumber::{gherkin::Step, given, then, when};
use observability_narration_core::{n, with_narration_context, NarrationContext};

// ============================================================================
// Given Steps - Setup Context
// ============================================================================

#[given(regex = r#"^a narration context with job_id "([^"]+)"$"#)]
async fn context_with_job_id(world: &mut World, job_id: String) {
    world.context = Some(NarrationContext::new().with_job_id(job_id));
}

#[given(regex = r#"^a narration context with correlation_id "([^"]+)"$"#)]
async fn context_with_correlation_id(world: &mut World, correlation_id: String) {
    world.context = Some(NarrationContext::new().with_correlation_id(correlation_id));
}

#[given(regex = r#"^a narration context with actor "([^"]+)"$"#)]
async fn context_with_actor(world: &mut World, actor: String) {
    let actor_str = match actor.as_str() {
        "test-actor" => "test-actor",
        "full-actor" => "full-actor",
        _ => "unknown",
    };
    world.context = Some(NarrationContext::new().with_actor(actor_str));
}

#[given(regex = r#"^a narration context with:$"#)]
async fn context_with_fields(world: &mut World, step: &Step) {
    let mut ctx = NarrationContext::new();

    if let Some(table) = step.table.as_ref() {
        for row in table.rows.iter().skip(1) {
            let field = &row[0];
            let value = &row[1];

            match field.as_str() {
                "job_id" => ctx = ctx.with_job_id(value.clone()),
                "correlation_id" => ctx = ctx.with_correlation_id(value.clone()),
                "actor" => {
                    let actor_str = match value.as_str() {
                        "full-actor" => "full-actor",
                        _ => "unknown",
                    };
                    ctx = ctx.with_actor(actor_str);
                }
                _ => {}
            }
        }
    }

    world.context = Some(ctx);
}

#[given("an empty narration context")]
async fn empty_context(world: &mut World) {
    world.context = Some(NarrationContext::new());
}

#[given(regex = r#"^an outer context with job_id "([^"]+)"$"#)]
async fn outer_context(world: &mut World, job_id: String) {
    world.outer_context = Some(NarrationContext::new().with_job_id(job_id));
}

#[given(regex = r#"^an inner context with job_id "([^"]+)"$"#)]
async fn inner_context(world: &mut World, job_id: String) {
    world.inner_context = Some(NarrationContext::new().with_job_id(job_id));
}

#[given(regex = r#"^two concurrent tasks with different contexts:$"#)]
async fn concurrent_contexts(world: &mut World, step: &Step) {
    if let Some(table) = step.table.as_ref() {
        for row in table.rows.iter().skip(1) {
            let task = &row[0];
            let job_id = &row[1];

            match task.as_str() {
                "A" => world.context_a = Some(NarrationContext::new().with_job_id(job_id.clone())),
                "B" => world.context_b = Some(NarrationContext::new().with_job_id(job_id.clone())),
                _ => {}
            }
        }
    }
}

// ============================================================================
// When Steps - Emit Narration
// ============================================================================

#[when(regex = r#"^I emit narration with n!\("([^"]+)", "([^"]+)"\) in context$"#)]
async fn emit_in_context(world: &mut World, action: String, message: String) {
    if let Some(ctx) = world.context.clone() {
        let action_static: &'static str = Box::leak(action.into_boxed_str());
        with_narration_context(ctx, async move {
            n!(action_static, "{}", message);
        })
        .await;
    }
}

#[when(regex = r#"^I emit narration with n!\("([^"]+)", "([^"]+)"\) without context$"#)]
async fn emit_without_context(_world: &mut World, action: String, message: String) {
    let action_static: &'static str = Box::leak(action.into_boxed_str());
    n!(action_static, "{}", message);
}

#[when(regex = r#"^I emit multiple narrations in same context:$"#)]
async fn emit_multiple_in_context(world: &mut World, step: &Step) {
    if let Some(ctx) = world.context.clone() {
        if let Some(table) = step.table.as_ref() {
            let rows: Vec<(String, String)> =
                table.rows.iter().skip(1).map(|row| (row[0].clone(), row[1].clone())).collect();

            with_narration_context(ctx, async move {
                for (action, message) in rows {
                    let action_static: &'static str = Box::leak(action.into_boxed_str());
                    n!(action_static, "{}", message);
                }
            })
            .await;
        }
    }
}

#[when("I emit narration before await")]
async fn emit_before_await(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        with_narration_context(ctx, async move {
            n!("before", "Before await");
        })
        .await;
    }
}

#[when(regex = r#"^I await for (\d+) milliseconds$"#)]
async fn await_delay(_world: &mut World, ms: u64) {
    tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
}

#[when("I emit narration after await")]
async fn emit_after_await(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        with_narration_context(ctx, async move {
            n!("after", "After await");
        })
        .await;
    }
}

#[when("I manually propagate context to spawned task")]
async fn manual_propagate(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        let handle = tokio::spawn(async move {
            with_narration_context(ctx, async move {
                n!("spawned", "From spawned task");
            })
            .await;
        });
        handle.await.unwrap();
    }
}

#[when("spawned task emits narration")]
async fn spawned_emits(_world: &mut World) {
    // Already emitted in previous step
}

#[when("I spawn task without manual propagation")]
async fn spawn_without_propagation(_world: &mut World) {
    let handle = tokio::spawn(async move {
        n!("spawned_no_ctx", "From spawned task without context");
    });
    handle.await.unwrap();
}

#[when("both tasks emit narration concurrently")]
async fn concurrent_emit(world: &mut World) {
    let ctx_a = world.context_a.clone();
    let ctx_b = world.context_b.clone();

    let handle_a = tokio::spawn(async move {
        if let Some(ctx) = ctx_a {
            with_narration_context(ctx, async move {
                n!("task_a", "Task A message");
            })
            .await;
        }
    });

    let handle_b = tokio::spawn(async move {
        if let Some(ctx) = ctx_b {
            with_narration_context(ctx, async move {
                n!("task_b", "Task B message");
            })
            .await;
        }
    });

    handle_a.await.unwrap();
    handle_b.await.unwrap();
}

#[when("I emit narration in outer context")]
async fn emit_in_outer(world: &mut World) {
    if let Some(ctx) = world.outer_context.clone() {
        with_narration_context(ctx, async move {
            n!("outer", "Outer context");
        })
        .await;
    }
}

#[when("I emit narration in inner context")]
async fn emit_in_inner(world: &mut World) {
    if let Some(ctx) = world.inner_context.clone() {
        with_narration_context(ctx, async move {
            n!("inner", "Inner context");
        })
        .await;
    }
}

#[when("I emit narration in outer context again")]
async fn emit_in_outer_again(world: &mut World) {
    if let Some(ctx) = world.outer_context.clone() {
        with_narration_context(ctx, async move {
            n!("outer_again", "Outer context again");
        })
        .await;
    }
}

#[when("I use tokio::select! with context")]
async fn use_select(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        with_narration_context(ctx, async move {
            tokio::select! {
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(10)) => {
                    n!("select_branch_a", "Branch A selected");
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(20)) => {
                    n!("select_branch_b", "Branch B selected");
                }
            }
        })
        .await;
    }
}

#[when("selected branch emits narration")]
async fn selected_emits(_world: &mut World) {
    // Already emitted in previous step
}

#[when("I use tokio::timeout with context")]
async fn use_timeout(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        with_narration_context(ctx, async move {
            let result = tokio::time::timeout(tokio::time::Duration::from_millis(100), async {
                n!("timeout_operation", "Operation before timeout");
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            })
            .await;
            if result.is_ok() {
                n!("timeout_success", "Operation completed");
            }
        })
        .await;
    }
}

#[when("operation emits narration before timeout")]
async fn emits_before_timeout(_world: &mut World) {
    // Already emitted in previous step
}

#[when("I emit narration before channel send")]
async fn emit_before_channel(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);

        with_narration_context(ctx.clone(), async move {
            n!("before_send", "Before channel send");
            tx.send("message".to_string()).await.ok();
        })
        .await;

        // Receive in separate context to test isolation
        tokio::spawn(async move {
            rx.recv().await;
        });
    }
}

#[when("I send message through channel")]
async fn send_through_channel(_world: &mut World) {
    // Already sent in previous step
}

#[when("I emit narration after channel receive")]
async fn emit_after_channel(world: &mut World) {
    if let Some(ctx) = world.context.clone() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);

        tx.send("message".to_string()).await.ok();

        with_narration_context(ctx, async move {
            rx.recv().await;
            n!("after_receive", "After channel receive");
        })
        .await;
    }
}

#[when(regex = r#"^I use futures::join_all with (\d+) futures$"#)]
async fn use_join_all(world: &mut World, count: usize) {
    if let Some(ctx) = world.context.clone() {
        let mut futures = Vec::new();

        for i in 0..count {
            let ctx_clone = ctx.clone();
            let future = async move {
                with_narration_context(ctx_clone, async move {
                    n!("join_all_task", "Task {}", i);
                })
                .await;
            };
            futures.push(future);
        }

        futures::future::join_all(futures).await;
    }
}

#[when("each future emits narration")]
async fn each_future_emits(_world: &mut World) {
    // Already emitted in previous step
}

#[when(regex = r#"^I create (\d+) levels of nested async calls$"#)]
async fn create_nested_calls(world: &mut World, levels: usize) {
    if let Some(ctx) = world.context.clone() {
        with_narration_context(ctx, async move {
            nested_call(0, levels).await;
        })
        .await;
    }
}

fn nested_call(
    current: usize,
    max: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(async move {
        n!("nested_level", "Level {}", current);
        if current < max {
            nested_call(current + 1, max).await;
        }
    })
}

#[when("each level emits narration")]
async fn each_level_emits(_world: &mut World) {
    // Already emitted in previous step
}

// ============================================================================
// Then Steps - Assertions
// ============================================================================

#[then(regex = r#"^event (\d+) should have job_id "([^"]+)"$"#)]
async fn event_has_job_id(world: &mut World, event_num: usize, job_id: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(
            event_num > 0 && event_num <= captured.len(),
            "Event {} not found, only {} events captured",
            event_num,
            captured.len()
        );
        let event = &captured[event_num - 1];
        assert_eq!(
            event.job_id.as_deref(),
            Some(job_id.as_str()),
            "Event {} job_id mismatch",
            event_num
        );
    }
}

#[then(regex = r#"^event (\d+) should have correlation_id "([^"]+)"$"#)]
async fn event_has_correlation_id(world: &mut World, event_num: usize, correlation_id: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let event = &captured[event_num - 1];
        assert_eq!(event.correlation_id.as_deref(), Some(correlation_id.as_str()));
    }
}

#[then(regex = r#"^event (\d+) should have actor "([^"]+)"$"#)]
async fn event_has_actor(world: &mut World, event_num: usize, actor: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let event = &captured[event_num - 1];
        assert_eq!(event.actor, actor);
    }
}

#[then(regex = r#"^event (\d+) should have action "([^"]+)"$"#)]
async fn event_has_action(world: &mut World, event_num: usize, action: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let event = &captured[event_num - 1];
        assert_eq!(event.action, action);
    }
}

#[then(regex = r#"^event (\d+) should NOT have job_id$"#)]
async fn event_no_job_id(world: &mut World, event_num: usize) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let event = &captured[event_num - 1];
        assert!(event.job_id.is_none(), "Event {} should not have job_id", event_num);
    }
}

#[then(regex = r#"^event (\d+) should NOT have correlation_id$"#)]
async fn event_no_correlation_id(world: &mut World, event_num: usize) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let event = &captured[event_num - 1];
        assert!(event.correlation_id.is_none());
    }
}

#[then(regex = r#"^all events should have job_id "([^"]+)"$"#)]
async fn all_events_have_job_id(world: &mut World, job_id: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        for (i, event) in captured.iter().enumerate() {
            assert_eq!(
                event.job_id.as_deref(),
                Some(job_id.as_str()),
                "Event {} job_id mismatch",
                i + 1
            );
        }
    }
}

#[then(regex = r#"^event with action "([^"]+)" should have job_id "([^"]+)"$"#)]
async fn event_with_action_has_job_id(world: &mut World, action: String, job_id: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let event = captured
            .iter()
            .find(|e| e.action == action)
            .expect(&format!("Event with action '{}' not found", action));
        assert_eq!(event.job_id.as_deref(), Some(job_id.as_str()));
    }
}
