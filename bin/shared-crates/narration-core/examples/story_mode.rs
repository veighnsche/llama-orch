//! Example: Story Mode — Dialogue-Focused Narration
//!
//! This example demonstrates the `story` field for conversation-style debugging.
//! Story mode focuses on dialogue between components, making distributed system
//! interactions read like a screenplay or chat transcript.
//!
//! Run with: cargo run --example story_mode -p observability-narration-core

use observability_narration_core::{narrate, NarrationFields};

fn main() {
    // Initialize tracing subscriber for pretty output
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!("🎭 Story Mode Examples — Dialogue-Focused Narration\n");
    println!("Watch components talk to each other!\n");

    // Example 1: VRAM Request (Success)
    println!("📖 Scene 1: Asking for VRAM");
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "vram_request",
        target: "pool-managerd-3".to_string(),
        human: "Requesting 2048 MB VRAM on GPU 0 for model 'llama-7b'".to_string(),
        cute: Some("Orchestratord politely asks pool-managerd-3 for a cozy 2GB spot! 🏠".to_string()),
        story: Some("\"Do you have 2GB VRAM on GPU0?\" asked orchestratord. \"Yes!\" replied pool-managerd-3, \"Allocating now.\"".to_string()),
        pool_id: Some("default".to_string()),
        device: Some("GPU0".to_string()),
        ..Default::default()
    });

    println!();

    // Example 2: VRAM Request (Failure)
    println!("📖 Scene 2: Not Enough Room");
    narrate(NarrationFields {
        actor: "pool-managerd-3",
        action: "vram_denied",
        target: "orchestratord".to_string(),
        human: "VRAM allocation denied: requested 4096 MB, only 512 MB available on GPU 0".to_string(),
        cute: Some("Oh no! Pool-managerd-3 doesn't have enough room! 😟".to_string()),
        story: Some("\"Do you have 4GB VRAM?\" asked orchestratord. \"No,\" replied pool-managerd-3 sadly, \"only 512MB free.\"".to_string()),
        pool_id: Some("default".to_string()),
        device: Some("GPU0".to_string()),
        error_kind: Some("insufficient_vram".to_string()),
        ..Default::default()
    });

    println!();

    // Example 3: Worker Ready Callback
    println!("📖 Scene 3: Worker Checks In");
    narrate(NarrationFields {
        actor: "worker-gpu0-r1",
        action: "ready_callback",
        target: "pool-managerd-3".to_string(),
        human: "Worker ready with engine llamacpp-v1, 8 slots available".to_string(),
        cute: Some("Worker-gpu0-r1 waves hello and says they're ready to help! 👋✨".to_string()),
        story: Some("\"I'm ready!\" announced worker-gpu0-r1. \"Great!\" said pool-managerd-3, \"I'll mark you as live.\"".to_string()),
        worker_id: Some("worker-gpu0-r1".to_string()),
        pool_id: Some("default".to_string()),
        engine: Some("llamacpp-v1".to_string()),
        ..Default::default()
    });

    println!();

    // Example 4: Job Dispatch
    println!("📖 Scene 4: Dispatching a Job");
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "dispatch",
        target: "job-456".to_string(),
        human: "Dispatching job 'job-456' to worker-gpu0-r1 (ETA 420 ms)".to_string(),
        cute: Some("Orchestratord sends job-456 off to its new friend worker-gpu0-r1! 🎫".to_string()),
        story: Some("\"Can you handle job-456?\" asked orchestratord. \"Absolutely!\" replied worker-gpu0-r1, \"Send it over.\"".to_string()),
        job_id: Some("job-456".to_string()),
        worker_id: Some("worker-gpu0-r1".to_string()),
        predicted_start_ms: Some(420),
        ..Default::default()
    });

    println!();

    // Example 5: Heartbeat Check
    println!("📖 Scene 5: Checking In");
    narrate(NarrationFields {
        actor: "pool-managerd-3",
        action: "heartbeat_check",
        target: "worker-gpu0-r1".to_string(),
        human: "Heartbeat received from worker-gpu0-r1 (last seen 2500 ms ago)".to_string(),
        cute: Some("Pool-managerd-3 checks in: \"You still there?\" \"Yep!\" says worker-gpu0-r1! 💓".to_string()),
        story: Some("\"You still alive?\" asked pool-managerd-3. \"Yep, all good here!\" replied worker-gpu0-r1.".to_string()),
        worker_id: Some("worker-gpu0-r1".to_string()),
        pool_id: Some("default".to_string()),
        ..Default::default()
    });

    println!();

    // Example 6: Cancellation Request
    println!("📖 Scene 6: Cancelling a Job");
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "cancel_request",
        target: "job-789".to_string(),
        human: "Requesting cancellation of job 'job-789' from worker-gpu1-r0".to_string(),
        cute: Some("Orchestratord politely asks to cancel job-789. Worker agrees right away! 🛑".to_string()),
        story: Some("\"Can you cancel job-789?\" asked orchestratord. \"Sure thing!\" replied worker-gpu1-r0, \"Stopping now.\"".to_string()),
        job_id: Some("job-789".to_string()),
        worker_id: Some("worker-gpu1-r0".to_string()),
        ..Default::default()
    });

    println!();

    // Example 7: Model Verification
    println!("📖 Scene 7: Verifying the Model");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "seal_verify",
        target: "llama-7b".to_string(),
        human: "Verifying seal for model 'llama-7b' on GPU 0".to_string(),
        cute: Some("Vram-residency checks on llama-7b — everything looks perfect! 🔍💕".to_string()),
        story: Some("\"Is the seal intact?\" asked vram-residency. \"Yes, verified!\" confirmed GPU0.".to_string()),
        device: Some("GPU0".to_string()),
        model_ref: Some("llama-7b".to_string()),
        ..Default::default()
    });

    println!();

    // Example 8: Multi-Party Conversation
    println!("📖 Scene 8: Three-Way Conversation");
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "pool_query",
        target: "all-pools".to_string(),
        human: "Querying all pools for available capacity".to_string(),
        cute: Some("Orchestratord asks everyone for capacity — three pools wave their hands! 🙋".to_string()),
        story: Some(
            "\"Who has capacity?\" asked orchestratord. \
            \"I do!\" said pool-managerd-1. \
            \"Me too!\" said pool-managerd-2. \
            \"I have 8GB free!\" added pool-managerd-3."
                .to_string(),
        ),
        ..Default::default()
    });

    println!();

    // Example 9: Error Dialogue
    println!("📖 Scene 9: Something Went Wrong");
    narrate(NarrationFields {
        actor: "worker-gpu0-r1",
        action: "inference_error",
        target: "job-999".to_string(),
        human: "CRITICAL: Inference failed for job 'job-999': CUDA out of memory".to_string(),
        cute: Some("Oh no! Worker-gpu0-r1 ran out of memory! \"I'm so sorry!\" 😟💔".to_string()),
        story: Some(
            "\"Processing job-999...\" said worker-gpu0-r1. \
            Suddenly: \"ERROR! Out of memory!\" \
            \"What happened?\" asked orchestratord. \
            \"CUDA OOM,\" replied worker sadly."
                .to_string(),
        ),
        job_id: Some("job-999".to_string()),
        worker_id: Some("worker-gpu0-r1".to_string()),
        error_kind: Some("cuda_oom".to_string()),
        ..Default::default()
    });

    println!();

    // Example 10: Success Celebration
    println!("📖 Scene 10: Job Complete!");
    narrate(NarrationFields {
        actor: "worker-gpu0-r1",
        action: "job_complete",
        target: "job-456".to_string(),
        human: "Completed job 'job-456' successfully (2500 ms, 150 tokens)".to_string(),
        cute: Some("\"All done!\" cheers worker-gpu0-r1! \"Excellent work!\" says orchestratord! 🎉".to_string()),
        story: Some(
            "\"Job done!\" announced worker-gpu0-r1 proudly. \
            \"How'd it go?\" asked orchestratord. \
            \"Perfect! 150 tokens in 2.5 seconds!\" \
            \"Excellent!\" replied orchestratord."
                .to_string(),
        ),
        job_id: Some("job-456".to_string()),
        worker_id: Some("worker-gpu0-r1".to_string()),
        duration_ms: Some(2500),
        tokens_out: Some(150),
        ..Default::default()
    });

    println!("\n✨ End of story mode examples! ✨");
    println!("\n💡 Tip: Use story mode to make distributed system logs read like a screenplay!");
    println!("   Perfect for understanding complex multi-service interactions.");
}
