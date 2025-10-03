//! Example: Cute Mode — Children's Book Narration
//!
//! This example demonstrates the `cute` field for whimsical storytelling.
//!
//! Run with: cargo run --example cute_mode -p observability-narration-core

use observability_narration_core::{narrate, NarrationFields};

fn main() {
    // Initialize tracing subscriber for pretty output
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!("🎀 Cute Mode Examples — Children's Book Narration\n");

    // Example 1: VRAM Seal (Success)
    println!("📖 Story 1: Tucking a Model into VRAM");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "seal",
        target: "llama-7b".to_string(),
        human: "Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0 (5 ms)".to_string(),
        cute: Some("Tucked llama-7b safely into GPU0's warm 2GB nest! Sweet dreams! 🛏️✨".to_string()),
        device: Some("GPU0".to_string()),
        duration_ms: Some(5),
        ..Default::default()
    });

    println!();

    // Example 2: Seal Verification (Success)
    println!("📖 Story 2: Checking on the Model");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "verify",
        target: "llama-7b".to_string(),
        human: "Verified seal for shard 'llama-7b' on GPU 0 (1 ms)".to_string(),
        cute: Some("Checked on llama-7b — still sleeping soundly! All is well! 🔍💕".to_string()),
        device: Some("GPU0".to_string()),
        duration_ms: Some(1),
        ..Default::default()
    });

    println!();

    // Example 3: VRAM Allocation (Success)
    println!("📖 Story 3: Finding a Cozy Spot");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "allocate",
        target: "GPU1".to_string(),
        human: "Allocated 1024 MB VRAM on GPU 1 (8192 MB available, 3 ms)".to_string(),
        cute: Some("Found a perfect 1GB spot on GPU1! Plenty of room left! 🏠✨".to_string()),
        device: Some("GPU1".to_string()),
        duration_ms: Some(3),
        ..Default::default()
    });

    println!();

    // Example 4: Deallocation (Goodbye)
    println!("📖 Story 4: Saying Goodbye");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "deallocate",
        target: "bert-base".to_string(),
        human: "Deallocated 512 MB VRAM for shard 'bert-base' on GPU 0 (1536 MB still in use)".to_string(),
        cute: Some("Said goodbye to bert-base and tidied up 512 MB! Room for new friends! 👋🧹".to_string()),
        device: Some("GPU0".to_string()),
        ..Default::default()
    });

    println!();

    // Example 5: Error (Insufficient VRAM)
    println!("📖 Story 5: Oh No, Not Enough Room!");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "allocate_failed",
        target: "GPU0".to_string(),
        human: "VRAM allocation failed on GPU 0: requested 4096 MB, only 2048 MB available".to_string(),
        cute: Some("Oh dear! GPU0 doesn't have enough room (need 4GB, only 2GB free). Let's try elsewhere! 😟".to_string()),
        device: Some("GPU0".to_string()),
        error_kind: Some("insufficient_vram".to_string()),
        ..Default::default()
    });

    println!();

    // Example 6: Critical Error (Seal Verification Failed)
    println!("📖 Story 6: Something's Not Right!");
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "verify_failed",
        target: "model-x".to_string(),
        human: "CRITICAL: Seal verification failed for shard 'model-x' on GPU 0: digest mismatch".to_string(),
        cute: Some("Uh oh! model-x's safety seal looks different than expected! Time to investigate! 😟🔍".to_string()),
        device: Some("GPU0".to_string()),
        error_kind: Some("digest_mismatch".to_string()),
        ..Default::default()
    });

    println!();

    // Example 7: Orchestrator Admission (Queue)
    println!("📖 Story 7: Joining the Queue");
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-abc123".to_string(),
        human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'".to_string(),
        cute: Some("Request joins the queue at spot 3 — should start in about 420 ms! 🎫✨".to_string()),
        pool_id: Some("default".to_string()),
        queue_position: Some(3),
        predicted_start_ms: Some(420),
        ..Default::default()
    });

    println!();

    // Example 8: Job Completion
    println!("📖 Story 8: All Done!");
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "complete",
        target: "job-456".to_string(),
        human: "Completed job 'job-456' successfully (2500 ms total)".to_string(),
        cute: Some("Hooray! Finished job-456 perfectly! Great work everyone! 🎉✨".to_string()),
        job_id: Some("job-456".to_string()),
        duration_ms: Some(2500),
        ..Default::default()
    });

    println!("\n✨ End of cute stories! ✨");
}
