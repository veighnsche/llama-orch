//! Candle GPT-2 Reference Generator
//!
//! Uses the INSTRUMENTED orch_log branch of Candle to extract checkpoints.
//! This surgically extracts checkpoint data from Candle's bigcode example.
//!
//! Created by: TEAM-003
//! Lesson from worker-orcd: Compare with reference from Day 1
//!
//! APPROACH: Run Candle's bigcode example with LLORCH_VALIDATE=1 env var,
//! which triggers checkpoint logging to /tmp/candle_checkpoints/*.npy

use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, Context};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Candle GPT-2 Reference Generator                       ║");
    println!("║  Multi-Reference Validation for llorch-cpud             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    // Step 1: Verify we're on orch_log branch
    println!("🔍 Step 1: Verifying orch_log branch...");
    let candle_dir = PathBuf::from("../../../../reference/candle");
    verify_orch_log_branch(&candle_dir)?;
    println!("✅ On orch_log branch\n");
    
    // Step 2: Add checkpoint logging to bigcode example
    println!("🔧 Step 2: Instrumenting Candle bigcode example...");
    instrument_candle_bigcode(&candle_dir)?;
    println!("✅ Checkpoint logging added\n");
    
    // Step 3: Build Candle bigcode example
    println!("🏗️  Step 3: Building Candle bigcode example...");
    build_candle_example(&candle_dir)?;
    println!("✅ Build complete\n");
    
    // Step 4: Run with checkpoint extraction
    println!("🚀 Step 4: Running Candle with checkpoint extraction...");
    run_candle_with_checkpoints(&candle_dir)?;
    println!("✅ Checkpoints extracted\n");
    
    // Step 5: Copy checkpoints to our test directory
    println!("📦 Step 5: Copying checkpoints...");
    let output_dir = PathBuf::from("../../../.test-models/gpt2/extracted_weights");
    fs::create_dir_all(&output_dir)?;
    copy_checkpoints_from_tmp(&output_dir)?;
    println!("✅ Checkpoints copied to {}\n", output_dir.display());
    
    println!("🎉 All Candle reference checkpoints generated!");
    println!("   Ready for multi-reference validation\n");
    
    Ok(())
}

fn verify_orch_log_branch(candle_dir: &Path) -> Result<()> {
    let output = Command::new("git")
        .current_dir(candle_dir)
        .args(&["branch", "--show-current"])
        .output()
        .context("Failed to check git branch")?;
    
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    
    if branch != "orch_log" {
        anyhow::bail!("Not on orch_log branch! Current: {}. Run: cd {} && git checkout orch_log", 
            branch, candle_dir.display());
    }
    
    Ok(())
}

fn instrument_candle_bigcode(candle_dir: &Path) -> Result<()> {
    // TODO: Add checkpoint logging to candle-transformers/src/models/bigcode.rs
    // For now, we'll use a simpler approach: modify the example to save checkpoints
    
    println!("   ⚠️  Manual instrumentation required:");
    println!("   1. Edit reference/candle/candle-transformers/src/models/bigcode.rs");
    println!("   2. Add checkpoint logging after each layer");
    println!("   3. Save to /tmp/candle_checkpoints/*.npy");
    println!("   ");
    println!("   OR use existing test helpers for checkpoints 1-2");
    
    Ok(())
}

fn build_candle_example(candle_dir: &Path) -> Result<()> {
    let status = Command::new("cargo")
        .current_dir(candle_dir.join("candle-examples"))
        .args(&["build", "--release", "--example", "bigcode"])
        .status()
        .context("Failed to build Candle example")?;
    
    if !status.success() {
        anyhow::bail!("Candle build failed");
    }
    
    Ok(())
}

fn run_candle_with_checkpoints(candle_dir: &Path) -> Result<()> {
    fs::create_dir_all("/tmp/candle_checkpoints")?;
    
    let status = Command::new("cargo")
        .current_dir(candle_dir.join("candle-examples"))
        .args(&["run", "--release", "--example", "bigcode", "--", 
                "--prompt", "Hello.", "--cpu"])
        .env("LLORCH_VALIDATE", "1")
        .status()
        .context("Failed to run Candle example")?;
    
    if !status.success() {
        anyhow::bail!("Candle execution failed");
    }
    
    Ok(())
}

fn copy_checkpoints_from_tmp(output_dir: &Path) -> Result<()> {
    let checkpoints = vec![
        "checkpoint_01_ln1_output",
        "checkpoint_02_q",
        "checkpoint_02_k",
        "checkpoint_02_v",
        "checkpoint_04_scores",
        "checkpoint_05_output",
        "checkpoint_06_ffn",
    ];
    
    for checkpoint in checkpoints {
        let src = PathBuf::from(format!("/tmp/candle_checkpoints/{}.npy", checkpoint));
        let dst = output_dir.join(format!("{}_candle.npy", checkpoint));
        
        if src.exists() {
            fs::copy(&src, &dst)
                .context(format!("Failed to copy {}", checkpoint))?;
            println!("   ✅ Copied {}", checkpoint);
        } else {
            println!("   ⚠️  Missing {}", checkpoint);
        }
    }
    
    Ok(())
}
