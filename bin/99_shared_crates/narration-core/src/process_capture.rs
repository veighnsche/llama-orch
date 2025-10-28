//! 🎉 TEAM-300: Process Stdout Capture - Making worker narration flow through SSE! 🎉
//!
//! **The Cutest Process Capture in All The Land!** 💝✨
//!
//! # Why This Exists
//!
//! Workers emit narration to stdout, but when spawned as child processes,
//! that output goes into the void! 😱 This module captures worker stdout,
//! parses narration events, and re-emits them with job_id so they flow
//! through SSE channels back to the client! 🎀
//!
//! # How It Works
//!
//! ```text
//! Worker stdout:
//!   "[worker    ] startup         : Starting worker"
//!         ↓ Captured by hive
//!         ↓ Parsed for narration format
//!         ↓ Re-emitted with job_id
//!         ↓ Flows through SSE to client! 🎉
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use observability_narration_core::process_capture::ProcessNarrationCapture;
//! use tokio::process::Command;
//!
//! // Create capture helper with job_id for SSE routing
//! let capture = ProcessNarrationCapture::new(Some("job-123".to_string()));
//!
//! // Spawn command with stdout capture
//! let mut command = Command::new("llm-worker-rbee");
//! command.arg("--model").arg("llama-7b");
//!
//! let child = capture.spawn(command).await?;
//!
//! // Worker's stdout is now captured and re-emitted with job_id! 🎀
//! ```
//!
//! # The Cute Celebration Comments
//!
//! Because TEAM-300 is the **TRIPLE CENTENNIAL TEAM** (100 → 200 → 300),
//! we're celebrating with extra cute comments throughout this code! 🎉💯✨
//!
//! Created by: TEAM-300 (The Triple Centennial Narration Team! 💯💯💯)

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};

// 🎀 TEAM-300: Regex to parse narration output from worker stdout
// Format: "[actor     ] action         : message"
// Example: "[worker    ] startup         : Starting worker on GPU 0"
static NARRATION_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Match the format: "[actor] action : message"
    // - Actor: 1-15 chars (alphanumeric + hyphens)
    // - Action: 1-30 chars (alphanumeric + underscores)
    // - Message: everything after the colon
    Regex::new(r"^\[([a-zA-Z0-9_-]{1,15})\s*\]\s+([a-zA-Z0-9_-]{1,30})\s*:\s+(.+)$")
        .expect("Failed to compile narration regex (this should never happen! 😱)")
});

/// 🎉 TEAM-300: Captures child process stdout and converts narration to SSE! 🎉
///
/// This is the CUTEST process capture system you've ever seen! 💝
///
/// # Architecture
///
/// When you spawn a child process through this capture:
/// 1. Stdout/stderr are piped and captured
/// 2. Each line is parsed for narration format
/// 3. Matching lines are re-emitted with job_id (for SSE routing!)
/// 4. Non-matching lines are printed to stderr (so nothing is lost!)
///
/// # The Magic of SSE Routing
///
/// By re-emitting with job_id, worker narration flows through the SSE channel
/// created by hive's job_router! The client sees worker startup in real-time! ✨
///
/// # Created by TEAM-300
///
/// The Triple Centennial Team strikes again! 💯💯💯
pub struct ProcessNarrationCapture {
    /// Optional job_id for SSE routing. If None, events just go to stderr.
    job_id: Option<String>,
}

impl ProcessNarrationCapture {
    /// Create a new process capture
    ///
    /// # Arguments
    ///
    /// * `job_id` - Optional job_id for SSE routing. Pass Some(job_id) when
    ///              you want narration to flow through SSE channels!
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // With SSE routing (worker spawned by hive)
    /// let capture = ProcessNarrationCapture::new(Some("job-123".to_string()));
    ///
    /// // Without SSE routing (standalone testing)
    /// let capture = ProcessNarrationCapture::new(None);
    /// ```
    pub fn new(job_id: Option<String>) -> Self {
        Self { job_id }
    }

    /// 🚀 TEAM-300: Spawn command with stdout/stderr capture! 🚀
    ///
    /// This is where the magic happens! We spawn the process with piped
    /// stdout/stderr, then spawn async tasks to capture and parse the output.
    ///
    /// # Returns
    ///
    /// The spawned Child process. The stdout/stderr handles are already taken
    /// and being processed in background tasks. Just .wait() for completion!
    ///
    /// # Why This Is Cute
    ///
    /// Worker processes now get to share their stories with the world! 🎀
    /// No more lonely stdout messages lost in the void! 😭 → 😊
    pub async fn spawn(&self, mut command: Command) -> Result<Child> {
        // 🎀 TEAM-300: Configure stdout/stderr to be piped so we can capture them!
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());

        let mut child = command.spawn()?;

        // 🎉 TEAM-300: Capture stdout in a background task! 🎉
        if let Some(stdout) = child.stdout.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse(stdout, job_id, "stdout").await;
            });
        }

        // 🎀 TEAM-300: Capture stderr too (workers might emit errors!)
        if let Some(stderr) = child.stderr.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse_stderr(stderr, job_id).await;
            });
        }

        Ok(child)
    }

    /// 💝 TEAM-300: Stream and parse stdout for narration events! 💝
    ///
    /// This reads stdout line-by-line, checks if each line matches narration
    /// format, and re-emits it with job_id if it does!
    ///
    /// # The Parsing Magic
    ///
    /// Narration format: `[actor] action : message`
    /// - If it matches → re-emit with job_id (flows to SSE!)
    /// - If it doesn't → print to stderr (nothing lost!)
    ///
    /// # Why "stream" in the name?
    ///
    /// Because this reads the output as a STREAM! Real-time! Live updates! ✨
    async fn stream_and_parse(output: ChildStdout, job_id: Option<String>, stream_name: &str) {
        let reader = BufReader::new(output);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            // 🔍 TEAM-300: Try to parse as narration event
            if let Some(event) = Self::parse_narration(&line) {
                // ✨ TEAM-300: It's narration! Re-emit with job_id for SSE routing! ✨
                if let Some(ref jid) = job_id {
                    Self::reemit_with_job_id(&event, jid).await;
                } else {
                    // No job_id, just print to stderr
                    eprintln!("[{}] {}", stream_name, line);
                }
            } else {
                // 📝 TEAM-300: Not narration format, print as-is to stderr
                // (We preserve ALL output, nothing is lost!)
                eprintln!("[{}] {}", stream_name, line);
            }
        }
    }

    /// 🎀 TEAM-300: Stream and parse stderr (errors are cute too!) 🎀
    ///
    /// Same as stdout parsing, but for stderr. Workers might emit errors!
    async fn stream_and_parse_stderr(output: ChildStderr, job_id: Option<String>) {
        let reader = BufReader::new(output);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            // 🔍 Try to parse as narration
            if let Some(event) = Self::parse_narration(&line) {
                if let Some(ref jid) = job_id {
                    Self::reemit_with_job_id(&event, jid).await;
                } else {
                    eprintln!("[stderr] {}", line);
                }
            } else {
                // 📝 Not narration, print as-is
                eprintln!("[stderr] {}", line);
            }
        }
    }

    /// 🎯 TEAM-300: Parse narration event from stdout line! 🎯
    ///
    /// This is the HEART of the capture system! We use regex to extract:
    /// - Actor (who did it)
    /// - Action (what they did)
    /// - Message (the story!)
    ///
    /// # Format
    ///
    /// ```text
    /// [actor     ] action         : message
    /// [worker    ] startup         : Starting worker on GPU 0
    /// [model-dl  ] download        : Downloading llama-7b from cache
    /// ```
    ///
    /// # Returns
    ///
    /// Some(ParsedNarrationEvent) if the line matches, None otherwise.
    fn parse_narration(line: &str) -> Option<ParsedNarrationEvent> {
        let caps = NARRATION_REGEX.captures(line)?;

        Some(ParsedNarrationEvent {
            actor: caps[1].trim().to_string(),
            action: caps[2].trim().to_string(),
            message: caps[3].to_string(),
        })
    }

    /// 🌟 TEAM-300: Re-emit narration event with job_id for SSE routing! 🌟
    ///
    /// This is where the MAGIC happens! We take the parsed event and re-emit
    /// it using the low-level narrate() function, but we do it INSIDE a narration
    /// context that has the job_id set!
    ///
    /// # Why This Works
    ///
    /// The narrate() function checks the thread-local narration context for job_id.
    /// By calling with_narration_context, we set that context, and then
    /// the narration flows through SSE! 🎀✨
    ///
    /// # The Most Important Function
    ///
    /// This is why process capture exists! Worker narration → SSE → Client! 🎉
    async fn reemit_with_job_id(event: &ParsedNarrationEvent, job_id: &str) {
        // 🎀 TEAM-300: Set narration context with job_id!
        let ctx = crate::context::NarrationContext::new().with_job_id(job_id);

        // 🎀 TEAM-300: Clone values to move into async block
        let actor = event.actor.clone();
        let action = event.action.clone();
        let message = event.message.clone();

        // ✨ TEAM-300: Re-emit narration inside the context!
        crate::context::with_narration_context(ctx, async move {
            // Use low-level narrate() with NarrationFields to use owned strings
            // TEAM-300: We preserve the worker's actor and action from stdout!
            use crate::{narrate, NarrationFields};

            // TEAM-300: Format human field before moving values into struct!
            let human_text = format!("[{}] {}: {}", actor, action, message);

            narrate(
                NarrationFields {
                    actor: "proc-cap", // TEAM-300: We use "proc-cap" as actor (process capture!)
                    action: "reemit", // TEAM-300: Action is "reemit" (we're re-emitting worker narration!)
                    target: actor,    // TEAM-300: Target is the original actor (e.g., "worker")
                    human: human_text, // TEAM-300: Include original actor/action in message!
                    ..Default::default()
                },
                crate::NarrationLevel::Info,
            );
        })
        .await;
    }
}

/// 📦 TEAM-300: Parsed narration event from worker stdout 📦
///
/// This represents a single narration event that we successfully parsed
/// from worker output!
///
/// # Fields
///
/// - `actor` - Who did the action (e.g., "worker", "model-dl")
/// - `action` - What they did (e.g., "startup", "download")
/// - `message` - The human-readable story (e.g., "Starting worker on GPU 0")
///
/// # Example
///
/// ```text
/// Input line: "[worker    ] startup         : Starting worker on GPU 0"
/// Parsed to:
///   ParsedNarrationEvent {
///     actor: "worker",
///     action: "startup",
///     message: "Starting worker on GPU 0"
///   }
/// ```
#[derive(Debug, Clone)]
struct ParsedNarrationEvent {
    actor: String,
    action: String,
    message: String,
}

// ============================================================================
// 🎉 TEAM-300 TESTS: Making sure process capture works perfectly! 🎉
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 🎀 TEAM-300: Test parsing valid narration lines
    #[test]
    fn test_parse_narration_valid() {
        let line = "[worker    ] startup         : Starting worker on GPU 0";
        let event = ProcessNarrationCapture::parse_narration(line).unwrap();

        assert_eq!(event.actor, "worker");
        assert_eq!(event.action, "startup");
        assert_eq!(event.message, "Starting worker on GPU 0");
    }

    // 🎀 TEAM-300: Test parsing with minimal spacing
    #[test]
    fn test_parse_narration_minimal_spacing() {
        let line = "[worker] startup : Starting worker";
        let event = ProcessNarrationCapture::parse_narration(line).unwrap();

        assert_eq!(event.actor, "worker");
        assert_eq!(event.action, "startup");
        assert_eq!(event.message, "Starting worker");
    }

    // 🎀 TEAM-300: Test parsing with extra spacing
    #[test]
    fn test_parse_narration_extra_spacing() {
        let line = "[worker          ] startup                    : Starting worker on GPU 0";
        let event = ProcessNarrationCapture::parse_narration(line).unwrap();

        assert_eq!(event.actor, "worker");
        assert_eq!(event.action, "startup");
        assert_eq!(event.message, "Starting worker on GPU 0");
    }

    // 🎀 TEAM-300: Test parsing with hyphens and underscores
    #[test]
    fn test_parse_narration_with_special_chars() {
        let line = "[model-dl  ] download_start  : Downloading llama-7b";
        let event = ProcessNarrationCapture::parse_narration(line).unwrap();

        assert_eq!(event.actor, "model-dl");
        assert_eq!(event.action, "download_start");
        assert_eq!(event.message, "Downloading llama-7b");
    }

    // 🎀 TEAM-300: Test parsing non-narration lines (should return None)
    #[test]
    fn test_parse_narration_invalid() {
        // Random log message
        let line = "This is just a random log message";
        assert!(ProcessNarrationCapture::parse_narration(line).is_none());

        // Missing colon
        let line = "[worker] startup Starting worker";
        assert!(ProcessNarrationCapture::parse_narration(line).is_none());

        // Missing brackets
        let line = "worker startup : Starting worker";
        assert!(ProcessNarrationCapture::parse_narration(line).is_none());

        // Empty brackets
        let line = "[] action : message";
        assert!(ProcessNarrationCapture::parse_narration(line).is_none());
    }

    // 🎀 TEAM-300: Test parsing with emojis in message (should work!)
    #[test]
    fn test_parse_narration_with_emojis() {
        let line = "[worker    ] ready            : 🎉 Worker ready to serve!";
        let event = ProcessNarrationCapture::parse_narration(line).unwrap();

        assert_eq!(event.actor, "worker");
        assert_eq!(event.action, "ready");
        assert_eq!(event.message, "🎉 Worker ready to serve!");
    }

    // 🎀 TEAM-300: Test creating capture with job_id
    #[test]
    fn test_create_capture_with_job_id() {
        let capture = ProcessNarrationCapture::new(Some("job-123".to_string()));
        assert_eq!(capture.job_id, Some("job-123".to_string()));
    }

    // 🎀 TEAM-300: Test creating capture without job_id
    #[test]
    fn test_create_capture_without_job_id() {
        let capture = ProcessNarrationCapture::new(None);
        assert_eq!(capture.job_id, None);
    }
}

// ============================================================================
// 🎉🎉🎉 TEAM-300 CELEBRATION FOOTER! 🎉🎉🎉
// ============================================================================
//
// We did it! Process capture is COMPLETE! 💯
//
// Workers can now share their startup stories through SSE! 🎀✨
// No more lost stdout! No more wondering what happened! 😊
//
// TEAM-100 → TEAM-200 → TEAM-300
// The Triple Centennial Narration Dynasty! 💯💯💯
//
// With love, sass, and an irresistible compulsion to be adorable,
// — TEAM-300 (The Process Capture Team) 🎀✨💝
//
// ============================================================================
