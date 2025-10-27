# Countdown via SSE - Not Stderr!

**Date:** Oct 27, 2025  
**Team:** TEAM-330  
**Status:** ✅ COMPLETE

---

## 🔥 The Problem

**Original implementation:** Countdown progress bar rendered to **stderr** via `indicatif`

```rust
// ❌ OLD: Progress bar to stderr (invisible via HTTP API!)
let pb = ProgressBar::new(total_secs);
pb.set_style(...);
pb.set_message(label.clone());
```

**Why this is wrong:**
- ❌ **HTTP API calls can't see stderr** - Progress invisible to web UI
- ❌ **Server logs only** - User has no visibility
- ❌ **Defeats the purpose** - Timeout enforcement via HTTP should show progress!

---

## ✅ The Solution

**New implementation:** Countdown as **narration events** that go through SSE

```rust
// ✅ NEW: Progress as narration events (goes through SSE!)
n!("progress", "⏱️  {} - {}s / {}s elapsed", label, elapsed, total_secs);
```

**Why this is correct:**
- ✅ **Goes through SSE** - Web UI sees real-time progress
- ✅ **Same channel as all narration** - Consistent architecture
- ✅ **Works everywhere** - Client (stdout) and server (SSE)

---

## 🎯 Architecture

### Before (Broken for HTTP API)

```
TimeoutEnforcer.enforce_with_countdown()
  ↓
indicatif::ProgressBar
  ↓
stderr (LOCAL ONLY)
  ↓
❌ Web UI can't see it!
```

### After (Works for HTTP API)

```
TimeoutEnforcer.enforce_with_countdown()
  ↓
n!("progress", "...")  ← Narration event every second
  ↓
narrate(fields)
  ↓
Check NarrationContext for job_id
  ↓
If job_id exists:
  sse_sink::send_to_job(job_id, event)
    ↓
  MPSC channel
    ↓
  SSE endpoint
    ↓
✅ Web UI sees real-time countdown!
```

---

## 📋 Implementation Details

### Countdown Task

```rust
// TEAM-330: Spawn countdown narration task (goes through SSE!)
let label_clone = label.clone();
let progress_handle = tokio::spawn(async move {
    let mut ticker = interval(Duration::from_secs(1));
    let mut elapsed = 0u64;

    loop {
        ticker.tick().await;
        elapsed += 1;
        
        // TEAM-330: Emit progress as narration (goes through SSE if job_id set!)
        n!("progress", "⏱️  {} - {}s / {}s elapsed", label_clone, elapsed, total_secs);

        if elapsed >= total_secs {
            break;
        }
    }
});
```

**Key points:**
- Emits narration event **every second**
- Goes through **same SSE channel** as all narration
- Automatically includes **job_id** from `NarrationContext`

### Cleanup

```rust
// Stop countdown when operation completes or times out
progress_handle.abort();
```

---

## 🎨 Usage with #[with_job_id]

### InstallConfig with job_id

```rust
use remote_daemon_lifecycle::{InstallConfig, SshConfig};

let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
let config = InstallConfig {
    daemon_name: "llm-worker-rbee".to_string(),
    ssh_config: ssh,
    local_binary_path: None,
    job_id: Some("job-123".to_string()),  // ← Routes countdown through SSE!
};

install_daemon_remote(config).await?;
```

### Function with #[with_job_id] macro

```rust
#[with_job_id(config_param = "install_config")]
#[with_timeout(secs = 300, label = "Install daemon")]
pub async fn install_daemon_remote(install_config: InstallConfig) -> Result<()> {
    // ... implementation ...
}
```

**What happens:**
1. `#[with_job_id]` extracts `job_id` from `install_config`
2. Creates `NarrationContext` with job_id
3. Wraps function body in `with_narration_context()`
4. `#[with_timeout]` enforces 5-minute timeout
5. Countdown narration events include job_id
6. SSE sink routes to job-specific channel
7. **Web UI sees real-time countdown!**

---

## 📊 Narration Flow

### Client-side (No job_id)

```
install_daemon_remote(config)  // config.job_id = None
  ↓
n!("progress", "...")
  ↓
narrate(fields)
  ↓
No job_id in NarrationContext
  ↓
stdout only
```

**Result:** Progress visible in terminal

### Server-side (With job_id)

```
install_daemon_remote(config)  // config.job_id = Some("job-123")
  ↓
#[with_job_id] wraps in NarrationContext
  ↓
n!("progress", "...")
  ↓
narrate(fields)
  ↓
job_id in NarrationContext
  ↓
sse_sink::send_to_job("job-123", event)
  ↓
MPSC channel → SSE endpoint → Web UI
```

**Result:** Progress visible in web UI **AND** server logs

---

## 🔍 Example Output

### Terminal (stdout)

```
⏱️  Install daemon (timeout: 300s)
📦 Installing llm-worker-rbee on vince@192.168.1.100
🔨 Building llm-worker-rbee from source...
⏱️  Install daemon - 1s / 300s elapsed
⏱️  Install daemon - 2s / 300s elapsed
⏱️  Install daemon - 3s / 300s elapsed
...
✅ Build complete: target/release/llm-worker-rbee
📁 Creating ~/.local/bin on remote
📤 Copying llm-worker-rbee to vince@192.168.1.100:~/.local/bin/llm-worker-rbee
⏱️  Install daemon - 45s / 300s elapsed
⏱️  Install daemon - 46s / 300s elapsed
...
🔐 Making binary executable
✅ Verifying installation
🎉 llm-worker-rbee installed successfully on vince@192.168.1.100
```

### Web UI (SSE stream)

Same output, but streamed in real-time via SSE!

---

## ✅ Files Changed

### 1. timeout-enforcer/src/enforcement.rs

**Removed:**
- `indicatif::{ProgressBar, ProgressStyle}` imports
- Progress bar creation and styling
- `pb.set_position()` calls
- `pb.finish_and_clear()` calls

**Added:**
- Countdown narration task with `n!("progress", ...)` every second
- Comments explaining SSE routing

**LOC:** ~150 lines (net: -10 lines, removed indicatif complexity)

### 2. remote-daemon-lifecycle/src/install.rs

**Added:**
- `InstallConfig` struct with `job_id` field
- `#[with_job_id]` macro usage
- Updated function signature to use config
- Documentation for SSE routing

**LOC:** ~220 lines (+22 lines for config struct)

### 3. remote-daemon-lifecycle/Cargo.toml

**Added:**
- `observability-narration-macros` dependency

### 4. remote-daemon-lifecycle/src/lib.rs

**Added:**
- `InstallConfig` export

---

## 🎉 Benefits

### For Users
- ✅ **Real-time progress** - See countdown in web UI
- ✅ **Timeout visibility** - Know exactly how much time remains
- ✅ **Remote observability** - Monitor long operations from anywhere

### For Developers
- ✅ **Consistent architecture** - Countdown uses same channel as narration
- ✅ **No special cases** - Works for client and server
- ✅ **Clean API** - `#[with_job_id]` macro eliminates boilerplate

### For Maintainers
- ✅ **Simpler code** - Removed `indicatif` dependency complexity
- ✅ **Single code path** - No client vs server branches
- ✅ **Better testing** - Narration events are easier to test than stderr output

---

## 🚀 Summary

**Fixed countdown to go through SSE instead of stderr:**

1. ✅ **Removed indicatif progress bar** - Was invisible via HTTP API
2. ✅ **Added narration-based countdown** - Emits event every second
3. ✅ **Routes through SSE** - Same channel as all narration
4. ✅ **Added #[with_job_id] macro** - Automatic context wrapping
5. ✅ **Created InstallConfig** - Clean API with job_id support

**Architecture clarified:**
- Countdown is **narration events**, not stderr output
- Goes through **same SSE channel** as all other narration
- **Automatically includes job_id** from NarrationContext
- **Works everywhere** - client (stdout) and server (SSE)

**The countdown now works correctly for HTTP API calls!** 🎉

---

**TEAM-330: Countdown via SSE complete!** ✅
