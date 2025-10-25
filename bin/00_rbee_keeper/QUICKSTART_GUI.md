# Quick Start: rbee-keeper GUI

**TEAM-293: Get the GUI running in 5 minutes**

## Prerequisites

- Rust toolchain (already installed if you built rbee-keeper CLI)
- Node.js and pnpm (for frontend)

```bash
# Check Node.js
node --version  # Should be v18+

# Install pnpm if needed
npm install -g pnpm
```

## Step 1: Install Dependencies

```bash
cd /home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui
pnpm install
```

## Step 2: Generate Icons (Optional)

```bash
# For now, we'll skip this - Tauri will use defaults
# Later, you can generate proper icons:
# cargo install tauri-cli
# cargo tauri icon path/to/icon.png
```

## Step 3: Run Development Mode

Open two terminals:

**Terminal 1 - Frontend Dev Server:**
```bash
cd /home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui
pnpm dev
```

**Terminal 2 - Tauri App:**
```bash
cd /home/vince/Projects/llama-orch/bin/00_rbee_keeper
cargo tauri dev
```

The GUI window should open automatically!

## Step 4: Explore the Interface

The GUI has 6 main sections:

1. **Status** - View all hives and workers
2. **Queen** - Manage queen-rbee daemon
3. **Hives** - Install, start, stop hives
4. **Workers** - Spawn and manage workers
5. **Models** - Download and manage models
6. **Inference** - Run LLM inference

## Common Tasks

### Start Queen (if not running)

1. Click **Queen** in the navigation
2. Click **Start Queen**
3. Wait for success message

### Start a Hive

1. Click **Hives** in the navigation
2. Scroll to "Start Hive" section
3. Enter host (e.g., "localhost")
4. Enter port (default: 9000)
5. Click **Start Hive**

### Run Inference

1. Click **Inference** in the navigation
2. Enter model (e.g., "HF:TinyLlama/TinyLlama-1.1B-Chat-v1.0")
3. Enter prompt (e.g., "Hello, how are you?")
4. Adjust parameters if needed
5. Click **Run Inference**

## Building for Production

```bash
cd /home/vince/Projects/llama-orch/bin/00_rbee_keeper
cargo tauri build
```

The installer will be in `target/release/bundle/`:
- **Linux**: `.deb` package
- **macOS**: `.app` bundle
- **Windows**: `.msi` installer

## Troubleshooting

### "Cannot find module @tauri-apps/api"

```bash
cd ui
pnpm install
```

### "Port 5173 already in use"

Kill the process using that port or change the port in `ui/vite.config.js`.

### "Queen-rbee not responding"

Make sure queen-rbee is running:
```bash
./rbee-keeper queen start
```

### "Hive not found"

Start a hive first:
```bash
./rbee-keeper hive start --host localhost --port 9000
```

## Architecture

The GUI uses the **same handler functions** as the CLI, so behavior is identical:

```
User clicks button
    ↓
React calls Tauri command
    ↓
Tauri command calls handler (same as CLI)
    ↓
Handler calls queen-rbee HTTP API
    ↓
Response formatted as JSON
    ↓
React displays result
```

## Next Steps

1. Try all the different pages
2. Compare CLI vs GUI for the same operation
3. Read `README_GUI.md` for detailed architecture
4. Build production bundle: `cargo tauri build`

## Getting Help

- CLI still works: `./rbee-keeper --help`
- Both CLI and GUI use the same backend
- Check queen-rbee logs for errors
- See `README_GUI.md` for full documentation
