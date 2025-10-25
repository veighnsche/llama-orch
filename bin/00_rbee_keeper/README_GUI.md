# rbee-keeper GUI

**TEAM-293: Tauri-based GUI for rbee-keeper**

A graphical user interface for all rbee-keeper CLI commands, built with Tauri and React.

## Architecture

```
rbee-keeper/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── tauri_main.rs        # GUI entry point (Tauri app)
│   ├── lib.rs               # Shared library
│   ├── tauri_commands.rs    # Tauri command handlers
│   ├── cli/                 # CLI argument definitions
│   ├── handlers/            # Business logic (shared by CLI & GUI)
│   ├── config.rs            # Configuration
│   └── job_client.rs        # HTTP client
├── ui/                      # React frontend
│   ├── src/
│   │   ├── main.jsx         # React entry point
│   │   ├── App.jsx          # Main app with routing
│   │   ├── pages/           # Page components
│   │   │   ├── Status.jsx
│   │   │   ├── Queen.jsx
│   │   │   ├── Hives.jsx
│   │   │   ├── Workers.jsx
│   │   │   ├── Models.jsx
│   │   │   └── Inference.jsx
│   │   └── index.css        # Styles
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── tauri.conf.json          # Tauri configuration
├── build.rs                 # Tauri build script
└── Cargo.toml               # Rust dependencies
```

## Key Design Principles

### 1. **Shared Business Logic**

Both CLI and GUI use the **same handler functions**. This ensures:
- ✅ Consistent behavior across interfaces
- ✅ Single source of truth for business logic
- ✅ Bugs fixed once benefit both interfaces

```rust
// CLI uses handlers directly
handle_hive(HiveAction::Start { ... }, &queen_url).await

// GUI wraps handlers in Tauri commands
#[tauri::command]
pub async fn hive_start(...) -> Result<String, String> {
    handlers::handle_hive(HiveAction::Start { ... }, &queen_url).await
}
```

### 2. **Thin Tauri Layer**

Tauri commands in `src/tauri_commands.rs` are **thin wrappers** that:
1. Parse GUI input
2. Call existing handlers
3. Format response as JSON

This keeps the Tauri-specific code minimal and maintainable.

### 3. **Type Safety**

All commands return `Result<String, String>` where the string is JSON:
```typescript
{
  "success": boolean,
  "message": string,
  "data": string | null
}
```

## Building the GUI

### Prerequisites

```bash
# Install Tauri CLI
cargo install tauri-cli

# Install Node dependencies
cd ui
pnpm install
```

### Development Mode

```bash
# Terminal 1: Start React dev server
cd ui
pnpm dev

# Terminal 2: Start Tauri app (watches Rust changes)
cargo tauri dev
```

### Production Build

```bash
# Build the GUI application
cargo tauri build

# Binary will be at:
# - Linux: target/release/bundle/deb/rbee-keeper-gui_0.1.0_amd64.deb
# - macOS: target/release/bundle/macos/rbee-keeper-gui.app
# - Windows: target/release/bundle/msi/rbee-keeper-gui_0.1.0_x64.msi
```

### CLI-only Build

```bash
# Build just the CLI (no GUI)
cargo build --bin rbee-keeper --release
```

## Available Commands

All CLI commands are available in the GUI:

### Status
- Get system status (all hives and workers)

### Queen Management
- Start/Stop/Status
- Rebuild (with/without local hive)
- Build info
- Install/Uninstall

### Hive Management
- List all hives
- Install/Uninstall hive
- Start/Stop hive
- Get hive details
- Check hive status
- Refresh capabilities

### Worker Management
- Spawn worker
- List worker processes
- Get process details
- Kill worker process

### Model Management
- Download model
- List models
- Get model details
- Delete model

### Inference
- Run inference with full parameter control:
  - Model selection
  - Prompt input
  - Max tokens
  - Temperature
  - Top P / Top K
  - Device selection
  - Worker ID
  - Streaming

## File Organization

### Rust Side

- **`src/lib.rs`**: Library exposing shared code
- **`src/main.rs`**: CLI entry point (unchanged)
- **`src/tauri_main.rs`**: GUI entry point (registers Tauri commands)
- **`src/tauri_commands.rs`**: Tauri command wrappers (all 30+ commands)

### Frontend Side

- **`ui/src/App.jsx`**: Main app with React Router
- **`ui/src/pages/*.jsx`**: Page components for each section
- **`ui/src/index.css`**: Global styles (dark theme)

## Icon Generation

To generate icons for your app:

```bash
# Create a 1024x1024 PNG icon
# Then run:
cargo tauri icon path/to/your-icon.png

# This generates all required icon sizes in icons/
```

## Testing

```bash
# Test CLI (as before)
cargo test --bin rbee-keeper

# Test library
cargo test --lib

# Test GUI (manually)
cargo tauri dev
```

## Distribution

### Linux

```bash
cargo tauri build
# Output: target/release/bundle/deb/rbee-keeper-gui_*.deb
sudo dpkg -i target/release/bundle/deb/rbee-keeper-gui_*.deb
```

### macOS

```bash
cargo tauri build
# Output: target/release/bundle/macos/rbee-keeper-gui.app
# Drag to Applications folder
```

### Windows

```bash
cargo tauri build
# Output: target/release/bundle/msi/rbee-keeper-gui_*.msi
# Double-click to install
```

## Comparison: CLI vs GUI

| Feature | CLI | GUI |
|---------|-----|-----|
| **Entry Point** | `src/main.rs` | `src/tauri_main.rs` |
| **Interface** | Terminal | Desktop app |
| **Input** | Command line args | Forms |
| **Output** | stdout/stderr | Visual cards |
| **Scripting** | ✅ Excellent | ❌ Not scriptable |
| **User-Friendly** | Moderate | ✅ High |
| **Remote SSH** | ✅ Works everywhere | Desktop only |
| **Distribution** | Single binary | OS-specific bundle |

## When to Use Which?

### Use CLI when:
- ✅ Scripting/automation
- ✅ Remote SSH sessions
- ✅ CI/CD pipelines
- ✅ Server environments

### Use GUI when:
- ✅ Interactive exploration
- ✅ Visual feedback needed
- ✅ Desktop environment
- ✅ New users learning the system

## Code Quality

All code follows engineering rules:

- ✅ **TEAM-293 signatures** on all new files
- ✅ **No TODO markers** (all features implemented)
- ✅ **Shared logic** between CLI and GUI
- ✅ **Type-safe** Tauri commands
- ✅ **Documented** architecture and usage

## Next Steps

1. **Generate Icons**
   ```bash
   cargo tauri icon icons/source.png
   ```

2. **Install Frontend Dependencies**
   ```bash
   cd ui && pnpm install
   ```

3. **Run Development Mode**
   ```bash
   cargo tauri dev
   ```

4. **Build Production Bundle**
   ```bash
   cargo tauri build
   ```

## Troubleshooting

### "tauri command not found"
```bash
cargo install tauri-cli
```

### "Failed to resolve module @tauri-apps/api"
```bash
cd ui && pnpm install
```

### "Port 5173 already in use"
```bash
# Kill the process using port 5173
# Or change port in ui/vite.config.js
```

### "Build failed: missing icons"
```bash
# Generate icons first
cargo tauri icon icons/your-icon.png
```

## References

- [Tauri Documentation](https://tauri.app/v1/guides/)
- [React Documentation](https://react.dev)
- [Vite Documentation](https://vitejs.dev)
