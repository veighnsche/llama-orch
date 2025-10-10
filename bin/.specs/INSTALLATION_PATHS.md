# Installation Paths - Industry Standards

**TEAM-035**  
**Date:** 2025-10-10

## Problem

When deploying to remote machines via SSH/git, we need standardized paths for:
- Binaries
- Configuration
- Data (models, databases)
- Logs
- Runtime state

## Industry Standards

### XDG Base Directory Specification

Following freedesktop.org standards:

```bash
# User-specific binaries
~/.local/bin/rbee
~/.local/bin/rbee-hive
~/.local/bin/llm-worker-rbee

# User-specific configuration
~/.config/rbee/config.toml
~/.config/rbee/pools.toml

# User-specific data
~/.local/share/rbee/models/
~/.local/share/rbee/models.db

# User-specific cache
~/.cache/rbee/downloads/

# Runtime state
~/.local/state/rbee/logs/
```

### System-wide Installation (Optional)

```bash
# System binaries
/usr/local/bin/rbee
/usr/local/bin/rbee-hive
/usr/local/bin/llm-worker-rbee

# System configuration
/etc/rbee/config.toml

# System data
/var/lib/rbee/models/
/var/lib/rbee/models.db

# Logs
/var/log/rbee/

# Runtime
/run/rbee/
```

## Implementation

### 1. Binary Installation

```bash
# Install to user bin (default)
cargo install --path bin/rbee-keeper --root ~/.local
cargo install --path bin/rbee-hive --root ~/.local
cargo install --path bin/llm-worker-rbee --root ~/.local

# Or use install script
./scripts/install.sh --user
./scripts/install.sh --system  # requires sudo
```

### 2. Configuration Discovery

Priority order:
1. `RBEE_CONFIG` environment variable
2. `~/.config/rbee/config.toml`
3. `/etc/rbee/config.toml`
4. Built-in defaults

### 3. Data Directories

```toml
# ~/.config/rbee/config.toml
[paths]
models_dir = "~/.local/share/rbee/models"
catalog_db = "~/.local/share/rbee/models.db"
cache_dir = "~/.cache/rbee"
log_dir = "~/.local/state/rbee/logs"
```

### 4. Environment Variables

```bash
# Override paths
export RBEE_CONFIG=~/.config/rbee/config.toml
export RBEE_MODELS_DIR=~/.local/share/rbee/models
export RBEE_CATALOG_DB=~/.local/share/rbee/models.db
export RBEE_CACHE_DIR=~/.cache/rbee
export RBEE_LOG_DIR=~/.local/state/rbee/logs

# Runtime overrides
export RBEE_WORKER_HOST=127.0.0.1
export RBEE_MODEL_BASE_DIR=.test-models  # For development
```

## Remote Deployment

### SSH-based Deployment

```bash
# 1. Clone repo on remote
ssh mac.home.arpa "git clone https://github.com/user/llama-orch ~/llama-orch"

# 2. Build on remote
ssh mac.home.arpa "cd ~/llama-orch && cargo build --release"

# 3. Install binaries
ssh mac.home.arpa "cd ~/llama-orch && ./scripts/install.sh --user"

# 4. Create config
ssh mac.home.arpa "mkdir -p ~/.config/rbee && cat > ~/.config/rbee/config.toml << 'EOF'
[pool]
name = \"mac\"
listen_addr = \"0.0.0.0:8080\"

[paths]
models_dir = \"~/.local/share/rbee/models\"
catalog_db = \"~/.local/share/rbee/models.db\"
EOF"

# 5. Start daemon
ssh mac.home.arpa "~/.local/bin/rbee-hive daemon"
```

### Using rbee CLI

```bash
# Deploy to remote pool
rbee pool deploy --host mac.home.arpa

# This does:
# 1. Check if git repo exists
# 2. Pull latest changes
# 3. Build binaries
# 4. Install to ~/.local/bin
# 5. Create default config if missing
# 6. Restart daemon
```

## Directory Structure

```
~/.local/
├── bin/
│   ├── rbee              # CLI
│   ├── rbee-hive         # Pool manager daemon
│   └── llm-worker-rbee   # Worker daemon
├── share/
│   └── rbee/
│       ├── models/       # Downloaded models
│       │   ├── tinyllama/
│       │   ├── qwen/
│       │   └── phi3/
│       └── models.db     # SQLite catalog
└── state/
    └── rbee/
        └── logs/
            ├── rbee-hive.log
            └── worker-*.log

~/.config/
└── rbee/
    ├── config.toml       # Main config
    └── pools.toml        # Known pools

~/.cache/
└── rbee/
    └── downloads/        # Temporary download cache
```

## Migration from Current Setup

### Current (Development)
```bash
# Binaries
target/debug/rbee
target/debug/rbee-hive
target/debug/llm-worker-rbee

# Models
.test-models/

# Config
.llorch.toml.example
```

### New (Production)
```bash
# Binaries
~/.local/bin/rbee
~/.local/bin/rbee-hive
~/.local/bin/llm-worker-rbee

# Models
~/.local/share/rbee/models/

# Config
~/.config/rbee/config.toml
```

## Install Script

Create `scripts/install.sh`:

```bash
#!/bin/bash
# Install rbee binaries and setup directories

set -e

INSTALL_TYPE="${1:---user}"

if [ "$INSTALL_TYPE" = "--user" ]; then
    BIN_DIR="$HOME/.local/bin"
    CONFIG_DIR="$HOME/.config/rbee"
    DATA_DIR="$HOME/.local/share/rbee"
    STATE_DIR="$HOME/.local/state/rbee"
elif [ "$INSTALL_TYPE" = "--system" ]; then
    BIN_DIR="/usr/local/bin"
    CONFIG_DIR="/etc/rbee"
    DATA_DIR="/var/lib/rbee"
    STATE_DIR="/var/log/rbee"
else
    echo "Usage: $0 [--user|--system]"
    exit 1
fi

echo "Installing rbee to $BIN_DIR..."

# Create directories
mkdir -p "$BIN_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR/models"
mkdir -p "$STATE_DIR/logs"

# Build and install binaries
cargo build --release
cp target/release/rbee "$BIN_DIR/"
cp target/release/rbee-hive "$BIN_DIR/"
cp target/release/llm-worker-rbee "$BIN_DIR/"

# Create default config if missing
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    cat > "$CONFIG_DIR/config.toml" << EOF
[pool]
name = "$(hostname)"
listen_addr = "0.0.0.0:8080"

[paths]
models_dir = "$DATA_DIR/models"
catalog_db = "$DATA_DIR/models.db"
log_dir = "$STATE_DIR/logs"
EOF
    echo "Created default config at $CONFIG_DIR/config.toml"
fi

echo "✅ Installation complete!"
echo ""
echo "Binaries installed to: $BIN_DIR"
echo "Config directory: $CONFIG_DIR"
echo "Data directory: $DATA_DIR"
echo ""
echo "Add to PATH if needed:"
echo "  export PATH=\"$BIN_DIR:\$PATH\""
```

## Usage After Installation

```bash
# Start pool manager
rbee-hive daemon

# Use CLI
rbee pool models list --host mac.home.arpa
rbee infer --node mac --model "hf:..." --prompt "..."

# Check installation
which rbee
rbee --version
```

---

**Status:** Specification complete  
**Next:** Implement install script and config loading
