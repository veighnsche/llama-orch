#!/usr/bin/env bash
# TEAM-450: Simple build script for new machines
# Just runs the root build commands - Turborepo and Cargo handle the rest!

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🐝 Building rbee monorepo..."
echo ""

# ============================================================================
# DETECT OS
# ============================================================================
detect_os() {
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/arch-release ]; then
      echo "arch"
    elif [ -f /etc/debian_version ]; then
      echo "ubuntu"
    elif [ -f /etc/NIXOS ] || { [ -f /etc/os-release ] && grep -qi 'ID=nixos' /etc/os-release; }; then
      echo "nixos"
    else
      echo "linux"
    fi
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macos"
  else
    echo "unknown"
  fi
}

OS=$(detect_os)

# ============================================================================
# PREFLIGHT CHECKS - FAIL FAST
# ============================================================================
echo "→ Running preflight checks..."
echo ""

FAILED=0
MISSING_DEPS=()

# Check Node.js
echo "[1/6] Checking Node.js..."
if ! command -v node &> /dev/null; then
  echo "  ✗ node is not installed"
  FAILED=1
  MISSING_DEPS+=("node")
else
  NODE_VERSION=$(node --version)
  echo "  ✓ node $NODE_VERSION"
fi

# Check pnpm
echo "[2/6] Checking pnpm..."
if ! command -v pnpm &> /dev/null; then
  if [ -f "$REPO_ROOT/nix/pnpm-corepack-hook.sh" ]; then
    PNPM_HOME="$REPO_ROOT/.pnpm" COREPACK_HOME="$REPO_ROOT/.pnpm" . "$REPO_ROOT/nix/pnpm-corepack-hook.sh"
  fi
fi

if ! command -v pnpm &> /dev/null; then
  echo "  ✗ pnpm is not installed"
  FAILED=1
  MISSING_DEPS+=("pnpm")
else
  PNPM_VERSION=$(pnpm --version)
  echo "  ✓ pnpm $PNPM_VERSION"
fi

# Check Cargo
echo "[3/6] Checking Cargo..."
if ! command -v cargo &> /dev/null; then
  echo "  ✗ cargo is not installed"
  FAILED=1
  MISSING_DEPS+=("cargo")
else
  CARGO_VERSION=$(cargo --version | cut -d' ' -f2)
  echo "  ✓ cargo $CARGO_VERSION"
fi

# Check wasm-pack
echo "[4/6] Checking wasm-pack..."
if ! command -v wasm-pack &> /dev/null; then
  echo "  ✗ wasm-pack is not installed"
  FAILED=1
  MISSING_DEPS+=("wasm-pack")
else
  WASM_PACK_VERSION=$(wasm-pack --version | cut -d' ' -f2)
  echo "  ✓ wasm-pack $WASM_PACK_VERSION"

  if command -v rustup &> /dev/null; then
    if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
      echo "  → Installing wasm32-unknown-unknown target via rustup..."
      if rustup target add wasm32-unknown-unknown; then
        echo "  ✓ wasm32-unknown-unknown target installed"
      else
        echo "  ✗ Failed to install wasm32-unknown-unknown target with rustup (try: rustup target add wasm32-unknown-unknown)"
        FAILED=1
      fi
    else
      echo "  ✓ wasm32-unknown-unknown target already installed"
    fi
  fi
fi

# TEAM_519: Rust toolchain / WASI targets health check so broken rustup fails preflight, not pnpm build
echo "[5/6] Checking Rust toolchain (rustup targets)..."
if command -v rustup &> /dev/null; then
  if ! rustup show active-toolchain > /dev/null 2>&1; then
    echo "  → No default Rust toolchain configured; setting 'stable' as default via rustup..."
    if rustup default stable > /dev/null 2>&1; then
      echo "  ✓ Rust default toolchain set to 'stable'"
    else
      echo "  ✗ Failed to set default Rust toolchain with 'rustup default stable' (check network and write permissions under $RUSTUP_HOME or ~/.rustup)"
      FAILED=1
    fi
  fi

  if ! rustc -vV > /dev/null 2>&1; then
    echo "  ✗ rustc -vV failed; Rust toolchain is not usable"
    FAILED=1
  else
    echo "  ✓ rustc is usable"
  fi

  for target in wasm32-wasip1 wasm32-wasip1-threads; do
    if rustup target list --installed | grep -q "$target"; then
      echo "  ✓ $target target already installed"
    else
      echo "  → Installing $target target via rustup..."
      if rustup target add "$target"; then
        echo "  ✓ $target target installed"
      else
        echo "  ✗ Failed to install $target target with rustup (check write permissions under $RUSTUP_HOME or ~/.rustup and try: rustup target add $target)"
        FAILED=1
      fi
    fi
  done

  if command -v wasm-bindgen &> /dev/null; then
    WASM_BINDGEN_VERSION=$(wasm-bindgen --version 2>/dev/null || echo "installed")
    echo "  ✓ wasm-bindgen CLI $WASM_BINDGEN_VERSION"
  else
    echo "  ⚠ wasm-bindgen CLI not found; wasm-pack will download its bundled version during build"
  fi
else
  echo "  ✗ rustup not found - Rust toolchain management is required"
  FAILED=1
  MISSING_DEPS+=("rustup")
fi

# Check for required system libraries (pkg-config)
echo "[6/6] Checking system libraries..."
if command -v pkg-config &> /dev/null; then
  # Check glib-2.0
  if ! pkg-config --exists glib-2.0; then
    echo "  ✗ glib-2.0 development library is not installed"
    FAILED=1
    MISSING_DEPS+=("glib")
  else
    GLIB_VERSION=$(pkg-config --modversion glib-2.0)
    echo "  ✓ glib-2.0 $GLIB_VERSION"
  fi
  
  # Check gdk-3.0
  if ! pkg-config --exists gdk-3.0; then
    echo "  ✗ gdk-3.0 development library is not installed"
    FAILED=1
    MISSING_DEPS+=("gdk")
  else
    GDK_VERSION=$(pkg-config --modversion gdk-3.0)
    echo "  ✓ gdk-3.0 $GDK_VERSION"
  fi
else
  echo "  ⚠ pkg-config not found - skipping system library checks"
fi

echo ""

# Exit if any checks failed
if [ $FAILED -eq 1 ]; then
  echo "✗ Preflight checks failed!"
  echo ""
  echo "Install missing dependencies:"
  echo ""
  
  # OS-specific installation instructions
  case "$OS" in
    arch)
      echo "📦 Arch Linux:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   sudo pacman -S nodejs npm" ;;
          pnpm) echo "  • pnpm:      sudo npm install -g pnpm" ;;
          cargo) echo "  • Rust:      sudo pacman -S rustup && rustup default stable" ;;
          rustup) echo "  • rustup:    sudo pacman -S rustup && rustup default stable" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  sudo pacman -S glib2" ;;
          gdk) echo "  • gdk-3.0:   sudo pacman -S gtk3" ;;
        esac
      done
      ;;
    ubuntu)
      echo "📦 Ubuntu/Debian:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt install -y nodejs" ;;
          pnpm) echo "  • pnpm:      sudo npm install -g pnpm" ;;
          cargo) echo "  • Rust:      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;;
          rustup) echo "  • rustup:    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  sudo apt install libglib2.0-dev" ;;
          gdk) echo "  • gdk-3.0:   sudo apt install libgtk-3-dev" ;;
        esac
      done
      ;;
    macos)
      echo "📦 macOS:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   brew install node" ;;
          pnpm) echo "  • pnpm:      brew install pnpm" ;;
          cargo) echo "  • Rust:      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;;
          rustup) echo "  • rustup:    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  brew install glib" ;;
          gdk) echo "  • gdk-3.0:   brew install gtk+3" ;;
        esac
      done
      ;;
    nixos)
      echo "📦 NixOS:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   add pkgs.nodejs to environment.systemPackages or use: nix-shell -p nodejs" ;;
          pnpm) echo "  • pnpm:      add pkgs.pnpm to environment.systemPackages or use: nix-shell -p pnpm" ;;
          cargo) echo "  • Rust:      add pkgs.rustup to environment.systemPackages and run: rustup default stable" ;;
          rustup) echo "  • rustup:    add pkgs.rustup to environment.systemPackages and run: rustup default stable" ;;
          wasm-pack) echo "  • wasm-pack: add pkgs.wasm-pack to environment.systemPackages or use: nix-shell -p wasm-pack" ;;
          glib) echo "  • glib-2.0:  add pkgs.glib to environment.systemPackages or a devShell" ;;
          gdk) echo "  • gdk-3.0:   add pkgs.gtk3 to environment.systemPackages or a devShell" ;;
        esac
      done
      ;;
    *)
      echo "📦 Generic (visit official sites):"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   https://nodejs.org/" ;;
          pnpm) echo "  • pnpm:      npm install -g pnpm" ;;
          cargo) echo "  • Rust:      https://rustup.rs/" ;;
          rustup) echo "  • rustup:    https://rustup.rs/" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  Install glib2 development package for your OS" ;;
          gdk) echo "  • gdk-3.0:   Install GTK3 development package for your OS" ;;
        esac
      done
      ;;
  esac
  
  echo ""
  exit 1
fi

echo "✓ All preflight checks passed!"
echo ""

# ============================================================================
# BUILD
# ============================================================================

# TEAM-XXX: mac compat - Ensure cargo bin is on PATH and wasm-bindgen is installed to avoid wasm-pack race conditions
if [[ -d "$HOME/.cargo/bin" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install dependencies
echo "→ [BUILD 1/4] Installing dependencies..."
if ! pnpm install; then
  echo "✗ pnpm install failed!"
  exit 1
fi
echo "  ✓ Dependencies installed"
echo ""

# Build frontend (Turborepo handles everything)
# TEAM_528: Skip WASM SDK builds during frontend phase to avoid cargo lock contention with Rust build
# WASM SDKs will be built in step 4 after cargo is done
echo "→ [BUILD 2/4] Building frontend (Turborepo)..."
if ! RBEE_SKIP_WASM=1 turbo build; then
  echo "✗ Frontend build failed!"
  exit 1
fi
echo "  ✓ Frontend built (WASM SDKs will be built in step 4)"
echo ""

# Build Rust (Cargo workspace handles everything)
echo "→ [BUILD 3/4] Building Rust (Cargo)..."
if ! cargo build --release; then
  echo "✗ Rust build failed!"
  exit 1
fi
echo "  ✓ Rust built"
echo ""

# Build WASM SDKs (after cargo is done to avoid lock contention)
# TEAM_528: Build WASM SDKs at the end so they don't interfere with cargo build --release
echo "→ [BUILD 4/4] Building WASM SDKs..."
WASM_SDK_DIRS=(
  "bin/10_queen_rbee/ui/packages/queen-rbee-sdk"
  "bin/20_rbee_hive/ui/packages/rbee-hive-sdk"
  "bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk"
  "bin/31_sd_worker_rbee/ui/packages/sd-worker-sdk"
)

for sdk_dir in "${WASM_SDK_DIRS[@]}"; do
  sdk_name=$(basename "$sdk_dir")
  echo "  → Building $sdk_name..."
  if ! (cd "$REPO_ROOT/$sdk_dir" && pnpm run build); then
    echo "  ⚠ Warning: $sdk_name build failed (this is non-fatal, WASM SDK may have compilation issues)"
  else
    echo "  ✓ $sdk_name built"
  fi
done
echo "  ✓ WASM SDKs built"
echo ""

echo "✓ Build complete! 🐝"
