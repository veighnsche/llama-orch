#!/bin/bash
# Setup macOS build environment
# Created by: TEAM-451
# Run this on mac: ./scripts/setup-mac.sh

set -e

echo "🍎 Setting up macOS build environment..."
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script must run on macOS"
    exit 1
fi

echo "📋 System Information:"
echo "  OS: $(sw_vers -productName) $(sw_vers -productVersion)"
echo "  Arch: $(uname -m)"
echo ""

# 1. Check/Install Rust
echo "1️⃣ Checking Rust installation..."
if command -v rustc &> /dev/null; then
    echo "  ✅ Rust installed: $(rustc --version)"
else
    echo "  📥 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  ✅ Rust installed: $(rustc --version)"
fi
echo ""

# 2. Check/Install pnpm
echo "2️⃣ Checking pnpm installation..."
if command -v pnpm &> /dev/null; then
    echo "  ✅ pnpm installed: $(pnpm --version)"
else
    echo "  📥 Installing pnpm..."
    curl -fsSL https://get.pnpm.io/install.sh | sh -
    
    # Add pnpm to PATH for current session
    export PNPM_HOME="$HOME/Library/pnpm"
    export PATH="$PNPM_HOME:$PATH"
    
    # Add to shell profile
    if [[ -f "$HOME/.zshrc" ]]; then
        echo 'export PNPM_HOME="$HOME/Library/pnpm"' >> "$HOME/.zshrc"
        echo 'export PATH="$PNPM_HOME:$PATH"' >> "$HOME/.zshrc"
        echo "  ✅ Added pnpm to ~/.zshrc"
    fi
    
    if [[ -f "$HOME/.bash_profile" ]]; then
        echo 'export PNPM_HOME="$HOME/Library/pnpm"' >> "$HOME/.bash_profile"
        echo 'export PATH="$PNPM_HOME:$PATH"' >> "$HOME/.bash_profile"
        echo "  ✅ Added pnpm to ~/.bash_profile"
    fi
    
    echo "  ✅ pnpm installed: $(pnpm --version)"
fi
echo ""

# 3. Install Node.js via pnpm (if needed)
echo "3️⃣ Checking Node.js installation..."
if command -v node &> /dev/null; then
    echo "  ✅ Node.js installed: $(node --version)"
else
    echo "  📥 Installing Node.js via pnpm..."
    pnpm env use --global lts
    echo "  ✅ Node.js installed: $(node --version)"
fi
echo ""

# 4. Check Xcode Command Line Tools
echo "4️⃣ Checking Xcode Command Line Tools..."
if xcode-select -p &> /dev/null; then
    echo "  ✅ Xcode Command Line Tools installed"
else
    echo "  📥 Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "  ⏳ Please complete the installation in the popup, then re-run this script"
    exit 0
fi
echo ""

# 5. Test build
echo "5️⃣ Testing Rust build..."
if [[ -f "Cargo.toml" ]]; then
    echo "  🔨 Building rbee-keeper (this may take a while)..."
    cargo build --package rbee-keeper --release
    echo "  ✅ Build successful!"
else
    echo "  ⚠️  Not in rbee directory, skipping build test"
fi
echo ""

# 6. Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ macOS build environment setup complete!"
echo ""
echo "Installed:"
echo "  • Rust: $(rustc --version | cut -d' ' -f2)"
echo "  • Cargo: $(cargo --version | cut -d' ' -f2)"
echo "  • pnpm: $(pnpm --version)"
echo "  • Node.js: $(node --version)"
echo ""
echo "Next steps:"
echo "  1. Restart your terminal (or run: source ~/.zshrc)"
echo "  2. Test: cargo build --package rbee-keeper"
echo "  3. Setup GitHub Actions runner (see MAC_SETUP_GUIDE.md)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
