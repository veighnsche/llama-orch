#!/bin/bash
# llama-orch Development Workstation Setup Script for macOS
#
# This script automates the complete setup of a macOS system as a
# development workstation for llama-orch (CPU-only, no CUDA).
#
# Usage:
#   ./setup-dev-workstation-macos.sh [OPTIONS]
#
# Options:
#   --skip-rust       Skip Rust installation
#   --skip-tests      Skip running tests after setup
#   --skip-homebrew   Skip Homebrew installation
#   --help            Show this help message
#
# Requirements:
#   - macOS 12.0+ (Monterey or later)
#   - Internet connection
#   - Xcode Command Line Tools (will be installed if missing)
#
# What this script installs:
#   - Xcode Command Line Tools
#   - Homebrew (if not present)
#   - Rust toolchain (rustup, cargo, rustc)
#   - CMake (for future Metal support)
#   - Build tools
#
# After installation:
#   - All Rust tests passing (479 tests)
#   - System ready for CPU-only inference and development
#
# Note: For GPU acceleration on Apple Silicon, Metal backend
#       implementation is required (see APPLE_ARM_PORTING_ANALYSIS.md)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SKIP_RUST=false
SKIP_TESTS=false
SKIP_HOMEBREW=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-rust)
            SKIP_RUST=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-homebrew)
            SKIP_HOMEBREW=true
            shift
            ;;
        --help)
            grep '^#' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_macos_version() {
    log_info "Checking macOS version..."
    
    local version=$(sw_vers -productVersion)
    local major=$(echo "$version" | cut -d. -f1)
    
    log_success "Running on macOS $version"
    
    if [ "$major" -lt 12 ]; then
        log_warning "This script is tested on macOS 12.0+. You're running $version"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

install_xcode_tools() {
    log_info "Checking Xcode Command Line Tools..."
    
    if xcode-select -p &> /dev/null; then
        log_success "Xcode Command Line Tools already installed"
        xcode-select -p
    else
        log_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        
        log_warning "Please complete the Xcode installation in the dialog"
        log_warning "Press Enter when installation is complete..."
        read
        
        if xcode-select -p &> /dev/null; then
            log_success "Xcode Command Line Tools installed"
        else
            log_error "Xcode Command Line Tools installation failed"
            exit 1
        fi
    fi
}

install_homebrew() {
    if [ "$SKIP_HOMEBREW" = true ]; then
        log_info "Skipping Homebrew installation (--skip-homebrew)"
        return
    fi
    
    log_info "Checking Homebrew..."
    
    if command -v brew &> /dev/null; then
        log_success "Homebrew already installed: $(brew --version | head -1)"
    else
        log_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon
        if [[ $(uname -m) == 'arm64' ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        
        log_success "Homebrew installed"
    fi
}

install_build_tools() {
    log_info "Installing build tools..."
    
    # Install CMake (for future Metal support)
    if command -v cmake &> /dev/null; then
        log_success "CMake already installed: $(cmake --version | head -1)"
    else
        log_info "Installing CMake via Homebrew..."
        brew install cmake
        log_success "CMake installed"
    fi
    
    # Verify tools
    cmake --version | head -1
    clang --version | head -1
}

install_rust() {
    if [ "$SKIP_RUST" = true ]; then
        log_info "Skipping Rust installation (--skip-rust)"
        return
    fi
    
    log_info "Installing Rust toolchain..."
    
    if command -v rustc &> /dev/null; then
        log_warning "Rust already installed: $(rustc --version)"
        read -p "Reinstall? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    # Install rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Source cargo env
    source "$HOME/.cargo/env"
    
    log_success "Rust installed: $(rustc --version)"
}

create_llorch_config() {
    log_info "Creating .llorch.toml configuration..."
    
    # Find the project root
    local project_root
    if [ -f "../../.llorch.toml.example" ]; then
        project_root="../.."
    elif [ -f "../.llorch.toml.example" ]; then
        project_root=".."
    elif [ -f ".llorch.toml.example" ]; then
        project_root="."
    else
        log_warning "Cannot find .llorch.toml.example, skipping config creation"
        return
    fi
    
    local config_file="$project_root/.llorch.toml"
    
    if [ -f "$config_file" ]; then
        log_warning "Configuration file already exists: $config_file"
        return
    fi
    
    # Create config with CUDA disabled for macOS
    cat > "$config_file" << 'EOF'
# llama-orch Local Build Configuration (macOS)

[build]
# CUDA support (disabled on macOS - no NVIDIA GPUs)
cuda = false

# Auto-detect CUDA toolkit (disabled on macOS)
auto_detect_cuda = false

[development]
# Enable verbose build output (default: false)
verbose_build = false

# Skip CUDA tests (always true on macOS)
skip_cuda_tests = true
EOF
    
    log_success "Configuration file created (CUDA disabled for macOS)"
}

run_rust_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_info "Skipping tests (--skip-tests)"
        return
    fi
    
    log_info "Running Rust tests..."
    
    # Find worker-orcd directory
    local worker_dir
    if [ -d "bin/worker-orcd" ]; then
        worker_dir="bin/worker-orcd"
    elif [ -d "../bin/worker-orcd" ]; then
        worker_dir="../bin/worker-orcd"
    elif [ -d "worker-orcd" ]; then
        worker_dir="worker-orcd"
    else
        log_error "Cannot find worker-orcd directory"
        return 1
    fi
    
    cd "$worker_dir"
    
    # Source cargo env
    source "$HOME/.cargo/env"
    
    log_info "Running library tests..."
    cargo test --lib --no-fail-fast
    
    log_success "All Rust tests passed"
    
    cd - > /dev/null
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "  SETUP COMPLETE"
    echo "=========================================="
    echo ""
    
    echo "Installed components:"
    echo "  ✓ Xcode Command Line Tools"
    
    if [ "$SKIP_HOMEBREW" = false ]; then
        echo "  ✓ Homebrew $(brew --version | head -1 | cut -d' ' -f2)"
    fi
    
    echo "  ✓ CMake $(cmake --version | head -1 | cut -d' ' -f3)"
    
    if [ "$SKIP_RUST" = false ]; then
        echo "  ✓ Rust $(rustc --version 2>/dev/null | cut -d' ' -f2 || echo 'toolchain')"
    fi
    
    echo ""
    
    if [ "$SKIP_TESTS" = false ]; then
        echo "Test results:"
        echo "  ✓ Rust library tests: PASSED"
        echo ""
    fi
    
    echo "Platform: macOS (CPU-only)"
    echo "  ℹ️  CUDA not available on macOS"
    echo "  ℹ️  For GPU acceleration, Metal backend is required"
    echo "  ℹ️  See: bin/worker-orcd/APPLE_ARM_PORTING_ANALYSIS.md"
    echo ""
    
    echo "Next steps:"
    echo "  1. Build worker: cd bin/worker-orcd && cargo build --release"
    echo "  2. Download model: mkdir -p ~/models && cd ~/models"
    echo "     curl -L -o qwen-0.5b.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    echo "  3. Run worker (CPU): ./target/release/worker-orcd --model ~/models/qwen-0.5b.gguf"
    echo ""
    
    log_success "Development workstation setup complete!"
}

# Main execution
main() {
    echo "=========================================="
    echo "  llama-orch macOS Dev Setup"
    echo "=========================================="
    echo ""
    
    check_macos_version
    
    log_info "Starting installation..."
    echo ""
    
    # Install components
    install_xcode_tools
    install_homebrew
    install_build_tools
    install_rust
    
    # Configuration
    create_llorch_config
    
    # Test
    run_rust_tests
    
    # Summary
    print_summary
}

# Run main function
main "$@"
