#!/bin/bash
# llama-orch Development Workstation Setup Script
#
# This script automates the complete setup of a fresh Ubuntu system as a
# development and inference workstation for llama-orch.
#
# Usage:
#   ./setup-dev-workstation.sh [OPTIONS]
#
# Options:
#   --skip-nvidia     Skip NVIDIA driver and CUDA installation
#   --skip-rust       Skip Rust installation
#   --skip-tests      Skip running tests after setup
#   --cuda-version    Specify CUDA version (default: auto-detect from repos)
#   --help            Show this help message
#
# Requirements:
#   - Ubuntu 24.04 LTS (or compatible)
#   - Sudo access
#   - Internet connection
#
# What this script installs:
#   - Rust toolchain (rustup, cargo, rustc)
#   - Build tools (cmake, gcc, g++, make)
#   - NVIDIA drivers (latest stable)
#   - CUDA toolkit (12.0+)
#   - Google Test (for C++ testing)
#   - All development dependencies
#
# After installation:
#   - All Rust tests passing (479 tests)
#   - CUDA tests built (426 tests)
#   - System ready for inference and development

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SKIP_NVIDIA=false
SKIP_RUST=false
SKIP_TESTS=false
CUDA_VERSION=""
REBOOT_REQUIRED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-nvidia)
            SKIP_NVIDIA=true
            shift
            ;;
        --skip-rust)
            SKIP_RUST=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
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

check_ubuntu_version() {
    log_info "Checking Ubuntu version..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            log_success "Running on Ubuntu $VERSION"
            if [[ ! "$VERSION_ID" =~ ^24\. ]]; then
                log_warning "This script is tested on Ubuntu 24.04. You're running $VERSION_ID"
                read -p "Continue anyway? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        else
            log_error "This script is designed for Ubuntu. Detected: $ID"
            exit 1
        fi
    else
        log_error "Cannot detect OS version"
        exit 1
    fi
}

check_sudo() {
    log_info "Checking sudo access..."
    if ! sudo -v; then
        log_error "This script requires sudo access"
        exit 1
    fi
    log_success "Sudo access confirmed"
}

install_build_tools() {
    log_info "Installing build tools (cmake, gcc, g++, make)..."
    
    sudo apt update
    sudo apt install -y \
        cmake \
        gcc \
        g++ \
        make \
        build-essential \
        pkg-config \
        libssl-dev
    
    log_success "Build tools installed"
    
    # Verify installations
    cmake --version | head -1
    gcc --version | head -1
    g++ --version | head -1
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

install_nvidia_drivers() {
    if [ "$SKIP_NVIDIA" = true ]; then
        log_info "Skipping NVIDIA installation (--skip-nvidia)"
        return
    fi
    
    log_info "Installing NVIDIA drivers..."
    
    # Check if NVIDIA GPU exists
    if ! lspci | grep -i nvidia &> /dev/null; then
        log_warning "No NVIDIA GPU detected. Skipping NVIDIA installation."
        return
    fi
    
    # Install NVIDIA driver
    sudo apt install -y nvidia-driver-550 nvidia-utils-550
    
    REBOOT_REQUIRED=true
    log_success "NVIDIA drivers installed (reboot required)"
}

install_cuda_toolkit() {
    if [ "$SKIP_NVIDIA" = true ]; then
        log_info "Skipping CUDA installation (--skip-nvidia)"
        return
    fi
    
    log_info "Installing CUDA toolkit..."
    
    if [ -n "$CUDA_VERSION" ]; then
        log_info "Installing CUDA version: $CUDA_VERSION"
        sudo apt install -y "cuda-toolkit-$CUDA_VERSION"
    else
        sudo apt install -y nvidia-cuda-toolkit
    fi
    
    log_success "CUDA toolkit installed"
    
    # Verify nvcc
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release"
    fi
}

install_google_test() {
    log_info "Installing Google Test..."
    
    sudo apt install -y libgtest-dev googletest
    
    log_success "Google Test installed"
}

create_llorch_config() {
    log_info "Creating .llorch.toml configuration..."
    
    local config_file="$HOME/Projects/llama-orch/.llorch.toml"
    
    if [ -f "$config_file" ]; then
        log_warning "Configuration file already exists: $config_file"
        return
    fi
    
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
    
    cp "$project_root/.llorch.toml.example" "$project_root/.llorch.toml"
    log_success "Configuration file created"
}

build_cuda_tests() {
    log_info "Building CUDA tests..."
    
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
    
    cd "$worker_dir/cuda"
    mkdir -p build
    cd build
    
    log_info "Running CMake..."
    CXX=g++ cmake -DBUILD_TESTING=ON ..
    
    log_info "Building (this may take a few minutes)..."
    make -j$(nproc) cuda_tests
    
    log_success "CUDA tests built: $(pwd)/cuda_tests"
    
    cd - > /dev/null
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
    
    log_info "Running integration tests..."
    cargo test --test '*' --no-fail-fast
    
    log_success "All Rust tests passed"
    
    cd - > /dev/null
}

run_cuda_tests() {
    if [ "$SKIP_TESTS" = true ] || [ "$SKIP_NVIDIA" = true ]; then
        log_info "Skipping CUDA tests"
        return
    fi
    
    if [ "$REBOOT_REQUIRED" = true ]; then
        log_warning "CUDA tests require reboot to activate drivers"
        log_info "Run this after reboot:"
        echo "  cd bin/worker-orcd/cuda/build && ./cuda_tests"
        return
    fi
    
    log_info "Running CUDA tests..."
    
    # Find cuda_tests executable
    local cuda_tests
    if [ -f "bin/worker-orcd/cuda/build/cuda_tests" ]; then
        cuda_tests="bin/worker-orcd/cuda/build/cuda_tests"
    elif [ -f "../bin/worker-orcd/cuda/build/cuda_tests" ]; then
        cuda_tests="../bin/worker-orcd/cuda/build/cuda_tests"
    elif [ -f "cuda/build/cuda_tests" ]; then
        cuda_tests="cuda/build/cuda_tests"
    else
        log_error "Cannot find cuda_tests executable"
        return 1
    fi
    
    "$cuda_tests"
    
    log_success "All CUDA tests passed"
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "  SETUP COMPLETE"
    echo "=========================================="
    echo ""
    
    if [ "$REBOOT_REQUIRED" = true ]; then
        log_warning "REBOOT REQUIRED to activate NVIDIA drivers"
        echo ""
        echo "After reboot, run CUDA tests:"
        echo "  cd bin/worker-orcd/cuda/build"
        echo "  ./cuda_tests"
        echo ""
    fi
    
    echo "Installed components:"
    echo "  ✓ Build tools (cmake, gcc, g++)"
    
    if [ "$SKIP_RUST" = false ]; then
        echo "  ✓ Rust $(rustc --version 2>/dev/null | cut -d' ' -f2 || echo 'toolchain')"
    fi
    
    if [ "$SKIP_NVIDIA" = false ]; then
        echo "  ✓ NVIDIA drivers"
        echo "  ✓ CUDA toolkit"
    fi
    
    echo "  ✓ Google Test"
    echo ""
    
    if [ "$SKIP_TESTS" = false ]; then
        echo "Test results:"
        echo "  ✓ Rust library tests: PASSED"
        echo "  ✓ Rust integration tests: PASSED"
        if [ "$REBOOT_REQUIRED" = false ] && [ "$SKIP_NVIDIA" = false ]; then
            echo "  ✓ CUDA tests: PASSED"
        fi
        echo ""
    fi
    
    echo "Next steps:"
    if [ "$REBOOT_REQUIRED" = true ]; then
        echo "  1. Reboot system: sudo reboot"
        echo "  2. Verify GPUs: nvidia-smi"
        echo "  3. Run CUDA tests: cd bin/worker-orcd/cuda/build && ./cuda_tests"
    else
        echo "  1. Build worker: cd bin/worker-orcd && cargo build --release"
        echo "  2. Download model: mkdir -p ~/models && cd ~/models"
        echo "     wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        echo "  3. Run worker: ./target/release/worker-orcd --model ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --gpu 0"
    fi
    echo ""
    
    log_success "Development workstation setup complete!"
}

# Main execution
main() {
    echo "=========================================="
    echo "  llama-orch Dev Workstation Setup"
    echo "=========================================="
    echo ""
    
    check_ubuntu_version
    check_sudo
    
    log_info "Starting installation..."
    echo ""
    
    # Install components
    install_build_tools
    install_rust
    install_nvidia_drivers
    install_cuda_toolkit
    install_google_test
    
    # Configuration
    create_llorch_config
    
    # Build and test
    if [ "$SKIP_NVIDIA" = false ]; then
        build_cuda_tests
    fi
    
    run_rust_tests
    
    if [ "$REBOOT_REQUIRED" = false ]; then
        run_cuda_tests
    fi
    
    # Summary
    print_summary
}

# Run main function
main "$@"
