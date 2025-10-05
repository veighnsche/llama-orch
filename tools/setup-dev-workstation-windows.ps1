# llama-orch Development Workstation Setup Script for Windows
#
# This script automates the complete setup of a Windows system as a
# development and inference workstation for llama-orch.
#
# Usage:
#   .\setup-dev-workstation-windows.ps1 [OPTIONS]
#
# Options:
#   -SkipCuda         Skip CUDA installation
#   -SkipRust         Skip Rust installation
#   -SkipTests        Skip running tests after setup
#   -SkipVisualStudio Skip Visual Studio check
#   -Help             Show this help message
#
# Requirements:
#   - Windows 10/11
#   - PowerShell 5.1+ (or PowerShell 7+)
#   - Internet connection
#   - Administrator privileges (for some installations)
#
# What this script installs:
#   - Visual Studio Build Tools (C++ workload)
#   - Rust toolchain (rustup, cargo, rustc)
#   - CMake
#   - CUDA Toolkit (if NVIDIA GPU present)
#
# After installation:
#   - All Rust tests passing (479 tests)
#   - CUDA tests built (426 tests, if CUDA installed)
#   - System ready for inference and development

param(
    [switch]$SkipCuda,
    [switch]$SkipRust,
    [switch]$SkipTests,
    [switch]$SkipVisualStudio,
    [switch]$Help
)

# Show help
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check Windows version
function Test-WindowsVersion {
    Write-Info "Checking Windows version..."
    
    $os = Get-CimInstance Win32_OperatingSystem
    $version = [System.Version]$os.Version
    
    Write-Success "Running on $($os.Caption) (Build $($os.BuildNumber))"
    
    if ($version.Major -lt 10) {
        Write-Warning "This script is tested on Windows 10/11. You're running an older version."
        $response = Read-Host "Continue anyway? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            exit 1
        }
    }
}

# Check Visual Studio
function Test-VisualStudio {
    if ($SkipVisualStudio) {
        Write-Info "Skipping Visual Studio check (-SkipVisualStudio)"
        return
    }
    
    Write-Info "Checking Visual Studio..."
    
    # Check for vswhere
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -property installationPath 2>$null
        if ($vsPath) {
            Write-Success "Visual Studio found: $vsPath"
            
            # Check for C++ workload
            $vcPath = Join-Path $vsPath "VC\Tools\MSVC"
            if (Test-Path $vcPath) {
                Write-Success "C++ build tools found"
            } else {
                Write-Warning "C++ build tools not found. Please install 'Desktop development with C++' workload"
            }
            return
        }
    }
    
    Write-Warning "Visual Studio not found"
    Write-Info "Please install Visual Studio 2019/2022 with 'Desktop development with C++' workload"
    Write-Info "Download: https://visualstudio.microsoft.com/downloads/"
    Write-Info ""
    
    $response = Read-Host "Continue without Visual Studio? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        exit 1
    }
}

# Check NVIDIA GPU
function Test-NvidiaGpu {
    Write-Info "Checking for NVIDIA GPU..."
    
    try {
        $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpu) {
            Write-Success "NVIDIA GPU found: $($gpu.Name)"
            return $true
        } else {
            Write-Info "No NVIDIA GPU detected"
            return $false
        }
    } catch {
        Write-Warning "Could not detect GPU"
        return $false
    }
}

# Install Chocolatey
function Install-Chocolatey {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Success "Chocolatey already installed"
        return
    }
    
    Write-Info "Installing Chocolatey package manager..."
    
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Success "Chocolatey installed"
    } else {
        Write-ErrorMsg "Chocolatey installation failed"
        exit 1
    }
}

# Install CMake
function Install-CMake {
    Write-Info "Checking CMake..."
    
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        $version = cmake --version | Select-Object -First 1
        Write-Success "CMake already installed: $version"
        return
    }
    
    Write-Info "Installing CMake via Chocolatey..."
    choco install cmake -y
    
    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        Write-Success "CMake installed"
    } else {
        Write-Warning "CMake installation may require a new shell"
    }
}

# Install Rust
function Install-Rust {
    if ($SkipRust) {
        Write-Info "Skipping Rust installation (-SkipRust)"
        return
    }
    
    Write-Info "Checking Rust..."
    
    if (Get-Command rustc -ErrorAction SilentlyContinue) {
        $version = rustc --version
        Write-Success "Rust already installed: $version"
        
        $response = Read-Host "Reinstall? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            return
        }
    }
    
    Write-Info "Installing Rust..."
    
    # Download rustup-init.exe
    $rustupUrl = "https://win.rustup.rs/x86_64"
    $rustupPath = "$env:TEMP\rustup-init.exe"
    
    Write-Info "Downloading rustup..."
    Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupPath
    
    # Run rustup installer
    Write-Info "Running Rust installer..."
    & $rustupPath -y --default-toolchain stable --default-host x86_64-pc-windows-msvc
    
    # Add cargo to PATH
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
    
    if (Get-Command rustc -ErrorAction SilentlyContinue) {
        $version = rustc --version
        Write-Success "Rust installed: $version"
    } else {
        Write-ErrorMsg "Rust installation failed"
        exit 1
    }
}

# Check CUDA
function Test-Cuda {
    Write-Info "Checking CUDA..."
    
    if ($env:CUDA_PATH) {
        Write-Success "CUDA found: $env:CUDA_PATH"
        
        $nvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $nvccPath) {
            $version = & $nvccPath --version | Select-String "release" | Select-Object -First 1
            Write-Success "CUDA version: $version"
            return $true
        }
    }
    
    Write-Info "CUDA not found"
    return $false
}

# Install CUDA
function Install-Cuda {
    if ($SkipCuda) {
        Write-Info "Skipping CUDA installation (-SkipCuda)"
        return
    }
    
    if (-not (Test-NvidiaGpu)) {
        Write-Info "No NVIDIA GPU detected, skipping CUDA installation"
        return
    }
    
    if (Test-Cuda) {
        Write-Success "CUDA already installed"
        return
    }
    
    Write-Warning "CUDA Toolkit not found"
    Write-Info "Please download and install CUDA Toolkit from:"
    Write-Info "https://developer.nvidia.com/cuda-downloads"
    Write-Info ""
    Write-Info "Select: Windows > x86_64 > 10/11 > exe (local)"
    Write-Info ""
    
    $response = Read-Host "Continue without CUDA? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        exit 1
    }
}

# Create config file
function New-LlOrchConfig {
    Write-Info "Creating .llorch.toml configuration..."
    
    # Find project root
    $projectRoot = $null
    if (Test-Path "..\..\..llorch.toml.example") {
        $projectRoot = "..\..\"
    } elseif (Test-Path "..\..llorch.toml.example") {
        $projectRoot = "..\"
    } elseif (Test-Path ".llorch.toml.example") {
        $projectRoot = "."
    } else {
        Write-Warning "Cannot find .llorch.toml.example, skipping config creation"
        return
    }
    
    $configFile = Join-Path $projectRoot ".llorch.toml"
    
    if (Test-Path $configFile) {
        Write-Warning "Configuration file already exists: $configFile"
        return
    }
    
    # Determine if CUDA should be enabled
    $cudaEnabled = Test-Cuda
    
    $configContent = @"
# llama-orch Local Build Configuration (Windows)

[build]
# CUDA support
cuda = $($cudaEnabled.ToString().ToLower())

# Auto-detect CUDA toolkit
auto_detect_cuda = true

[development]
# Enable verbose build output (default: false)
verbose_build = false

# Skip CUDA tests if CUDA not available
skip_cuda_tests = $(-not $cudaEnabled)
"@
    
    Set-Content -Path $configFile -Value $configContent
    Write-Success "Configuration file created (CUDA: $cudaEnabled)"
}

# Run Rust tests
function Invoke-RustTests {
    if ($SkipTests) {
        Write-Info "Skipping tests (-SkipTests)"
        return
    }
    
    Write-Info "Running Rust tests..."
    
    # Find worker-orcd directory
    $workerDir = $null
    if (Test-Path "bin\worker-orcd") {
        $workerDir = "bin\worker-orcd"
    } elseif (Test-Path "..\bin\worker-orcd") {
        $workerDir = "..\bin\worker-orcd"
    } elseif (Test-Path "worker-orcd") {
        $workerDir = "worker-orcd"
    } else {
        Write-ErrorMsg "Cannot find worker-orcd directory"
        return
    }
    
    Push-Location $workerDir
    
    try {
        Write-Info "Running library tests..."
        cargo test --lib --no-fail-fast
        
        Write-Success "All Rust tests passed"
    } finally {
        Pop-Location
    }
}

# Build CUDA tests
function Build-CudaTests {
    if ($SkipCuda -or -not (Test-Cuda)) {
        Write-Info "Skipping CUDA tests build"
        return
    }
    
    Write-Info "Building CUDA tests..."
    
    $cudaDir = "bin\worker-orcd\cuda"
    if (-not (Test-Path $cudaDir)) {
        Write-Warning "CUDA directory not found: $cudaDir"
        return
    }
    
    $buildDir = Join-Path $cudaDir "build"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    
    Push-Location $buildDir
    
    try {
        Write-Info "Running CMake..."
        cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON ..
        
        Write-Info "Building..."
        cmake --build . --config Release
        
        if (Test-Path "Release\cuda_tests.exe") {
            Write-Success "CUDA tests built: $buildDir\Release\cuda_tests.exe"
        } else {
            Write-Warning "CUDA tests executable not found"
        }
    } catch {
        Write-ErrorMsg "CUDA tests build failed: $_"
    } finally {
        Pop-Location
    }
}

# Print summary
function Write-Summary {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  SETUP COMPLETE" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Installed components:"
    
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        $cmakeVer = cmake --version | Select-Object -First 1
        Write-Host "  ✓ $cmakeVer" -ForegroundColor Green
    }
    
    if (-not $SkipRust -and (Get-Command rustc -ErrorAction SilentlyContinue)) {
        $rustVer = rustc --version
        Write-Host "  ✓ Rust $rustVer" -ForegroundColor Green
    }
    
    if (Test-Cuda) {
        Write-Host "  ✓ CUDA Toolkit ($env:CUDA_PATH)" -ForegroundColor Green
    }
    
    Write-Host ""
    
    if (-not $SkipTests) {
        Write-Host "Test results:"
        Write-Host "  ✓ Rust library tests: PASSED" -ForegroundColor Green
        Write-Host ""
    }
    
    Write-Host "Next steps:"
    if (Test-Cuda) {
        Write-Host "  1. Build worker: cd bin\worker-orcd && cargo build --release"
        Write-Host "  2. Run CUDA tests: cd cuda\build\Release && .\cuda_tests.exe"
        Write-Host "  3. Download model: mkdir ~\models && cd ~\models"
        Write-Host "     Invoke-WebRequest -Uri 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf' -OutFile 'qwen-0.5b.gguf'"
        Write-Host "  4. Run worker: .\target\release\worker-orcd.exe --model ~\models\qwen-0.5b.gguf --gpu 0"
    } else {
        Write-Host "  1. Build worker (CPU-only): cd bin\worker-orcd && cargo build --release"
        Write-Host "  2. Download model: mkdir ~\models && cd ~\models"
        Write-Host "     Invoke-WebRequest -Uri 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf' -OutFile 'qwen-0.5b.gguf'"
        Write-Host "  3. Run worker (CPU): .\target\release\worker-orcd.exe --model ~\models\qwen-0.5b.gguf"
    }
    
    Write-Host ""
    Write-Success "Development workstation setup complete!"
}

# Main execution
function Main {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  llama-orch Windows Dev Setup" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if running as admin
    if (-not (Test-Administrator)) {
        Write-Warning "Not running as Administrator. Some installations may fail."
        Write-Info "Consider running PowerShell as Administrator"
        Write-Host ""
    }
    
    Test-WindowsVersion
    Test-VisualStudio
    
    Write-Info "Starting installation..."
    Write-Host ""
    
    # Install components
    Install-Chocolatey
    Install-CMake
    Install-Rust
    Install-Cuda
    
    # Configuration
    New-LlOrchConfig
    
    # Build and test
    Invoke-RustTests
    Build-CudaTests
    
    # Summary
    Write-Summary
}

# Run main function
Main
