# Worker Installation Architecture

**Date:** 2025-11-01  
**Status:** 🚧 IN PROGRESS

## Overview

The worker catalog system enables **downloading, building, and installing** worker binaries on the user's device. This is NOT for spawning workers - it's for getting the worker binaries onto the system first.

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User browses worker catalog in Hive UI                  │
│    - Sees available workers (CPU, CUDA, Metal)              │
│    - Checks platform compatibility                          │
│    - Views build requirements                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. User clicks "Install Worker"                             │
│    - Frontend sends install request to Hive backend         │
│    - Includes: worker_id, target_path                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Hive backend fetches PKGBUILD from catalog               │
│    GET http://localhost:8787/workers/{id}/PKGBUILD          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Hive parses PKGBUILD (pkgbuild_parser.rs)                │
│    - Extracts source URL                                    │
│    - Extracts build commands                                │
│    - Extracts dependencies                                  │
│    - Extracts install path                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Hive downloads source code                               │
│    - git clone from source.url                              │
│    - Or download tarball                                    │
│    - Extract to temp directory                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Hive runs build commands                                 │
│    - Execute build() function from PKGBUILD                 │
│    - Example: cargo build --release --features cuda         │
│    - Stream output to UI via SSE                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Hive installs binary                                     │
│    - Execute package() function from PKGBUILD               │
│    - Copy binary to install_path                            │
│    - Set permissions (chmod +x)                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. Worker now available for spawning                        │
│    - Binary at /usr/local/bin/llm-worker-rbee-cuda          │
│    - Can be spawned via WorkerSpawn operation               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Worker Catalog Service (Hono - Port 8787)

**Location:** `bin/80-hono-worker-catalog/`

**Purpose:** Serves worker metadata and PKGBUILD files

**Endpoints:**
- `GET /workers` - List all available workers
- `GET /workers/{id}/PKGBUILD` - Download PKGBUILD for specific worker

**Data:**
- Worker metadata (version, platforms, dependencies, capabilities)
- PKGBUILD files (build instructions in Arch Linux format)

### 2. PKGBUILD Parser (Rust)

**Location:** `bin/20_rbee_hive/src/pkgbuild_parser.rs`

**Purpose:** Parse PKGBUILD files to extract build instructions

**Parses:**
- `pkgname` - Package name
- `pkgver` - Version
- `pkgrel` - Release number
- `pkgdesc` - Description
- `arch` - Supported architectures
- `license` - License
- `depends` - Runtime dependencies
- `makedepends` - Build dependencies
- `source` - Source URLs
- `sha256sums` - Checksums
- `build()` - Build function body
- `package()` - Package function body

**Example:**
```rust
use rbee_hive::pkgbuild_parser::PkgBuild;

let pkgbuild = PkgBuild::from_file("llm-worker-rbee-cpu.PKGBUILD")?;

println!("Name: {}", pkgbuild.pkgname);
println!("Version: {}", pkgbuild.pkgver);
println!("Build deps: {:?}", pkgbuild.makedepends);

if let Some(build_fn) = pkgbuild.build_fn {
    // Execute build commands
    execute_build(&build_fn)?;
}
```

### 3. Worker Installer (Rust - TODO)

**Location:** `bin/20_rbee_hive/src/worker_installer.rs` (to be created)

**Purpose:** Download, build, and install workers

**Functions:**
```rust
pub struct WorkerInstaller {
    catalog_url: String,
    temp_dir: PathBuf,
}

impl WorkerInstaller {
    /// Fetch PKGBUILD from catalog
    pub async fn fetch_pkgbuild(&self, worker_id: &str) -> Result<PkgBuild>;
    
    /// Download source code
    pub async fn download_source(&self, pkgbuild: &PkgBuild) -> Result<PathBuf>;
    
    /// Build worker binary
    pub async fn build_worker(&self, pkgbuild: &PkgBuild, src_dir: &Path) -> Result<PathBuf>;
    
    /// Install worker binary
    pub async fn install_worker(&self, pkgbuild: &PkgBuild, binary: &Path) -> Result<()>;
    
    /// Full installation flow
    pub async fn install(&self, worker_id: &str) -> Result<()>;
}
```

### 4. Frontend Integration (React)

**Location:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/`

**New Component:** `InstallWorkerView.tsx` (to be created)

**Features:**
- Browse available workers from catalog
- Show installation requirements
- Start installation process
- Show build progress (SSE streaming)
- Show installation status

## PKGBUILD Format

PKGBUILD is the Arch Linux package build format. It's a bash script with specific variables and functions.

**Example:**
```bash
# Maintainer: rbee team
pkgname=llm-worker-rbee-cuda
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (NVIDIA CUDA)"
arch=('x86_64')
license=('GPL-3.0-or-later')
depends=('gcc' 'cuda')
makedepends=('rust' 'cargo')
source=("https://github.com/user/llama-orch.git")
sha256sums=('SKIP')

build() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    cargo build --release --features cuda --bin llm-worker-rbee-cuda
}

package() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    install -Dm755 target/release/llm-worker-rbee-cuda \
        "$pkgdir/usr/local/bin/llm-worker-rbee-cuda"
}
```

## Installation vs Spawning

### Installation (This System)
- **Purpose:** Get worker binary onto the system
- **When:** Once per worker type (CPU, CUDA, Metal)
- **Output:** Binary file at `/usr/local/bin/llm-worker-rbee-cuda`
- **Duration:** Minutes (download + compile)

### Spawning (Existing System)
- **Purpose:** Start a worker process
- **When:** Every time you need a worker
- **Input:** Binary path, model, device
- **Duration:** Seconds (just process startup)

**Analogy:**
- **Installation** = Installing Chrome browser
- **Spawning** = Opening a new Chrome window

## Next Steps

### 1. Create Worker Installer Module

**File:** `bin/20_rbee_hive/src/worker_installer.rs`

**Implement:**
- Fetch PKGBUILD from catalog
- Download source (git clone or tarball)
- Execute build commands
- Install binary
- Stream progress via SSE

### 2. Add Installation API Endpoint

**File:** `bin/20_rbee_hive/src/main.rs`

**Endpoint:** `POST /v1/workers/install`

**Request:**
```json
{
  "worker_id": "llm-worker-rbee-cuda",
  "install_path": "/usr/local/bin"
}
```

**Response:** SSE stream with build progress

### 3. Create Frontend Install UI

**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/InstallWorkerView.tsx`

**Features:**
- List available workers from catalog
- Show platform compatibility
- Show build requirements
- Install button
- Progress bar with build output
- Success/error states

### 4. Update Worker Catalog UI

**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/SpawnWorkerView.tsx`

**Changes:**
- Check if worker binary is installed
- Show "Install Worker" button if not installed
- Disable spawn if worker not installed
- Link to installation view

## Dependencies

### Rust Crates Needed

```toml
[dependencies]
# Already have these
serde = { version = "1.0", features = ["derive"] }
thiserror = "2.0"

# Need to add these
reqwest = { version = "0.12", features = ["json"] }  # HTTP client
tokio = { version = "1", features = ["process"] }    # Run build commands
tempfile = "3.0"                                     # Temp directories
```

### System Dependencies

**For building workers:**
- `rust` and `cargo` (always required)
- `gcc` or `clang` (for C dependencies)
- `cuda` toolkit (for CUDA workers)
- `git` (for cloning source)

## Security Considerations

1. **Verify checksums:** Check sha256sums from PKGBUILD
2. **Sandbox builds:** Run builds in isolated environment
3. **Validate source URLs:** Only allow trusted domains
4. **User permissions:** May need sudo for system-wide installs
5. **Code signing:** Future: verify binary signatures

## Testing

### Unit Tests

```rust
#[test]
fn test_parse_pkgbuild() {
    let content = include_str!("../test_data/llm-worker-cpu.PKGBUILD");
    let pkgbuild = PkgBuild::parse(content).unwrap();
    
    assert_eq!(pkgbuild.pkgname, "llm-worker-rbee-cpu");
    assert_eq!(pkgbuild.pkgver, "0.1.0");
}

#[test]
fn test_download_source() {
    // Mock git clone
}

#[test]
fn test_build_worker() {
    // Mock cargo build
}
```

### Integration Tests

1. Fetch PKGBUILD from catalog
2. Parse PKGBUILD
3. Download source (to temp dir)
4. Build worker (with test features)
5. Verify binary exists
6. Clean up temp files

## Future Enhancements

1. **Binary cache:** Cache built binaries to avoid rebuilding
2. **Pre-built binaries:** Serve pre-compiled binaries for common platforms
3. **Update mechanism:** Check for worker updates
4. **Uninstall:** Remove installed workers
5. **Multiple versions:** Support multiple worker versions side-by-side
6. **Build profiles:** Debug vs release builds
7. **Custom build flags:** Allow users to customize build options

---

**Status:** PKGBUILD parser implemented ✅  
**Next:** Implement worker installer module
