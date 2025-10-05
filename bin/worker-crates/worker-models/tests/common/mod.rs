// Common test utilities
//
// Provides CUDA configuration detection and loud announcements for test runs

use std::path::Path;
use std::sync::Once;

static INIT: Once = Once::new();
static mut CUDA_ENABLED: bool = false;

/// Test configuration from .llorch.toml
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub cuda_enabled: bool,
    pub skip_cuda_tests: bool,
}

impl TestConfig {
    /// Load test configuration from .llorch.toml
    pub fn load() -> Self {
        let config_path = Path::new("../../.llorch.toml");
        
        if !config_path.exists() {
            eprintln!("\n⚠️  WARNING: .llorch.toml not found, using defaults (CUDA disabled for safety)");
            return Self {
                cuda_enabled: false,
                skip_cuda_tests: false,
            };
        }

        let content = match std::fs::read_to_string(config_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("\n⚠️  WARNING: Failed to read .llorch.toml: {}", e);
                return Self {
                    cuda_enabled: false,
                    skip_cuda_tests: false,
                };
            }
        };

        let value: toml::Value = match content.parse() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("\n⚠️  WARNING: Failed to parse .llorch.toml: {}", e);
                return Self {
                    cuda_enabled: false,
                    skip_cuda_tests: false,
                };
            }
        };

        let cuda_enabled = value
            .get("build")
            .and_then(|b| b.get("cuda"))
            .and_then(|c| c.as_bool())
            .unwrap_or(false);

        let skip_cuda_tests = value
            .get("development")
            .and_then(|d| d.get("skip_cuda_tests"))
            .and_then(|s| s.as_bool())
            .unwrap_or(false);

        Self {
            cuda_enabled,
            skip_cuda_tests,
        }
    }

    /// Check if CUDA tests should run
    pub fn should_run_cuda_tests(&self) -> bool {
        self.cuda_enabled && !self.skip_cuda_tests
    }
}

/// Initialize test environment and announce CUDA status LOUDLY
pub fn init_test_env() {
    INIT.call_once(|| {
        let config = TestConfig::load();
        
        unsafe {
            CUDA_ENABLED = config.cuda_enabled;
        }

        // LOUD announcement
        eprintln!("\n╔═══════════════════════════════════════════════════════════════╗");
        if config.cuda_enabled {
            eprintln!("║  🚀 CUDA ENABLED - Running tests with GPU support            ║");
            eprintln!("║                                                               ║");
            eprintln!("║  Configuration: .llorch.toml                                  ║");
            eprintln!("║  • build.cuda = true                                          ║");
            if config.skip_cuda_tests {
                eprintln!("║  • development.skip_cuda_tests = true                        ║");
                eprintln!("║                                                               ║");
                eprintln!("║  ⚠️  CUDA tests will be SKIPPED (skip_cuda_tests=true)       ║");
            } else {
                eprintln!("║  • development.skip_cuda_tests = false                       ║");
                eprintln!("║                                                               ║");
                eprintln!("║  ✅ CUDA tests will RUN                                       ║");
            }
        } else {
            eprintln!("║  ⛔ CUDA DISABLED - Running tests in STUB mode               ║");
            eprintln!("║                                                               ║");
            eprintln!("║  Configuration: .llorch.toml                                  ║");
            eprintln!("║  • build.cuda = false                                         ║");
            eprintln!("║                                                               ║");
            eprintln!("║  ⚠️  All CUDA-dependent tests will use STUB implementations   ║");
            eprintln!("║  ⚠️  No actual GPU operations will be performed               ║");
            eprintln!("║                                                               ║");
            eprintln!("║  To enable CUDA:                                              ║");
            eprintln!("║  1. Edit .llorch.toml                                         ║");
            eprintln!("║  2. Set build.cuda = true                                     ║");
            eprintln!("║  3. Rebuild: cargo clean && cargo test                        ║");
        }
        eprintln!("╚═══════════════════════════════════════════════════════════════╝\n");
    });
}

/// Check if CUDA is enabled (call after init_test_env)
pub fn is_cuda_enabled() -> bool {
    unsafe { CUDA_ENABLED }
}

/// Skip test if CUDA is required but not available
#[macro_export]
macro_rules! require_cuda {
    () => {
        $crate::common::init_test_env();
        if !$crate::common::is_cuda_enabled() {
            eprintln!("⏭️  SKIPPING: Test requires CUDA (build.cuda = false in .llorch.toml)");
            return;
        }
    };
}

/// Announce that a test is running in STUB mode
#[macro_export]
macro_rules! announce_stub_mode {
    ($test_name:expr) => {
        if !$crate::common::is_cuda_enabled() {
            eprintln!("🔧 [STUB MODE] {}: Using stub CUDA implementation", $test_name);
        }
    };
}

/// Conditionally run code only when CUDA is enabled
#[macro_export]
macro_rules! cuda_only {
    ($($code:tt)*) => {
        #[cfg(feature = "cuda")]
        {
            $($code)*
        }
    };
}

/// Conditionally run code only when CUDA is disabled (stub mode)
#[macro_export]
macro_rules! stub_only {
    ($($code:tt)*) => {
        #[cfg(not(feature = "cuda"))]
        {
            $($code)*
        }
    };
}
