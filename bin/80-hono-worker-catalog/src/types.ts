// Worker Catalog Types
// Matches Rust types in:
// - bin/25_rbee_hive_crates/worker-catalog/src/types.rs (WorkerType, Platform)
// - bin/97_contracts/worker-contract/src/types.rs (WorkerInfo)

/**
 * Worker type (backend acceleration)
 * Matches Rust enum: WorkerType { CpuLlm, CudaLlm, MetalLlm }
 */
export type WorkerType = "cpu" | "cuda" | "metal";

/**
 * Platform (operating system)
 * Matches Rust enum: Platform { Linux, MacOS, Windows }
 */
export type Platform = "linux" | "macos" | "windows";

/**
 * Architecture (CPU instruction set)
 */
export type Architecture = "x86_64" | "aarch64";

/**
 * Worker implementation type
 * - "llm-worker-rbee": Bespoke Candle-based worker (our implementation)
 * - "llama-cpp-adapter": llama.cpp wrapper (future)
 * - "vllm-adapter": vLLM wrapper (future)
 * - "ollama-adapter": Ollama wrapper (future)
 * - "comfyui-adapter": ComfyUI wrapper (future)
 */
export type WorkerImplementation = 
  | "llm-worker-rbee"
  | "llama-cpp-adapter"
  | "vllm-adapter"
  | "ollama-adapter"
  | "comfyui-adapter";

/**
 * Build system type
 */
export type BuildSystem = "cargo" | "cmake" | "pip" | "npm";

/**
 * Worker catalog entry
 * Provides all information needed to download, build, and install a worker
 */
export interface WorkerCatalogEntry {
  // ━━━ Identity ━━━
  /** Unique worker ID (e.g., "llm-worker-rbee-cpu") */
  id: string;
  
  /** Worker implementation type */
  implementation: WorkerImplementation;
  
  /** Worker type (backend) */
  worker_type: WorkerType;
  
  /** Version (semver) */
  version: string;
  
  // ━━━ Platform Support ━━━
  /** Supported platforms */
  platforms: Platform[];
  
  /** Supported architectures */
  architectures: Architecture[];
  
  // ━━━ Metadata ━━━
  /** Human-readable name */
  name: string;
  
  /** Short description */
  description: string;
  
  /** License (SPDX identifier) */
  license: string;
  
  // ━━━ Build Instructions ━━━
  /** URL to PKGBUILD file */
  pkgbuild_url: string;
  
  /** Build system */
  build_system: BuildSystem;
  
  /** Source repository */
  source: {
    type: "git" | "tarball";
    url: string;
    branch?: string;
    tag?: string;
    path?: string;  // Path within repo (e.g., "bin/30_llm_worker_rbee")
  };
  
  /** Build configuration */
  build: {
    /** Cargo features (for Rust) */
    features?: string[];
    /** Build profile (release, debug) */
    profile?: string;
    /** Additional build flags */
    flags?: string[];
  };
  
  // ━━━ Dependencies ━━━
  /** Runtime dependencies */
  depends: string[];
  
  /** Build dependencies */
  makedepends: string[];
  
  // ━━━ Binary Info ━━━
  /** Binary name (output) */
  binary_name: string;
  
  /** Installation path */
  install_path: string;
  
  // ━━━ Capabilities ━━━
  /** Supported model formats */
  supported_formats: string[];
  
  /** Maximum context length */
  max_context_length?: number;
  
  /** Supports streaming */
  supports_streaming: boolean;
  
  /** Supports batching */
  supports_batching: boolean;
}
