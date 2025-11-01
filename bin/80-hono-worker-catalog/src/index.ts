import { Hono } from "hono";

const app = new Hono<{ Bindings: CloudflareBindings }>();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WORKER CATALOG TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Matches Rust types in:
// - bin/25_rbee_hive_crates/worker-catalog/src/types.rs (WorkerType, Platform)
// - bin/97_contracts/worker-contract/src/types.rs (WorkerInfo)

/**
 * Worker type (backend acceleration)
 * Matches Rust enum: WorkerType { CpuLlm, CudaLlm, MetalLlm }
 */
type WorkerType = "cpu" | "cuda" | "metal";

/**
 * Platform (operating system)
 * Matches Rust enum: Platform { Linux, MacOS, Windows }
 */
type Platform = "linux" | "macos" | "windows";

/**
 * Architecture (CPU instruction set)
 */
type Architecture = "x86_64" | "aarch64";

/**
 * Worker implementation type
 * - "llm-worker-rbee": Bespoke Candle-based worker (our implementation)
 * - "llama-cpp-adapter": llama.cpp wrapper (future)
 * - "vllm-adapter": vLLM wrapper (future)
 * - "ollama-adapter": Ollama wrapper (future)
 * - "comfyui-adapter": ComfyUI wrapper (future)
 */
type WorkerImplementation = 
  | "llm-worker-rbee"
  | "llama-cpp-adapter"
  | "vllm-adapter"
  | "ollama-adapter"
  | "comfyui-adapter";

/**
 * Build system type
 */
type BuildSystem = "cargo" | "cmake" | "pip" | "npm";

/**
 * Worker catalog entry
 * Provides all information needed to download, build, and install a worker
 */
interface WorkerCatalogEntry {
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WORKER CATALOG DATA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const WORKERS: WorkerCatalogEntry[] = [
  {
    id: "llm-worker-rbee-cpu",
    implementation: "llm-worker-rbee",
    worker_type: "cpu",
    version: "0.1.0",
    platforms: ["linux", "macos", "windows"],
    architectures: ["x86_64", "aarch64"],
    name: "LLM Worker (CPU)",
    description: "Candle-based LLM inference worker with CPU acceleration",
    license: "GPL-3.0-or-later",
    pkgbuild_url: "/workers/llm-worker-rbee-cpu/PKGBUILD",
    build_system: "cargo",
    source: {
      type: "git",
      url: "https://github.com/user/llama-orch.git",
      branch: "main",
      path: "bin/30_llm_worker_rbee"
    },
    build: {
      features: ["cpu"],
      profile: "release"
    },
    depends: ["gcc"],
    makedepends: ["rust", "cargo"],
    binary_name: "llm-worker-rbee-cpu",
    install_path: "/usr/local/bin/llm-worker-rbee-cpu",
    supported_formats: ["gguf", "safetensors"],
    max_context_length: 32768,
    supports_streaming: true,
    supports_batching: false
  },
  {
    id: "llm-worker-rbee-cuda",
    implementation: "llm-worker-rbee",
    worker_type: "cuda",
    version: "0.1.0",
    platforms: ["linux", "windows"],
    architectures: ["x86_64"],
    name: "LLM Worker (CUDA)",
    description: "Candle-based LLM inference worker with NVIDIA CUDA acceleration",
    license: "GPL-3.0-or-later",
    pkgbuild_url: "/workers/llm-worker-rbee-cuda/PKGBUILD",
    build_system: "cargo",
    source: {
      type: "git",
      url: "https://github.com/user/llama-orch.git",
      branch: "main",
      path: "bin/30_llm_worker_rbee"
    },
    build: {
      features: ["cuda"],
      profile: "release"
    },
    depends: ["gcc", "cuda"],
    makedepends: ["rust", "cargo"],
    binary_name: "llm-worker-rbee-cuda",
    install_path: "/usr/local/bin/llm-worker-rbee-cuda",
    supported_formats: ["gguf", "safetensors"],
    max_context_length: 32768,
    supports_streaming: true,
    supports_batching: false
  },
  {
    id: "llm-worker-rbee-metal",
    implementation: "llm-worker-rbee",
    worker_type: "metal",
    version: "0.1.0",
    platforms: ["macos"],
    architectures: ["aarch64"],
    name: "LLM Worker (Metal)",
    description: "Candle-based LLM inference worker with Apple Metal acceleration",
    license: "GPL-3.0-or-later",
    pkgbuild_url: "/workers/llm-worker-rbee-metal/PKGBUILD",
    build_system: "cargo",
    source: {
      type: "git",
      url: "https://github.com/user/llama-orch.git",
      branch: "main",
      path: "bin/30_llm_worker_rbee"
    },
    build: {
      features: ["metal"],
      profile: "release"
    },
    depends: ["clang"],
    makedepends: ["rust", "cargo"],
    binary_name: "llm-worker-rbee-metal",
    install_path: "/usr/local/bin/llm-worker-rbee-metal",
    supported_formats: ["gguf", "safetensors"],
    max_context_length: 32768,
    supports_streaming: true,
    supports_batching: false
  }
];

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API ENDPOINTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// List all available worker variants
app.get("/workers", (c) => {
  return c.json({ workers: WORKERS });
});

// Serve PKGBUILD files
app.get("/workers/llm-worker-rbee-cpu/PKGBUILD", async (c) => {
  const pkgbuild = await c.env.ASSETS.fetch(new Request("http://placeholder/pkgbuilds/llm-worker-rbee-cpu.PKGBUILD"));
  return new Response(pkgbuild.body, {
    headers: { "Content-Type": "text/plain" }
  });
});

app.get("/workers/llm-worker-rbee-cuda/PKGBUILD", async (c) => {
  const pkgbuild = await c.env.ASSETS.fetch(new Request("http://placeholder/pkgbuilds/llm-worker-rbee-cuda.PKGBUILD"));
  return new Response(pkgbuild.body, {
    headers: { "Content-Type": "text/plain" }
  });
});

app.get("/workers/llm-worker-rbee-metal/PKGBUILD", async (c) => {
  const pkgbuild = await c.env.ASSETS.fetch(new Request("http://placeholder/pkgbuilds/llm-worker-rbee-metal.PKGBUILD"));
  return new Response(pkgbuild.body, {
    headers: { "Content-Type": "text/plain" }
  });
});

export default app;
