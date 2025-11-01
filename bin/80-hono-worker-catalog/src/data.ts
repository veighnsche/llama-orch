// Worker Catalog Data
// All available worker variants

import type { WorkerCatalogEntry } from "./types";

export const WORKERS: WorkerCatalogEntry[] = [
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
