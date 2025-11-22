// TEAM-482: Global Worker Catalog (GWC) types
// TEAM-483: SOURCE OF TRUTH - bin/80-global-worker-catalog imports from here
// TEAM-484: Added marketplace compatibility matrix, cover images, README URLs
// TEAM-485: Complete redesign - fixed contradictions, separated global vs per-variant
// TEAM_503: Added worker availability stages (alpha, beta, release, coming-soon)
// CANONICAL SOURCE: bin/97_contracts/artifacts-contract/src/worker.rs (via WASM)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ENUMS & PRIMITIVES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** Worker backend type (hardware acceleration) */
export type WorkerType = 'cpu' | 'cuda' | 'metal' | 'rocm'

/** Operating system */
export type Platform = 'linux' | 'macos' | 'windows'

/** CPU instruction set */
export type Architecture = 'x86_64' | 'aarch64'

/** Worker implementation language */
export type WorkerImplementation = 'rust' | 'python' | 'cpp'

/** Build system */
export type BuildSystem = 'cargo' | 'cmake' | 'pip' | 'npm'

/** Worker availability stage (lifecycle) */
export type WorkerAvailability = 'alpha' | 'beta' | 'release' | 'coming-soon'

/** Marketplace vendor (where models come from) */
export type MarketplaceVendor = 'huggingface' | 'civitai'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARKETPLACE COMPATIBILITY
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * HuggingFace marketplace filters
 * Defines which HF models this worker can run
 * 
 * TEAM-502: Expanded to support richer filter sidebar
 * - tasks: What the model does (text-generation, image-to-image, etc.)
 * - libraries: Framework/format (transformers, diffusers, etc.)
 * - formats: Model file formats (gguf, safetensors, pytorch, etc.)
 * - languages: Model training languages (optional - for multilingual models)
 * - licenses: Acceptable model licenses (optional - for compliance)
 * - minParameters: Minimum model size in billions (optional)
 * - maxParameters: Maximum model size in billions (optional)
 */
export interface HuggingFaceCompatibility {
  /** Supported HF tasks (e.g., 'text-generation', 'text-to-image') */
  tasks: string[]
  
  /** Supported HF libraries (e.g., 'transformers', 'diffusers') */
  libraries: string[]
  
  /** Supported model formats (e.g., 'gguf', 'safetensors', 'pytorch') */
  formats: string[]
  
  /** Supported languages (optional - for multilingual models) */
  languages?: string[]
  
  /** Acceptable licenses (optional - for compliance filtering) */
  licenses?: string[]
  
  /** Minimum model size in billions of parameters (optional) */
  minParameters?: number
  
  /** Maximum model size in billions of parameters (optional) */
  maxParameters?: number
}

/**
 * CivitAI marketplace filters
 * Defines which CivitAI models this worker can run
 */
export interface CivitAICompatibility {
  /** Supported CivitAI model types (e.g., 'Checkpoint', 'LORA') */
  modelTypes: string[]
  /** Supported CivitAI base models (e.g., 'SD 1.5', 'SDXL 1.0') */
  baseModels: string[]
}

/**
 * Marketplace compatibility for a worker
 * Defines which marketplace vendors and model types this worker supports
 * 
 * GLOBAL: Same across all variants (CPU/CUDA/Metal all support same model types)
 */
export interface MarketplaceCompatibility {
  /** HuggingFace compatibility (undefined = not supported) */
  huggingface?: HuggingFaceCompatibility
  
  /** CivitAI compatibility (undefined = not supported) */
  civitai?: CivitAICompatibility
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BUILD VARIANT (PER-BACKEND)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Build configuration for a specific variant
 */
export interface BuildConfig {
  /** Cargo features (for Rust) */
  features?: string[]
  /** Build profile (release, debug) */
  profile?: string
  /** Additional build flags */
  flags?: string[]
}

/**
 * Build variant for a worker
 * Each variant represents a different backend (CPU, CUDA, Metal, ROCm)
 * 
 * PER-VARIANT: Different for each backend
 */
export interface BuildVariant {
  // ━━━ Hardware Target ━━━
  /** Backend type (cpu, cuda, metal, rocm) */
  backend: WorkerType
  /** Supported platforms for this variant */
  platforms: Platform[]
  /** Supported architectures for this variant */
  architectures: Architecture[]
  
  // ━━━ Package Management ━━━
  /** URL to PKGBUILD file (binary version) */
  pkgbuildUrl: string
  /** URL to PKGBUILD file (git version) */
  pkgbuildUrlGit: string
  /** URL to Homebrew formula (binary version) */
  homebrewFormula: string
  /** URL to Homebrew formula (git version) */
  homebrewFormulaGit: string
  
  // ━━━ Build Configuration ━━━
  /** Build configuration */
  build: BuildConfig
  
  // ━━━ Dependencies ━━━
  /** Runtime dependencies specific to this variant */
  depends: string[]
  /** Build dependencies specific to this variant */
  makedepends: string[]
  
  // ━━━ Installation ━━━
  /** Binary name for this variant (e.g., "llm-worker-rbee") */
  binaryName: string
  /** Installation path for this variant */
  installPath: string
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WORKER CATALOG ENTRY
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Source repository configuration
 */
export interface SourceConfig {
  /** Source type */
  type: 'git' | 'tarball'
  /** Repository or tarball URL */
  url: string
  /** Git branch (for git sources) */
  branch?: string
  /** Git tag (for git sources) */
  tag?: string
  /** Path within repo (e.g., "bin/30_llm_worker_rbee") */
  path?: string
}

/**
 * Worker capabilities (GLOBAL - same across all variants)
 */
export interface WorkerCapabilities {
  /** Supported model formats (e.g., 'gguf', 'safetensors') */
  supportedFormats: string[]
  /** Maximum context length (for LLM workers) */
  maxContextLength?: number
  /** Supports streaming responses */
  supportsStreaming: boolean
  /** Supports batching requests */
  supportsBatching: boolean
}

/**
 * Worker catalog entry (from GWC API)
 * 
 * STRUCTURE:
 * - GLOBAL properties: Same across all variants (CPU/CUDA/Metal)
 * - PER-VARIANT properties: Different for each backend (in BuildVariant)
 */
export interface GWCWorker {
  // ━━━ Identity (GLOBAL) ━━━
  /** Unique worker ID (e.g., "llm-worker-rbee") */
  id: string
  /** Worker implementation language */
  implementation: WorkerImplementation
  /** Version (semver) */
  version: string
  
  // ━━━ Metadata (GLOBAL) ━━━
  /** Human-readable name (e.g., "LLM Worker") */
  name: string
  /** Short description */
  description: string
  /** License (SPDX identifier) */
  license: string
  /** Availability stage (alpha, beta, release, coming-soon) */
  availability?: WorkerAvailability
  /** Cover image URL (preferably 1:1 ratio) */
  coverImage?: string
  /** README URL (raw markdown) */
  readmeUrl?: string
  
  // ━━━ Build System (GLOBAL) ━━━
  /** Build system */
  buildSystem: BuildSystem
  /** Source repository (shared across all variants) */
  source: SourceConfig
  /** Available build variants (CPU, CUDA, Metal, ROCm) */
  variants: BuildVariant[]
  
  // ━━━ Capabilities (GLOBAL) ━━━
  /** Worker capabilities (same across all variants) */
  capabilities: WorkerCapabilities
  
  // ━━━ Marketplace Compatibility (GLOBAL) ━━━
  /** Which marketplace vendors and model types this worker supports */
  marketplaceCompatibility?: MarketplaceCompatibility
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * GWC API response for list endpoint
 */
export interface GWCListWorkersResponse {
  workers: GWCWorker[]
}

/**
 * GWC list workers parameters (for future filtering)
 */
export interface GWCListWorkersParams {
  /** Filter by backend type */
  backend?: WorkerType
  /** Filter by platform */
  platform?: Platform
  /** Limit number of results */
  limit?: number
}
