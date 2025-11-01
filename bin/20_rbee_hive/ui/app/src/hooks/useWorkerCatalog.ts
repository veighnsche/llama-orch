// Hook to fetch worker catalog from Hono catalog service
// Provides build instructions and metadata for available worker types

import { useQuery } from '@tanstack/react-query'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TYPES (matches Hono catalog service)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export type WorkerType = 'cpu' | 'cuda' | 'metal'
export type Platform = 'linux' | 'macos' | 'windows'
export type Architecture = 'x86_64' | 'aarch64'
export type WorkerImplementation = 
  | 'llm-worker-rbee'
  | 'llama-cpp-adapter'
  | 'vllm-adapter'
  | 'ollama-adapter'
  | 'comfyui-adapter'

export interface WorkerCatalogEntry {
  // Identity
  id: string
  implementation: WorkerImplementation
  worker_type: WorkerType
  version: string
  
  // Platform Support
  platforms: Platform[]
  architectures: Architecture[]
  
  // Metadata
  name: string
  description: string
  license: string
  
  // Build Instructions
  pkgbuild_url: string
  build_system: 'cargo' | 'cmake' | 'pip' | 'npm'
  source: {
    type: 'git' | 'tarball'
    url: string
    branch?: string
    tag?: string
    path?: string
  }
  build: {
    features?: string[]
    profile?: string
    flags?: string[]
  }
  
  // Dependencies
  depends: string[]
  makedepends: string[]
  
  // Binary Info
  binary_name: string
  install_path: string
  
  // Capabilities
  supported_formats: string[]
  max_context_length?: number
  supports_streaming: boolean
  supports_batching: boolean
}

export interface WorkerCatalogResponse {
  workers: WorkerCatalogEntry[]
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API CLIENT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const CATALOG_URL = 'http://localhost:8787'

async function fetchWorkerCatalog(): Promise<WorkerCatalogResponse> {
  const response = await fetch(`${CATALOG_URL}/workers`)
  
  if (!response.ok) {
    throw new Error(`Failed to fetch worker catalog: ${response.statusText}`)
  }
  
  return response.json()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HOOK
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export function useWorkerCatalog() {
  return useQuery({
    queryKey: ['worker-catalog'],
    queryFn: fetchWorkerCatalog,
    staleTime: 5 * 60 * 1000, // 5 minutes (catalog doesn't change often)
    retry: 3,
  })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPER FUNCTIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Get worker catalog entry by worker type
 */
export function getWorkerByType(
  catalog: WorkerCatalogResponse | undefined,
  workerType: WorkerType
): WorkerCatalogEntry | undefined {
  return catalog?.workers.find((w) => w.worker_type === workerType)
}

/**
 * Get current platform
 */
export function getCurrentPlatform(): Platform {
  const userAgent = navigator.userAgent.toLowerCase()
  
  if (userAgent.includes('mac')) return 'macos'
  if (userAgent.includes('win')) return 'windows'
  return 'linux'
}

/**
 * Check if worker is supported on current platform
 */
export function isWorkerSupported(
  worker: WorkerCatalogEntry,
  platform: Platform = getCurrentPlatform()
): boolean {
  return worker.platforms.includes(platform)
}

/**
 * Filter workers by platform
 */
export function getAvailableWorkers(
  catalog: WorkerCatalogResponse | undefined,
  platform: Platform = getCurrentPlatform()
): WorkerCatalogEntry[] {
  if (!catalog) return []
  return catalog.workers.filter((w) => isWorkerSupported(w, platform))
}
