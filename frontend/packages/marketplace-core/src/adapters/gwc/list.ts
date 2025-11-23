// TEAM-482: Global Worker Catalog (GWC) list API - Fetch all workers

import type { MarketplaceModel } from '../common'
import type { GWCListWorkersParams, GWCListWorkersResponse, GWCWorker } from './types'

/**
 * GWC API base URL
 * - Next.js Dev: http://localhost:7811 (global-worker-catalog dev port)
 * - Tauri: https://gwc.rbee.dev (Tauri can't access localhost)
 * - Prod: https://gwc.rbee.dev
 * - Override: NEXT_PUBLIC_GWC_API_URL (Next.js) or VITE_GWC_API_URL (Tauri)
 *
 * TEAM_529: marketplace-core serves both Next.js backend and Tauri frontend
 * - Next.js: Uses process.env.NEXT_PUBLIC_GWC_API_URL
 * - Tauri: Uses process.env.VITE_GWC_API_URL
 * - Check for window object to detect browser/Tauri environment
 */
const GWC_API_BASE =
  process.env.NEXT_PUBLIC_GWC_API_URL ||
  process.env.VITE_GWC_API_URL ||
  (process.env.NODE_ENV === 'development' && typeof window !== 'undefined' && !(window as any).__TAURI__
    ? 'http://localhost:7811'
    : 'https://gwc.rbee.dev')

/**
 * Convert GWC worker to MarketplaceModel
 */
export function convertGWCWorker(worker: GWCWorker): MarketplaceModel {
  // Get primary backend (first variant's backend)
  const primaryBackend = worker.variants?.[0]?.backend || 'cpu'

  // Get all supported backends
  const backends = worker.variants?.map((v) => v.backend).join(', ') || 'cpu'

  return {
    id: worker.id,
    name: worker.name,
    author: 'rbee', // All GWC workers are official rbee workers
    type: `${primaryBackend} worker`, // e.g., "cuda worker"
    description: worker.description,
    // TEAM-501: Map coverImage to imageUrl for display
    ...(worker.coverImage && { imageUrl: worker.coverImage }),
    tags: [
      worker.implementation, // rust, python, cpp
      ...(worker.variants?.map((v) => v.backend) || []), // cpu, cuda, metal, rocm
      ...(worker.capabilities.supportedFormats || []), // gguf, safetensors, etc.
    ],
    downloads: 0, // GWC doesn't track downloads yet
    likes: 0, // GWC doesn't track likes yet
    nsfw: false, // Workers are never NSFW
    createdAt: new Date(), // Not tracked by GWC
    updatedAt: new Date(), // Not tracked by GWC
    url: `${GWC_API_BASE}/workers/${worker.id}`, // Link to GWC API
    license: worker.license,
    // Additional metadata
    metadata: {
      version: worker.version,
      backends,
      implementation: worker.implementation,
      supportedFormats: worker.capabilities.supportedFormats.join(', '),
      supportsStreaming: worker.capabilities.supportsStreaming,
      supportsBatching: worker.capabilities.supportsBatching,
    },
  }
}

/**
 * Fetch raw GWC workers from the GWC API.
 *
 * @param params - Filter parameters (optional)
 * @returns Raw GWCWorker entries from the catalog
 */
export async function fetchGWCWorkers(params?: GWCListWorkersParams): Promise<GWCWorker[]> {
  const url = `${GWC_API_BASE}/workers`

  console.log('[GWC API] Fetching:', url)

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`GWC API error: ${response.status} ${response.statusText}`)
  }

  const data: GWCListWorkersResponse = await response.json()

  console.log(`[GWC API] Fetched ${data.workers.length} workers`)

  let workers = data.workers

  // Apply simple in-memory filtering for convenience; the HTTP API
  // does not yet support backend/platform filters directly.
  if (params?.backend) {
    const backend = params.backend
    workers = workers.filter((worker) => worker.variants.some((variant) => variant.backend === backend))
  }

  // NOTE: We intentionally do NOT filter by platform here yet; that
  // would require inspecting variants in more detail.

  if (params?.limit) {
    workers = workers.slice(0, params.limit)
  }

  return workers
}
