// TEAM-477: CivitAI model detail page for Tauri
// Uses @rbee/marketplace-core adapters (client-side fetching)
// Uses CivitAIModelDetail template (3-column design)

import { fetchCivitAIModel } from '@rbee/marketplace-core'
import { CivitAIModelDetail } from '@rbee/ui/marketplace'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'

export function CAIDetailsPage() {
  const { modelId } = useParams<{ modelId: string }>()

  // Fetch model data using marketplace-core adapter
  const { data: model, isLoading, error } = useQuery({
    queryKey: ['civitai-model', modelId],
    queryFn: async () => {
      if (!modelId) throw new Error('Model ID is required')
      // Parse numeric ID from slug (e.g., "civitai-12345" or "12345")
      const numericId = parseInt(modelId.replace(/^civitai-/, ''), 10)
      if (Number.isNaN(numericId)) {
        throw new Error(`Invalid CivitAI model ID: ${modelId}`)
      }
      return fetchCivitAIModel(numericId)
    },
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  // Loading state
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-7xl">
        <div className="text-center py-12 text-muted-foreground">Loading model details...</div>
      </div>
    )
  }

  // Error state
  if (error || !model) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-7xl">
        <div className="text-center py-12 text-destructive">
          Error loading model: {error ? String(error) : 'Model not found'}
        </div>
      </div>
    )
  }

  // Convert MarketplaceModel to CivitAIModelDetailProps
  const primaryVersionMetadata = model.metadata?.primaryVersion as
    | { id: number; name: string; baseModel?: string; downloadUrl?: string }
    | undefined

  const parsedModelId = Number.parseInt(model.id, 10)

  const primaryVersion = {
    id: primaryVersionMetadata?.id ?? (Number.isNaN(parsedModelId) ? 0 : parsedModelId),
    name:
      primaryVersionMetadata?.name ||
      (model.metadata?.version as string | undefined) ||
      'Latest',
    ...(primaryVersionMetadata?.baseModel || model.metadata?.baseModel
      ? { baseModel: (primaryVersionMetadata?.baseModel || model.metadata?.baseModel) as string }
      : {}),
    ...(model.description ? { description: model.description } : {}),
    createdAt: model.createdAt.toISOString(),
    updatedAt: model.updatedAt.toISOString(),
    ...(model.metadata?.publishedAt
      ? { publishedAt: model.metadata.publishedAt as string }
      : {}),
    ...(model.metadata?.trainedWords
      ? { trainedWords: model.metadata.trainedWords as string[] }
      : {}),
    images:
      model.imageUrl && !model.nsfw
        ? [
            {
              url: model.imageUrl,
              nsfw: false,
              width: 1024,
              height: 1024,
            },
          ]
        : [],
    files: [],
    ...(primaryVersionMetadata?.downloadUrl
      ? { downloadUrl: primaryVersionMetadata.downloadUrl }
      : {}),
    ...(model.metadata?.stats
      ? {
          stats: model.metadata.stats as {
            downloadCount: number
            ratingCount: number
            rating: number
            thumbsUpCount: number
          },
        }
      : {}),
  }

  const civitaiModelData = {
    id: model.id,
    name: model.name,
    description: model.description || '',
    author: model.author,
    downloads: model.downloads,
    likes: model.likes,
    rating: typeof model.metadata?.rating === 'number' ? model.metadata.rating : 0,
    size: model.sizeBytes ? formatBytes(model.sizeBytes) : 'Unknown',
    tags: model.tags,
    type: model.type,
    baseModel: (model.metadata?.baseModel as string) || 'Unknown',
    version: (model.metadata?.version as string) || 'Latest',
    images: model.imageUrl && !model.nsfw ? [{ url: model.imageUrl, nsfw: false, width: 1024, height: 1024 }] : [],
    files: [], // Files would come from metadata if available
    trainedWords: (model.metadata?.trainedWords as string[]) || undefined,
    allowCommercialUse: model.metadata?.allowCommercialUse || 'Unknown',
    externalUrl: model.url,
    externalLabel: 'View on CivitAI',
    versions: [primaryVersion],
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <CivitAIModelDetail model={civitaiModelData} />
    </div>
  )
}

// Helper function
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
}
