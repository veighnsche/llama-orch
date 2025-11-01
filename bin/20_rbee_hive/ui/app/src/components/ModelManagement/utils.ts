// TEAM-381: Utility functions for model detection and filtering

import type { HFModel } from './types'

/**
 * Detect architecture from model ID and tags
 * Examples:
 * - "meta-llama/Llama-2-7b-chat-hf" → ["llama"]
 * - "mistralai/Mistral-7B-v0.1" → ["mistral"]
 */
export function detectArchitecture(modelId: string, tags: string[]): string[] {
  const architectures: string[] = []
  const lowerModelId = modelId.toLowerCase()
  
  // Check model ID
  if (/llama|alpaca|vicuna/i.test(lowerModelId)) architectures.push('llama')
  if (/mistral|mixtral/i.test(lowerModelId)) architectures.push('mistral')
  if (/phi/i.test(lowerModelId)) architectures.push('phi')
  if (/gemma/i.test(lowerModelId)) architectures.push('gemma')
  if (/qwen/i.test(lowerModelId)) architectures.push('qwen')
  
  // Check tags
  tags.forEach((tag) => {
    const lowerTag = tag.toLowerCase()
    if (lowerTag.includes('llama')) architectures.push('llama')
    if (lowerTag.includes('mistral')) architectures.push('mistral')
    if (lowerTag.includes('phi')) architectures.push('phi')
    if (lowerTag.includes('gemma')) architectures.push('gemma')
    if (lowerTag.includes('qwen')) architectures.push('qwen')
  })
  
  return [...new Set(architectures)]
}

/**
 * Detect format from model ID and tags
 * Examples:
 * - tags: ["gguf"] → ["gguf"]
 * - modelId: "model-Q4_K_M.gguf" → ["gguf"]
 */
export function detectFormat(modelId: string, tags: string[]): string[] {
  const formats: string[] = []
  
  // GGUF indicators
  if (tags.includes('gguf') || /\.gguf/i.test(modelId)) {
    formats.push('gguf')
  }
  
  // SafeTensors indicators
  if (tags.includes('safetensors') || tags.includes('pytorch')) {
    formats.push('safetensors')
  }
  
  return formats
}

/**
 * Filter HuggingFace models by architecture, format, and license
 */
export function filterModels(
  models: HFModel[],
  filters: {
    formats: string[]
    architectures: string[]
    openSourceOnly: boolean
  }
): HFModel[] {
  return models.filter((model) => {
    // Architecture filter
    const modelArchs = detectArchitecture(model.modelId, model.tags)
    const hasMatchingArch =
      filters.architectures.length === 0 ||
      modelArchs.some((arch) => filters.architectures.includes(arch))
    
    if (!hasMatchingArch) return false
    
    // Format filter
    const modelFormats = detectFormat(model.modelId, model.tags)
    const hasMatchingFormat =
      filters.formats.length === 0 ||
      modelFormats.some((fmt) => filters.formats.includes(fmt))
    
    if (!hasMatchingFormat) return false
    
    // License filter (basic check)
    if (filters.openSourceOnly && model.gated) return false
    
    return true
  })
}

/**
 * Sort models by specified criteria
 */
export function sortModels(
  models: HFModel[],
  sortBy: 'downloads' | 'likes' | 'recent'
): HFModel[] {
  const sorted = [...models]
  
  if (sortBy === 'downloads') {
    sorted.sort((a, b) => b.downloads - a.downloads)
  } else if (sortBy === 'likes') {
    sorted.sort((a, b) => b.likes - a.likes)
  }
  // 'recent' would need lastModified field
  
  return sorted
}
