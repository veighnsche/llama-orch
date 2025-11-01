# TEAM-381: Model Filtering Strategy for Candle Workers

**Date:** 2025-11-01  
**Status:** 📋 PLANNING

## Candle Framework Capabilities

### Supported Model Formats
- ✅ **SafeTensors** - Primary format (native HuggingFace)
- ✅ **GGUF** - Quantized models (llama.cpp format)
- ✅ **GGML** - Legacy quantized format
- ✅ **PyTorch** - .pt/.pth files
- ✅ **NPZ** - NumPy format

### Supported Model Types (Current Candle)

**Text Generation (LLMs):**
- LLaMA v1, v2, v3 (including SOLAR-10.7B)
- Falcon
- StarCoder, StarCoder2
- Phi 1, 1.5, 2, 3
- Mamba, Minimal Mamba
- Gemma v1 (2b, 7b+), v2 (2b, 9b)
- Mistral 7b v0.1
- Mixtral 8x7b v0.1
- StableLM-3B-4E1T, StableLM-2-1.6B, Stable-Code-3B
- Replit-code-v1.5-3B
- BERT
- Yi-6B, Yi-34B
- Qwen1.5, Qwen1.5 MoE
- RWKV v5, v6

**Text-to-Text:**
- T5 and variants (FlanT5, UL2, MADLAD400, CoEdit)
- Marian MT (Machine Translation)

**Text-to-Image:**
- Stable Diffusion v1.5, v2.1, XL v1.0
- Wurstchen v2

**Image-to-Text:**
- BLIP
- TrOCR

**Audio:**
- Whisper (speech-to-text)
- EnCodec (audio compression)
- MetaVoice-1B (text-to-speech)
- Parler-TTS (text-to-speech)

**Computer Vision:**
- DINOv2, ConvMixer, EfficientNet, ResNet, ViT, VGG, RepVGG
- ConvNeXT, ConvNeXTv2, MobileOne, EfficientVit, MobileNetv4, Hiera, FastViT
- YOLO-v3, YOLO-v8
- Segment-Anything Model (SAM)
- SegFormer

## MVP Worker Types

### Phase 1: LLM Workers (Current)
**Worker:** `llm-worker` (Candle-based)

**Supported Formats:**
- ✅ GGUF (quantized, recommended for MVP)
- ✅ SafeTensors (full precision)

**Supported Architectures:**
- LLaMA family (most popular)
- Mistral/Mixtral
- Phi
- Gemma
- Qwen

### Phase 2: Specialized Workers (Future)
**Workers to add:**
1. `whisper-worker` - Speech-to-text
2. `stable-diffusion-worker` - Text-to-image
3. `vision-worker` - Image classification/detection
4. `embedding-worker` - Text embeddings (BERT-based)

## HuggingFace Filter Strategy

### MVP Filters (Phase 1 - LLM Only)

**Primary Filters:**
1. **Model Type** - `text-generation` (already applied)
2. **Format** - GGUF or SafeTensors
3. **Architecture** - LLaMA, Mistral, Phi, Gemma, Qwen
4. **Size** - < 30GB (fits on most GPUs)

**Secondary Filters (UI):**
5. **Quantization** - Q4, Q5, Q8 (for GGUF)
6. **License** - Open source only (apache-2.0, mit, llama2, etc.)
7. **Popularity** - Sort by downloads/likes

### HuggingFace API Query

```typescript
// MVP Query
const query = `https://huggingface.co/api/models?
  search=${query}
  &filter=text-generation
  &library=transformers
  &sort=downloads
  &limit=20`

// Future: Add format filter when HF API supports it
// &filter=gguf OR &filter=safetensors
```

### Frontend Filter UI

**Filter Panel (Left Sidebar):**
```
┌─────────────────────────────┐
│ Filters                     │
├─────────────────────────────┤
│ Format                      │
│ ☑ GGUF (Quantized)         │
│ ☐ SafeTensors (Full)       │
│                             │
│ Architecture                │
│ ☑ LLaMA                    │
│ ☑ Mistral                  │
│ ☑ Phi                      │
│ ☐ Gemma                    │
│ ☐ Qwen                     │
│                             │
│ Size                        │
│ ○ < 5GB  (Small)           │
│ ● < 15GB (Medium)          │
│ ○ < 30GB (Large)           │
│ ○ All                      │
│                             │
│ License                     │
│ ☑ Open Source Only         │
│                             │
│ Sort By                     │
│ ● Downloads                │
│ ○ Likes                    │
│ ○ Recent                   │
└─────────────────────────────┘
```

## Implementation Strategy

### Step 1: Client-Side Filtering (MVP)
Since HuggingFace API doesn't support all filters, we:
1. Fetch results from HF API (text-generation only)
2. Filter client-side by:
   - Model ID patterns (detect architecture from name)
   - Tags (gguf, safetensors, quantized)
   - Size (if available in model card)

### Step 2: Smart Architecture Detection
```typescript
function detectArchitecture(modelId: string, tags: string[]): string[] {
  const architectures = []
  
  // Check model ID
  if (/llama|alpaca|vicuna/i.test(modelId)) architectures.push('llama')
  if (/mistral|mixtral/i.test(modelId)) architectures.push('mistral')
  if (/phi/i.test(modelId)) architectures.push('phi')
  if (/gemma/i.test(modelId)) architectures.push('gemma')
  if (/qwen/i.test(modelId)) architectures.push('qwen')
  
  // Check tags
  tags.forEach(tag => {
    if (tag.includes('llama')) architectures.push('llama')
    if (tag.includes('mistral')) architectures.push('mistral')
    // ... etc
  })
  
  return [...new Set(architectures)]
}
```

### Step 3: Format Detection
```typescript
function detectFormat(modelId: string, tags: string[]): string[] {
  const formats = []
  
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
```

## UI Component Refactor

### Use Existing Components

**From `@rbee/ui/atoms`:**
- `Table` - Model list table
- `Badge` - Format/architecture tags
- `Button` - Download/action buttons
- `Input` - Search box
- `Checkbox` - Filter checkboxes
- `RadioGroup` - Size/sort options
- `Tabs` - View mode switcher
- `Card` - Container
- `Select` - Dropdown filters
- `Skeleton` - Loading states
- `Empty` - No results state
- `Spinner` - Loading spinner

**From `@rbee/ui/molecules`:**
- `FilterButton` - Filter toggle
- `SegmentedControl` - View mode switcher
- `MetricCard` - Stats display
- `ProgressBar` - Download progress
- `StatusKPI` - Model count badges

**From `@rbee/ui/organisms`:**
- `GPUSelector` - Device selection (future)

## Future Expansion

### Phase 2: Multi-Modal Workers

**Whisper Worker:**
- Filter: `pipeline_tag=automatic-speech-recognition`
- Format: SafeTensors
- Architecture: Whisper

**Stable Diffusion Worker:**
- Filter: `pipeline_tag=text-to-image`
- Format: SafeTensors
- Architecture: Stable Diffusion

**Vision Worker:**
- Filter: `pipeline_tag=image-classification`
- Format: SafeTensors
- Architecture: ResNet, ViT, etc.

### Dynamic Worker Detection

```typescript
// Future: Query backend for available workers
const workers = await fetch('/v1/workers/types')
// Returns: ['llm', 'whisper', 'stable-diffusion', 'vision']

// Adjust filters based on available workers
const filters = workers.map(type => getFiltersForWorker(type))
```

## Summary

**MVP Approach:**
1. ✅ Start with GGUF models (quantized, smaller, faster)
2. ✅ Focus on LLaMA/Mistral/Phi (most popular)
3. ✅ Client-side filtering (no backend changes)
4. ✅ Use existing UI components (no custom components)
5. ✅ Smart detection from model ID and tags

**Future:**
- Add SafeTensors support (full precision)
- Add multi-modal workers (Whisper, SD, Vision)
- Backend-side filtering (when scaling)
- Worker capability negotiation
