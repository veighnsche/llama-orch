# Model Management Component

**TEAM-381: Modular, readable component structure**

## Structure

```
ModelManagement/
├── index.tsx                    # Main component (orchestration)
├── types.ts                     # Shared TypeScript types
├── utils.ts                     # Utility functions (detection, filtering)
├── DownloadedModelsView.tsx     # Downloaded models table
├── LoadedModelsView.tsx         # Loaded models table
├── SearchResultsView.tsx        # HuggingFace search results
├── FilterPanel.tsx              # Search filter controls
├── ModelDetailsPanel.tsx        # Model details sidebar
└── README.md                    # This file
```

## Component Responsibilities

### `index.tsx` - Main Component
- **Role:** Orchestration and state management
- **State:** View mode, selected model, search query, filters
- **Responsibilities:**
  - Manage tabs (Downloaded, Loaded, Search)
  - Coordinate between child components
  - Handle model operations (load, unload, delete)
- **Size:** ~170 lines

### `types.ts` - Type Definitions
- **Role:** Shared TypeScript interfaces
- **Exports:**
  - `ViewMode` - Tab selection type
  - `Model` - Local model interface
  - `HFModel` - HuggingFace model interface
  - `FilterState` - Search filter state
- **Size:** ~30 lines

### `utils.ts` - Utility Functions
- **Role:** Pure functions for detection and filtering
- **Functions:**
  - `detectArchitecture()` - Detect model architecture from ID/tags
  - `detectFormat()` - Detect model format (GGUF, SafeTensors)
  - `filterModels()` - Client-side filtering
  - `sortModels()` - Sort by downloads/likes/recent
- **Size:** ~100 lines

### `DownloadedModelsView.tsx` - Downloaded Models
- **Role:** Display models downloaded to disk
- **Features:**
  - Loading skeleton
  - Error state
  - Empty state
  - Model table with actions (Load, Delete)
- **Size:** ~130 lines

### `LoadedModelsView.tsx` - Loaded Models
- **Role:** Display models loaded in RAM
- **Features:**
  - Empty state
  - Model table with VRAM usage
  - Unload action
- **Size:** ~90 lines

### `SearchResultsView.tsx` - HuggingFace Search
- **Role:** Search and display HuggingFace models
- **Features:**
  - Debounced search (500ms)
  - Client-side filtering
  - Loading/error/empty states
  - Architecture and format badges
  - Download action
- **Size:** ~200 lines

### `FilterPanel.tsx` - Search Filters
- **Role:** Filter controls for HuggingFace search
- **Filters:**
  - Format (GGUF, SafeTensors)
  - Architecture (LLaMA, Mistral, Phi, Gemma, Qwen)
  - Max Size (5GB, 15GB, 30GB, All)
  - License (Open Source Only)
  - Sort By (Downloads, Likes, Recent)
- **Size:** ~160 lines

### `ModelDetailsPanel.tsx` - Model Details
- **Role:** Display selected model details and actions
- **Features:**
  - Model metadata (ID, name, size, status)
  - Context-aware actions (Load/Unload/Delete)
  - Loading states
- **Size:** ~80 lines

## Usage

```tsx
import { ModelManagement } from './components/ModelManagement'

function App() {
  return <ModelManagement />
}
```

## Benefits of This Structure

### ✅ Readability
- Each file has a single, clear purpose
- Easy to find and modify specific features
- Self-documenting file names

### ✅ Maintainability
- Changes to one view don't affect others
- Easy to add new views or filters
- Clear separation of concerns

### ✅ Testability
- Each component can be tested in isolation
- Pure utility functions are easy to unit test
- Mock data can be injected via props

### ✅ Reusability
- `FilterPanel` can be reused for other searches
- `utils.ts` functions can be used elsewhere
- Table components follow consistent patterns

### ✅ Performance
- Components only re-render when their props change
- Utility functions are pure (no side effects)
- Debounced search prevents excessive API calls

## Key Features

### Smart Model Detection
```typescript
// Detects architecture from model ID
detectArchitecture("meta-llama/Llama-2-7b", [])
// → ["llama"]

// Detects format from tags
detectFormat("model.gguf", ["gguf", "quantized"])
// → ["gguf"]
```

### Client-Side Filtering
```typescript
// Filter 50 results from HuggingFace API
const filtered = filterModels(results, {
  formats: ['gguf'],
  architectures: ['llama', 'mistral'],
  openSourceOnly: true
})
// → Only GGUF LLaMA/Mistral models
```

### Composition Pattern
```tsx
// Empty states use composition
<Empty>
  <EmptyHeader>
    <EmptyMedia><Icon /></EmptyMedia>
    <EmptyTitle>Title</EmptyTitle>
    <EmptyDescription>Description</EmptyDescription>
  </EmptyHeader>
</Empty>
```

## Future Improvements

### Phase 2: Backend Integration
- [ ] Implement `downloadModel` operation
- [ ] Show download progress with `ProgressBar`
- [ ] Real-time model list updates via SSE
- [ ] Model size estimation for HuggingFace results

### Phase 3: Multi-Modal Workers
- [ ] Add Whisper worker support (speech-to-text)
- [ ] Add Stable Diffusion worker (text-to-image)
- [ ] Add Vision worker (image classification)
- [ ] Dynamic filter panel based on available workers

### Phase 4: Advanced Features
- [ ] Model comparison view
- [ ] Batch operations (download/delete multiple)
- [ ] Model tags and favorites
- [ ] Search history
- [ ] Recommended models based on hardware

## Dependencies

**UI Components:**
- `@rbee/ui/atoms` - Base components (Table, Card, Button, etc.)
- `@rbee/ui/molecules` - Composed components (Empty states)

**Hooks:**
- `@rbee/rbee-hive-react` - Model operations and data fetching

**Icons:**
- `lucide-react` - Icon library

## Testing Strategy

### Unit Tests
- `utils.ts` - Test detection and filtering functions
- Each view component - Test rendering and interactions

### Integration Tests
- Main component - Test tab switching and state management
- Search flow - Test query → filter → results → download

### E2E Tests
- Full workflow: Search → Filter → Download → Load → Inference
- Error handling: Network errors, empty states, invalid models

## Performance Considerations

### Optimizations
- ✅ Debounced search (500ms)
- ✅ Client-side filtering (no backend load)
- ✅ Memoized utility functions
- ✅ Lazy loading of tabs

### Future Optimizations
- [ ] Virtual scrolling for large result sets
- [ ] Pagination for HuggingFace results
- [ ] Cache search results
- [ ] Prefetch model metadata

## Summary

This modular structure makes the Model Management component:
- **Easy to read** - Each file is focused and small
- **Easy to maintain** - Changes are isolated
- **Easy to test** - Components are independent
- **Easy to extend** - New features fit naturally

Total size: ~960 lines split across 8 files (avg ~120 lines/file)
vs. Original monolith: ~800 lines in 1 file
