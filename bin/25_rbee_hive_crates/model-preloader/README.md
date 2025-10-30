# rbee-hive-model-preloader

**Status:** 🚧 STUB - Awaiting implementation  
**Purpose:** Pre-load GGUF models into RAM for faster VRAM transfer

---

## Why This Matters

When spawning a worker, loading a large GGUF model from disk to VRAM can take several seconds. By pre-loading frequently used models into RAM, we can significantly reduce worker startup time:

- **Without pre-loading:** Disk → VRAM (slow, 5-10 seconds for large models)
- **With pre-loading:** RAM → VRAM (fast, 1-2 seconds)

## Architecture

```
Model Pre-loader
    ↓
Load GGUF from disk into RAM (mmap or read)
    ↓
Keep in RAM cache (LRU eviction)
    ↓
When worker spawns, transfer from RAM → VRAM (fast!)
```

## Example Usage (Future)

```rust
use rbee_hive_model_preloader::{ModelPreloader, PreloadConfig};

// Create pre-loader with 16GB RAM cache
let preloader = ModelPreloader::new(PreloadConfig {
    max_cache_size_gb: 16.0,
    eviction_policy: EvictionPolicy::LRU,
    auto_preload_count: 3,
});

// Pre-load frequently used models
preloader.preload("llama-3.2-1b").await?;
preloader.preload("llama-3.2-3b").await?;

// When spawning worker, model is already in RAM
let model_data = preloader.get("llama-3.2-1b").await?;
// Worker can now transfer from RAM → VRAM quickly
```

## Implementation Phases

### Phase 1: Basic Pre-loading
- [ ] Load GGUF file into RAM buffer
- [ ] Simple in-memory cache (HashMap)
- [ ] Get pre-loaded model data

### Phase 2: Cache Management
- [ ] LRU eviction policy
- [ ] Configurable cache size
- [ ] Track access patterns

### Phase 3: Optimizations
- [ ] mmap support
- [ ] madvise hints
- [ ] Async pre-loading
- [ ] Background eviction

### Phase 4: Integration
- [ ] Wire into worker spawn flow
- [ ] Metrics collection
- [ ] Auto pre-load popular models

## Strategies

**1. mmap (Memory-mapped file)**
- OS handles paging
- Lazy loading (only loads pages when accessed)
- Efficient for large files

**2. Read into buffer**
- Explicitly read entire file into RAM
- Full control over memory
- Predictable behavior

**3. Hybrid**
- mmap + madvise(MADV_WILLNEED)
- Hint OS to pre-load pages
- Best of both worlds

## Cache Management

- **LRU eviction** when cache is full
- **Track access patterns** (which models are used most)
- **Configurable cache size** (default: 50% of system RAM)
- **Auto pre-load** top N most-used models at startup

## Integration with Hive

```rust
// In main.rs
let preloader = ModelPreloader::new(PreloadConfig::default());

// Auto pre-load popular models
preloader.auto_preload().await?;

// Add to JobState
let state = JobState {
    registry,
    model_catalog,
    worker_catalog,
    model_preloader: Arc::new(preloader), // ← New!
};

// In worker spawn
if let Some(model_data) = state.model_preloader.get(&model_id).await? {
    // Fast path: model already in RAM
    spawn_worker_with_preloaded_model(model_data).await?;
} else {
    // Slow path: load from disk
    spawn_worker_from_disk(&model_path).await?;
}
```

## Metrics

Track:
- Cache hit rate
- Average load time (with vs without cache)
- RAM usage
- Eviction count
- Most accessed models

## Benefits

**Performance:**
- 3-5x faster worker startup for cached models
- Reduced disk I/O
- Better user experience

**Resource Utilization:**
- Use idle RAM for caching
- Reduce disk wear
- Optimize VRAM transfer

**Scalability:**
- Handle burst worker spawns
- Pre-warm cache during idle time
- Predictable performance

## Current Status

**Stub created with:**
- ✅ Basic types defined (PreloadConfig, EvictionPolicy, CacheStats)
- ✅ API surface designed
- ✅ Documentation written
- ✅ Tests scaffolded
- ⏳ Implementation pending

**Next steps:**
1. Implement basic RAM caching
2. Add LRU eviction
3. Wire into worker spawn flow
4. Add metrics collection
