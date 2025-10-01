# model-cache

**Preloads and caches model weights in system RAM**

Manages RAM-based model weight cache to accelerate VRAM loading. Pre-fetches models from disk to RAM, enabling faster worker startup and model swapping.

**Key responsibilities:**
- Preload model weights into RAM (warm cache)
- LRU eviction when RAM pressure increases
- Fast handoff to worker-orcd for VRAM loading
- Monitor RAM usage and cache hit rates